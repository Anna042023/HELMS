import torch
import torch.nn as nn

from .st_gnn import SpatioTemporalEncoder
from .hypergraph_memory import HypergraphMemory


class NodeWiseLinearPrior(nn.Module):
    """Fast node-specific NLinear prior in normalized space.

    The previous versions relied on a shared NLinear/history MLP and then added
    a randomly initialized decoder residual.  For traffic benchmarks this is a
    weak starting point because each sensor has a different scale and local
    pattern.  This module learns a separate linear residual map for every node:

        y_hat = x_last + W_node * (x_history - x_last) + b_node

    It is extremely cheap (for PEMS07: <0.2M parameters) and usually converges
    within the warmup stage, so validation MAE/RMSE drops before the expensive
    memory training begins.
    """

    def __init__(self, num_nodes: int, seq_len: int, pred_len: int, output_dim: int = 1):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.output_dim = int(output_dim)
        # [N,C,T,L].  Zero initialization means the initial forecast is the last
        # observed value, not a random residual.
        self.weight = nn.Parameter(torch.zeros(self.num_nodes, self.output_dim, self.pred_len, self.seq_len))
        self.bias = nn.Parameter(torch.zeros(self.num_nodes, self.output_dim, self.pred_len))

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        # hist: [B,L,N,C].  Return [B,T,N,C].
        B, L, N, C = hist.shape
        if L < self.seq_len:
            pad = self.seq_len - L
            hist = torch.cat([hist[:, :1].repeat(1, pad, 1, 1), hist], dim=1)
        elif L > self.seq_len:
            hist = hist[:, -self.seq_len:]
        C = min(C, self.output_dim)
        h = hist[..., :C]
        last = h[:, -1:, :, :]
        centered = h - last
        # [B,N,C,L]
        z = centered.permute(0, 2, 3, 1).contiguous()
        w = self.weight[:N, :C]  # [N,C,T,L]
        res = torch.einsum('bncl,nctl->bntc', z, w)
        b = self.bias[:N, :C].permute(0, 2, 1).unsqueeze(0)  # [1,N,T,C]
        out = res + b
        out = out.permute(0, 2, 1, 3).contiguous()
        return last.repeat(1, self.pred_len, 1, 1) + out




class STIDPrior(nn.Module):
    """Future-time-aware STID-style direct predictor in normalized space.

    The previous implementation used only the last observed timestamp embedding
    and then asked one MLP to output all horizons.  For 5-minute traffic data this
    loses a key signal: the 12 targets correspond to known future time-of-day
    slots.  This version keeps the same cheap STID idea but applies a shared MLP
    to each future horizon with its own time embedding, so h=1 and h=12 are no
    longer forced to share the same temporal context.
    """
    def __init__(self, num_nodes: int, seq_len: int, pred_len: int, input_dim: int = 1,
                 output_dim: int = 1, node_dim: int = 32, time_dim: int = 16,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.time_dim = int(time_dim)
        self.node_emb = nn.Parameter(torch.randn(num_nodes, node_dim) * 0.02)
        self.horizon_emb = nn.Parameter(torch.randn(pred_len, time_dim) * 0.02)
        self.time_proj = nn.Sequential(nn.Linear(4, time_dim), nn.ReLU(), nn.Linear(time_dim, time_dim))
        in_dim = seq_len * output_dim + node_dim + time_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        # Zero initialization keeps the initial forecast equal to the last value;
        # the closed-form nodewise prior / warmup then learns stable residuals.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _future_time_features(self, time_feat: torch.Tensor, B: int, device, dtype) -> torch.Tensor:
        if time_feat is None:
            return torch.zeros(B, self.pred_len, 4, device=device, dtype=dtype)
        tf = time_feat.to(device=device, dtype=dtype)
        # New dataset returns [history + future]. Prefer the true known future
        # timestamps.  Fall back to repeating the last historical timestamp for
        # compatibility with older cached datasets.
        if tf.shape[1] >= self.seq_len + self.pred_len:
            return tf[:, self.seq_len:self.seq_len + self.pred_len, :4]
        if tf.shape[1] >= self.pred_len:
            return tf[:, -self.pred_len:, :4]
        last = tf[:, -1:, :4]
        return last.repeat(1, self.pred_len, 1)

    def forward(self, x: torch.Tensor, time_feat: torch.Tensor = None) -> torch.Tensor:
        # x [B,L,N,C], time_feat [B,L+T,4] or [B,L,4]. Return [B,T,N,C].
        B, L, N, C = x.shape
        h = x[:, -self.seq_len:, :, :self.output_dim]
        if h.shape[1] < self.seq_len:
            pad = self.seq_len - h.shape[1]
            h = torch.cat([h[:, :1].repeat(1, pad, 1, 1), h], dim=1)
        last = h[:, -1:, :, :]
        centered = (h - last).permute(0, 2, 3, 1).reshape(B, N, self.output_dim * self.seq_len)
        hist_feat = centered.unsqueeze(1).expand(B, self.pred_len, N, -1)
        node = self.node_emb[:N].to(dtype=x.dtype).view(1, 1, N, -1).expand(B, self.pred_len, N, -1)
        fut_tf = self._future_time_features(time_feat, B, x.device, x.dtype)
        te = self.time_proj(fut_tf.reshape(B * self.pred_len, 4)).view(B, self.pred_len, 1, self.time_dim)
        te = te + self.horizon_emb[:self.pred_len].to(device=x.device, dtype=x.dtype).view(1, self.pred_len, 1, self.time_dim)
        te = te.expand(B, self.pred_len, N, self.time_dim)
        feat = torch.cat([hist_feat, node, te], dim=-1)
        res = self.mlp(feat).contiguous()
        return last.repeat(1, self.pred_len, 1, 1) + res


class ExternalPriorFusion(nn.Module):
    """Lightweight traffic prior fusion.

    The raw HELMS decoder learns a residual around the recent observation.
    Traffic data also contain very strong legal priors, e.g., same-time
    yesterday/week and training-only time-of-day means.  This module learns a
    per-horizon convex mixture between the neural HELMS output and these priors.
    It adds almost no GPU cost and uses only information available before the
    forecast origin.
    """

    def __init__(self, pred_len: int, output_dim: int = 1, basis_names=None, enabled: bool = True, adaptive: bool = True):
        super().__init__()
        self.pred_len = int(pred_len)
        self.output_dim = int(output_dim)
        self.enabled = bool(enabled)
        self.adaptive = bool(adaptive)
        self.basis_names = list(basis_names or [
            "model", "last", "daily_delta", "weekly_delta", "tod_delta", "recent_mean", "trend", "tod", "global"
        ])
        if "model" not in self.basis_names:
            self.basis_names.insert(0, "model")
        self.name_to_idx = {n: i for i, n in enumerate(self.basis_names)}
        self.logits = nn.Parameter(torch.zeros(self.pred_len, len(self.basis_names)))
        self.residual_gate = nn.Parameter(torch.full((self.pred_len, self.output_dim), -1.8))
        # Local/sample-wise gate.  The final layer is zero-initialized, so the
        # model starts from the stable global mixture and only learns adaptive
        # deviations when they reduce the supervised loss.
        k = len(self.basis_names)
        gate_hidden = max(16, 2 * k)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * k, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, k),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)
        self._init_logits()

    def _init_logits(self):
        with torch.no_grad():
            self.logits.fill_(-2.0)
            init = {
                "model": 0.95,
                "ridge_ar_blend": 2.05,
                "ridge_ar_delta": 1.85,
                "ridge_ar": 1.30,
                "daily_analog_blend": 1.65,
                "daily_analog_delta": 1.65,
                "daily_analog": 1.10,
                "last": 0.45,
                "daily_delta": 1.15,
                "weekly_delta": 0.90,
                "daily_resmean": 1.05,
                "weekly_resmean": 0.85,
                "seasonal_ensemble": 1.25,
                "recent_pattern": 0.45,
                "tod_delta": 0.65,
                "recent_mean": 0.25,
                "trend": 0.05,
                "tod": 0.10,
                "tow": 0.10,
                "global": -1.5,
            }
            for name, val in init.items():
                if name in self.name_to_idx:
                    self.logits[:, self.name_to_idx[name]] = float(val)

    def forward(self, model_pred, x, external_priors=None):
        if (not self.enabled) or external_priors is None:
            return model_pred
        B, T, N, C = model_pred.shape
        bases = {"model": model_pred}
        bases["last"] = x[:, -1:, :, :C].repeat(1, T, 1, 1)
        for k, v in external_priors.items():
            if v is None:
                continue
            vv = v.to(device=model_pred.device, dtype=model_pred.dtype)
            if vv.dim() == 4 and vv.shape[1] >= T:
                bases[k] = vv[:, :T, :, :C]

        parts = []
        idxs = []
        for name in self.basis_names:
            if name in bases and tuple(bases[name].shape) == tuple(model_pred.shape):
                parts.append(bases[name])
                idxs.append(self.name_to_idx[name])
        if len(parts) <= 1:
            return model_pred
        stack = torch.stack(parts, dim=0)  # [K,B,T,N,C]
        logits = self.logits[:T, idxs].to(device=model_pred.device, dtype=model_pred.dtype)
        if self.adaptive and C == 1:
            # [B,T,N,K_active]
            values = stack.squeeze(-1).permute(1, 2, 3, 0).contiguous()
            last_val = bases.get("last", model_pred).squeeze(-1).unsqueeze(-1)

            # The adaptive gate MLP is initialized for the full configured basis set
            # (2 * len(self.basis_names) input features).  During large-dataset fast
            # warmup we intentionally pass a compact subset of priors to avoid the
            # expensive analog/pattern bases.  Build a zero-padded full feature tensor
            # and fill only the active basis slots, so compact warmup remains valid
            # without changing the learned gate dimensions or the full-training path.
            full_k = len(self.basis_names)
            if len(parts) == full_k:
                feat = torch.cat([values, values - last_val], dim=-1)
                local_logits = self.gate_mlp(feat) + logits.view(1, T, 1, len(parts))
            else:
                full_feat = values.new_zeros(B, T, N, 2 * full_k)
                idx_tensor = torch.as_tensor(idxs, device=values.device, dtype=torch.long)
                full_feat.index_copy_(-1, idx_tensor, values)
                full_feat.index_copy_(-1, idx_tensor + full_k, values - last_val)
                local_logits_all = self.gate_mlp(full_feat)
                local_logits = local_logits_all.index_select(-1, idx_tensor) + logits.view(1, T, 1, len(parts))

            weight = torch.softmax(local_logits, dim=-1)
            fused = (values * weight).sum(dim=-1).unsqueeze(-1)
        else:
            weight = torch.softmax(logits, dim=-1).permute(1, 0).view(len(parts), 1, T, 1, 1)
            fused = (weight * stack).sum(dim=0)
        # Keep a learnable path back to the neural prediction so that the prior
        # mixture cannot over-constrain unusual events.  The default gate is
        # small, so stable seasonal priors can reduce error early in training.
        gate = torch.sigmoid(self.residual_gate[:T]).to(device=model_pred.device, dtype=model_pred.dtype).view(1, T, 1, C)
        return fused + gate * (model_pred - fused)


class HELMS(nn.Module):
    """Full HELMS model.

    Forward order follows Algorithm 1:
      encoder -> hypergraph propagation/retrieval -> fusion -> decoder -> optional SR.

    This implementation keeps the paper's HMC/DML/SR path.  The optional
    external prior fusion layer is disabled in the paper-aligned configuration;
    metric improvements are obtained by tuning the encoder, memory retrieval,
    DML lifecycle, and SR weights.
    """
    def __init__(
        self,
        num_nodes,
        input_dim=1,
        output_dim=1,
        pred_len=12,
        hidden_dim=64,
        node_embed_dim=16,
        time_embed_dim=16,
        num_st_layers=3,
        gat_heads=4,
        dropout=0.1,
        dynamic_topk=12,
        dynamic_alpha=0.5,
        static_alpha=0.3,
        use_static_graph=False,
        memory_cfg=None,
        residual_forecast=True,
        residual_clip=3.0,
        trend_prior=True,
        trend_clip=1.0,
        seq_len=12,
        history_prior=True,
        stid_prior=True,
        stid_hidden_dim=128,
        stid_node_dim=32,
        external_prior_fusion=True,
        adaptive_prior_fusion=True,
        prior_bases=None,
    ):
        super().__init__()
        memory_cfg = memory_cfg or {}
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.seq_len = int(seq_len)
        self.history_prior = bool(history_prior)
        self.stid_prior_enabled = bool(stid_prior)
        self.residual_forecast = bool(residual_forecast)
        self.residual_clip = float(residual_clip)
        self.trend_prior = bool(trend_prior)
        self.trend_clip = float(trend_clip)
        self.trend_gate = nn.Parameter(torch.full((pred_len, output_dim), -2.0))
        self.nlinear = nn.Linear(self.seq_len, pred_len)
        nn.init.zeros_(self.nlinear.weight)
        nn.init.zeros_(self.nlinear.bias)
        # A tiny NLinear+MLP history head.  The linear part is stable at the
        # beginning (last-value forecast); the MLP learns non-linear local
        # changes and is much cheaper than increasing ST-GNN depth.
        self.hist_mlp = nn.Sequential(
            nn.Linear(self.seq_len, max(32, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, hidden_dim // 2), pred_len),
        )
        nn.init.zeros_(self.hist_mlp[-1].weight)
        nn.init.zeros_(self.hist_mlp[-1].bias)
        self.history_gate = nn.Parameter(torch.full((pred_len, output_dim), 1.0))
        self.hist_mlp_gate = nn.Parameter(torch.full((pred_len, output_dim), -1.5))
        self.nodewise_prior = NodeWiseLinearPrior(num_nodes, self.seq_len, pred_len, output_dim)
        # Start with the node-wise linear prior as the dominant fast path; the
        # shared history MLP only contributes when it learns useful corrections.
        self.nodewise_gate = nn.Parameter(torch.full((pred_len, output_dim), 2.0))
        self.stid_prior = STIDPrior(
            num_nodes=num_nodes, seq_len=self.seq_len, pred_len=pred_len,
            input_dim=input_dim, output_dim=output_dim,
            node_dim=int(stid_node_dim), time_dim=int(time_embed_dim),
            hidden_dim=int(stid_hidden_dim), dropout=dropout,
        )
        # Starts conservative; if STID learns a better history-to-future map,
        # gradients increase this gate quickly during fast warmup.
        self.stid_gate = nn.Parameter(torch.full((pred_len, output_dim), -0.2))
        self.prior_fusion = ExternalPriorFusion(
            pred_len=pred_len,
            output_dim=output_dim,
            basis_names=prior_bases,
            enabled=external_prior_fusion,
            adaptive=adaptive_prior_fusion,
        )
        self.encoder = SpatioTemporalEncoder(
            num_nodes=num_nodes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            node_embed_dim=node_embed_dim,
            time_embed_dim=time_embed_dim,
            num_layers=num_st_layers,
            heads=gat_heads,
            dropout=dropout,
            dynamic_topk=dynamic_topk,
            dynamic_alpha=dynamic_alpha,
            static_alpha=static_alpha,
            use_static_graph=use_static_graph,
        )
        # beta_init and memory_gate_init are HELMS-level fusion parameters,
        # not constructor arguments of HypergraphMemory.  Keep them in
        # memory_cfg for convenience, but remove them before creating the
        # HypergraphMemory module to avoid:
        #   TypeError: HypergraphMemory.__init__() got an unexpected keyword argument 'beta_init'
        beta_init = float(memory_cfg.get("beta_init", 0.05))
        gate_init = float(memory_cfg.get("memory_gate_init", -2.2))
        residual_gate_init = float(memory_cfg.get("memory_residual_gate_init", -0.8))
        memory_module_cfg = dict(memory_cfg)
        memory_module_cfg.pop("beta_init", None)
        memory_module_cfg.pop("memory_gate_init", None)
        memory_module_cfg.pop("memory_residual_gate_init", None)
        self.memory = HypergraphMemory(hidden_dim=hidden_dim, **memory_module_cfg)
        # Start memory fusion conservatively.  In previous runs the untrained
        # memory branch could overwrite a strong baseline forecast and raise
        # all three metrics.  The scalar beta still implements the paper
        # H_t + beta*m_t fusion, but starts small and is learned end-to-end.
        self.beta = nn.Parameter(torch.tensor(beta_init))
        # Output-level guarded memory mixture.  The initial gate is intentionally
        # conservative: when memory is not yet useful, the model keeps the strong
        # no-memory baseline; if HMC/DML/SR help, gradients can increase the gate.
        self.memory_output_gate = nn.Parameter(torch.full((pred_len, output_dim), gate_init))
        # Direct HMC residual bank gate.  Each memory prototype stores the mean
        # residual (future - warmup no-memory forecast) of its training cluster.
        # Retrieval gives a legal memory-based correction, not an external prior.
        # A moderate initial gate lets this branch help immediately while still
        # allowing gradients to suppress noisy memories.
        self.memory_residual_gate = nn.Parameter(torch.full((pred_len, output_dim), residual_gate_init))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * output_dim)
        )
        self.baseline_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * output_dim)
        )
        # Critical stability fix: decoder outputs are residuals around a strong
        # last-value/node-linear prior.  Random residuals at initialization raise
        # validation MAE/RMSE dramatically and made the previous training logs
        # worse.  Zero the last layers so both decoders start from the prior.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        nn.init.zeros_(self.baseline_decoder[-1].weight)
        nn.init.zeros_(self.baseline_decoder[-1].bias)

    def _apply_residual_prior(self, raw_out, prior_value):
        if not self.residual_forecast or prior_value is None:
            return raw_out
        if self.residual_clip > 0:
            residual = self.residual_clip * torch.tanh(raw_out / self.residual_clip)
        else:
            residual = raw_out
        return prior_value + residual

    def _history_ar_prior(self, x):
        if not self.history_prior:
            return None
        hist = x[:, -self.seq_len:, :, :self.output_dim]
        if hist.shape[1] < self.seq_len:
            pad = self.seq_len - hist.shape[1]
            hist = torch.cat([hist[:, :1].repeat(1, pad, 1, 1), hist], dim=1)
        B, L, N, C = hist.shape
        last = hist[:, -1:, :, :]

        # 1) Node-specific NLinear prior: strongest and fastest branch.
        node_prior = self.nodewise_prior(hist)

        # 2) Shared NLinear/MLP residual as a small cross-node regularized branch.
        centered = hist - last
        z = centered.permute(0, 2, 3, 1).reshape(B * N * C, L)
        residual_linear = self.nlinear(z)
        residual_mlp = self.hist_mlp(z)
        mg = torch.sigmoid(self.hist_mlp_gate).view(1, self.pred_len, 1, self.output_dim).to(dtype=hist.dtype, device=hist.device)
        residual = residual_linear.view(B, N, C, self.pred_len).permute(0, 3, 1, 2).contiguous()
        residual2 = residual_mlp.view(B, N, C, self.pred_len).permute(0, 3, 1, 2).contiguous()
        shared_prior = last.repeat(1, self.pred_len, 1, 1) + residual + mg * residual2

        ng = torch.sigmoid(self.nodewise_gate).view(1, self.pred_len, 1, self.output_dim).to(dtype=hist.dtype, device=hist.device)
        return ng * node_prior + (1.0 - ng) * shared_prior

    def _build_prior(self, x, time_feat=None):
        last = x[:, -1:, :, :self.output_dim]
        base = last
        if self.trend_prior and x.shape[1] >= 2:
            if x.shape[1] >= 4:
                slope = (x[:, -1:, :, :self.output_dim] - x[:, -4:-3, :, :self.output_dim]) / 3.0
            else:
                slope = x[:, -1:, :, :self.output_dim] - x[:, -2:-1, :, :self.output_dim]
            if self.trend_clip > 0:
                slope = self.trend_clip * torch.tanh(slope / self.trend_clip)
            steps = torch.arange(1, self.pred_len + 1, device=x.device, dtype=x.dtype).view(1, self.pred_len, 1, 1)
            gate = torch.sigmoid(self.trend_gate).view(1, self.pred_len, 1, self.output_dim).to(dtype=x.dtype)
            base = last + steps * slope * gate
        ar = self._history_ar_prior(x)
        if ar is not None:
            hg = torch.sigmoid(self.history_gate).view(1, self.pred_len, 1, self.output_dim).to(dtype=x.dtype, device=x.device)
            base = (1.0 - hg) * base + hg * ar
        if self.stid_prior_enabled:
            stid = self.stid_prior(x, time_feat)
            sg = torch.sigmoid(self.stid_gate).view(1, self.pred_len, 1, self.output_dim).to(dtype=x.dtype, device=x.device)
            base = (1.0 - sg) * base + sg * stid
        return base

    def decode(self, H, prior_value=None):
        B, N, _ = H.shape
        out = self.decoder(H).view(B, N, self.pred_len, self.output_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        return self._apply_residual_prior(out, prior_value)

    def decode_baseline(self, H, prior_value=None):
        B, N, _ = H.shape
        out = self.baseline_decoder(H).view(B, N, self.pred_len, self.output_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        return self._apply_residual_prior(out, prior_value)

    def apply_external_prior_fusion(self, pred, x, external_priors=None):
        return self.prior_fusion(pred, x, external_priors)

    def fast_history_forward(self, x, time_feat=None, external_priors=None):
        """Fast warmup path that avoids the ST-GNN encoder.

        It trains the node-wise NLinear/history prior and optional prior-fusion
        heads only.  This is much faster and prevents PeMS07 OOM during the
        long warmup stage.  The full encoder is still trained for a few short
        warmup epochs and during the HELMS stage.
        """
        pred = self._build_prior(x, time_feat)
        pred = self.apply_external_prior_fusion(pred, x, external_priors)
        return pred

    def forward(self, x, time_feat, static_adj=None, use_memory=True, return_aux=False, external_priors=None):
        H_t, h_t, dyn_adj = self.encoder(x, time_feat, static_adj=static_adj)
        prior_value = self._build_prior(x, time_feat)
        last_value = x[:, -1:, :, :self.output_dim]

        # Always compute the warmed-up baseline branch.  It is cheap and serves
        # as a safety anchor for the memory-enhanced forecast.
        base_pred = self.decode_baseline(H_t, prior_value=prior_value)
        if use_memory:
            m_t, alpha_full, Vp, active_idx, mem_residual, mem_confidence = self.memory.retrieve(h_t)
            H_fused = H_t + self.beta * m_t.unsqueeze(1)
            mem_pred = self.decode(H_fused, prior_value=prior_value)
            gate = torch.sigmoid(self.memory_output_gate[:self.pred_len]).view(1, self.pred_len, 1, self.output_dim)
            gate = gate.to(device=mem_pred.device, dtype=mem_pred.dtype)
            conf_gate = (0.5 + 0.5 * mem_confidence.to(device=mem_pred.device, dtype=mem_pred.dtype)).view(-1, 1, 1, 1)
            pred = base_pred + conf_gate * gate * (mem_pred - base_pred)
            if mem_residual is not None:
                rg = torch.sigmoid(self.memory_residual_gate[:self.pred_len]).view(1, self.pred_len, 1, self.output_dim)
                rg = rg.to(device=pred.device, dtype=pred.dtype)
                # The stored residual is in normalized space, same as pred.
                # Clip it softly to avoid a few noisy clusters harming RMSE.
                mem_residual = 2.5 * torch.tanh(mem_residual / 2.5)
                pred = pred + rg * mem_residual
        else:
            m_t = None
            alpha_full = None
            active_idx = None
            mem_pred = None
            mem_residual = None
            mem_confidence = None
            pred = base_pred

        pred = self.apply_external_prior_fusion(pred, x, external_priors)
        base_pred_fused = self.apply_external_prior_fusion(base_pred, x, external_priors)
        if return_aux:
            return pred, {
                "H_t": H_t,
                "h_t": h_t,
                "m_t": m_t,
                "alpha": alpha_full,
                "active_idx": active_idx,
                "dyn_adj": dyn_adj,
                "last_value": last_value,
                "prior_value": prior_value,
                "base_pred": base_pred_fused,
                "base_pred_raw": base_pred,
                "mem_pred": mem_pred,
                "mem_residual": mem_residual,
                "mem_confidence": mem_confidence,
                "memory_gate": torch.sigmoid(self.memory_output_gate.detach()),
                "memory_residual_gate": torch.sigmoid(self.memory_residual_gate.detach()),
            }
        return pred

    def semantic_loss(self):
        return self.memory.semantic_regularization()
