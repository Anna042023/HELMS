from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphMemory(nn.Module):
    """HMC + hypergraph propagation + lifecycle state.

    This version fixes two practical issues that can make metrics poor:
      1) online hypergraph refresh previously discarded temporal/co-occurrence
         edges created during initialization;
      2) memory retrieval used unnormalized dense attention, often making the
         memory response too diffuse.
    """
    def __init__(self, hidden_dim=64, init_memory_size=200, max_memory_size=400,
                 semantic_dim=64, hyper_neighbors=5, transition_threshold=0.3,
                 cooc_topk=5, cooc_threshold=5, sr_pairs=1000, temperature=0.35,
                 retrieve_topk=0, normalize_retrieval=False, utility_retrieval_weight=0.15,
                 residual_confidence_floor=0.45, residual_confidence_power=1.0,
                 residual_shrink_count=8.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.init_memory_size = init_memory_size
        self.max_memory_size = max_memory_size
        self.semantic_dim = semantic_dim
        self.hyper_neighbors = hyper_neighbors
        self.transition_threshold = transition_threshold
        self.cooc_topk = cooc_topk
        self.cooc_threshold = cooc_threshold
        self.sr_pairs = sr_pairs
        self.temperature = temperature
        self.retrieve_topk = retrieve_topk
        self.normalize_retrieval = bool(normalize_retrieval)
        self.utility_retrieval_weight = float(utility_retrieval_weight)
        self.residual_confidence_floor = float(residual_confidence_floor)
        self.residual_confidence_power = float(residual_confidence_power)
        self.residual_shrink_count = float(residual_shrink_count)

        self.prototypes = nn.Parameter(torch.zeros(max_memory_size, hidden_dim))
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_gate = nn.Linear(hidden_dim, 1, bias=False)
        self.propagation_scale = nn.Parameter(torch.tensor(0.5))

        self.register_buffer("active_mask", torch.zeros(max_memory_size, dtype=torch.bool))
        self.register_buffer("utility", torch.zeros(max_memory_size))
        self.register_buffer("last_access", torch.zeros(max_memory_size, dtype=torch.long))
        self.register_buffer("access_count", torch.zeros(max_memory_size))
        self.register_buffer("core_mask", torch.zeros(max_memory_size, dtype=torch.bool))
        self.register_buffer("semantic_embeddings", torch.zeros(max_memory_size, semantic_dim))
        # Cluster-level future residual/value prototypes. Shape is initialized
        # after HMC because pred_len/num_nodes are dataset dependent.  The
        # residual bank stores mean (future - no-memory baseline) for each
        # memory cluster, allowing memory retrieval to contribute a direct
        # data-driven correction while staying inside HELMS/HMC.
        self.register_buffer("forecast_residuals", torch.empty(0))
        self.register_buffer("residual_counts", torch.zeros(max_memory_size))
        self.register_buffer("incidence", torch.zeros(max_memory_size, 1))
        self.tags = [""] * max_memory_size
        self.descriptions = [""] * max_memory_size
        self.current_step = 0
        self._cached_transition_edges = []
        self._cached_cooc_edges = []

    @property
    def active_indices(self):
        return torch.where(self.active_mask)[0]

    @property
    def active_k(self):
        return int(self.active_mask.sum().item())

    @torch.no_grad()
    def initialize(self, centers: torch.Tensor, semantic_embeddings: torch.Tensor,
                   tags=None, descriptions=None, assignment_labels: Optional[torch.Tensor] = None,
                   states: Optional[torch.Tensor] = None, forecast_residuals: Optional[torch.Tensor] = None,
                   residual_counts: Optional[torch.Tensor] = None):
        k = min(centers.shape[0], self.max_memory_size)
        self.prototypes.data.zero_()
        self.prototypes.data[:k].copy_(centers[:k].to(self.prototypes.device))
        self.active_mask.zero_(); self.active_mask[:k] = True
        self.utility.zero_(); self.utility[:k] = 0.5
        self.last_access.zero_(); self.access_count.zero_(); self.core_mask.zero_()
        self.semantic_embeddings.zero_()
        sem = semantic_embeddings[:k]
        if sem.shape[-1] != self.semantic_dim:
            raise ValueError(f"semantic dim mismatch {sem.shape[-1]} != {self.semantic_dim}")
        self.semantic_embeddings[:k].copy_(sem.to(self.semantic_embeddings.device))
        if tags is not None:
            for i in range(k):
                self.tags[i] = tags[i]
        if descriptions is not None:
            for i in range(k):
                self.descriptions[i] = descriptions[i]
        self.residual_counts.zero_()
        if forecast_residuals is not None:
            # forecast_residuals: [K,T,N,C] in normalized space.  Allocate a
            # max-size bank so newly-created memories can be appended safely.
            # Small clusters are noisy, so shrink their correction toward zero.
            fr = forecast_residuals[:k].to(device=self.prototypes.device, dtype=self.prototypes.dtype)
            if residual_counts is None:
                cnt = torch.ones(k, device=self.prototypes.device, dtype=self.prototypes.dtype)
            else:
                cnt = residual_counts[:k].to(device=self.prototypes.device, dtype=self.prototypes.dtype).clamp_min(1.0)
            shrink = torch.sqrt(cnt / (cnt + self.residual_shrink_count)).view(k, *([1] * (fr.dim() - 1)))
            fr = fr * shrink
            shape = (self.max_memory_size,) + tuple(fr.shape[1:])
            bank = torch.zeros(shape, device=self.prototypes.device, dtype=self.prototypes.dtype)
            bank[:k].copy_(fr)
            self.forecast_residuals = bank
            self.residual_counts[:k].copy_(cnt)
        else:
            self.forecast_residuals = torch.empty(0, device=self.prototypes.device, dtype=self.prototypes.dtype)
        self._cached_transition_edges = []
        self._cached_cooc_edges = []
        self.rebuild_hypergraph(assignment_labels=assignment_labels, states=states, update_cache=True)

    @staticmethod
    def _edge_key(edge):
        return tuple(sorted(set(int(x) for x in edge)))

    @torch.no_grad()
    def rebuild_hypergraph(self, assignment_labels: Optional[torch.Tensor] = None,
                           states: Optional[torch.Tensor] = None,
                           update_cache: bool = False):
        idx = self.active_indices
        k = len(idx)
        device = self.prototypes.device
        if k == 0:
            self.incidence = torch.zeros(self.max_memory_size, 1, device=device)
            return
        active_set = set(int(i) for i in idx.detach().cpu().tolist())
        V = F.normalize(self.prototypes.data[idx], dim=-1)
        edges = []

        # Similarity hyperedges: refreshed online from current prototypes.
        sim = V @ V.t()
        sim.fill_diagonal_(-1e9)
        m = min(self.hyper_neighbors, max(1, k - 1))
        nbr = torch.topk(sim, k=m, dim=-1).indices
        for i in range(k):
            e = [int(idx[i])]
            e += [int(idx[j]) for j in nbr[i]]
            edges.append(self._edge_key(e))

        new_transition_edges = []
        new_cooc_edges = []

        # Temporal transition hyperedges P(m_j | m_i) above threshold.
        if assignment_labels is not None and len(assignment_labels) > 1:
            labels = assignment_labels.detach().cpu().long()
            valid = (labels >= 0) & (labels < k)
            labels = labels[valid]
            if len(labels) > 1:
                counts = torch.zeros(k, k)
                for a, b in zip(labels[:-1], labels[1:]):
                    counts[int(a), int(b)] += 1
                denom = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
                prob = counts / denom
                trans = torch.where(prob > self.transition_threshold)
                for a, b in zip(trans[0].tolist(), trans[1].tolist()):
                    if a != b:
                        edge = self._edge_key([int(idx[a]), int(idx[b])])
                        edges.append(edge)
                        new_transition_edges.append(edge)

        # Co-occurrence hyperedges: top activated prototypes in same state.
        if states is not None and len(states) > 0:
            states = F.normalize(states.to(device), dim=-1)
            scores = states @ V.t()
            kk = min(self.cooc_topk, k)
            top = torch.topk(scores, k=kk, dim=-1).indices.detach().cpu()
            pair_counts = {}
            for row in top:
                row = row.tolist()
                for a_pos in range(len(row)):
                    for b_pos in range(a_pos + 1, len(row)):
                        a, b = sorted((row[a_pos], row[b_pos]))
                        pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
            for (a, b), cnt in pair_counts.items():
                if cnt >= self.cooc_threshold:
                    edge = self._edge_key([int(idx[a]), int(idx[b])])
                    edges.append(edge)
                    new_cooc_edges.append(edge)

        if update_cache:
            # Incremental online update: merge new temporal/co-occurrence edges
            # with previous valid relational edges instead of discarding them.
            trans = list(self._cached_transition_edges) + list(new_transition_edges)
            cooc = list(self._cached_cooc_edges) + list(new_cooc_edges)
            self._cached_transition_edges = []
            self._cached_cooc_edges = []
            seen_t, seen_c = set(), set()
            for edge in trans:
                if all(int(v) in active_set for v in edge) and edge not in seen_t:
                    seen_t.add(edge); self._cached_transition_edges.append(edge)
            for edge in cooc:
                if all(int(v) in active_set for v in edge) and edge not in seen_c:
                    seen_c.add(edge); self._cached_cooc_edges.append(edge)
            # Bound cache size for memory/time stability.
            self._cached_transition_edges = self._cached_transition_edges[-4 * self.max_memory_size:]
            self._cached_cooc_edges = self._cached_cooc_edges[-4 * self.max_memory_size:]
        # Keep relational edges during online similarity refresh.
        for edge in self._cached_transition_edges + self._cached_cooc_edges:
            if all(int(v) in active_set for v in edge):
                edges.append(edge)

        uniq, seen = [], set()
        for e in edges:
            if len(e) > 0 and e not in seen:
                seen.add(e)
                uniq.append(list(e))
        if not uniq:
            uniq = [[int(i)] for i in idx]
        H = torch.zeros(self.max_memory_size, len(uniq), device=device)
        for col, e in enumerate(uniq):
            H[e, col] = 1.0
        self.incidence = H

    def propagate(self):
        idx = self.active_indices
        if len(idx) == 0:
            raise RuntimeError("Memory is not initialized.")
        V_all = self.prototypes
        H = self.incidence.to(V_all.device, V_all.dtype)
        if H.numel() == 0:
            return V_all[idx]
        active_float = self.active_mask.to(V_all.dtype).unsqueeze(1)
        H = H * active_float
        edge_degree = H.sum(dim=0).clamp_min(1.0)
        edge_feat = (H.t() @ V_all) / edge_degree.unsqueeze(-1)
        edge_weight = torch.sigmoid(self.edge_gate(edge_feat)).squeeze(-1)
        H_weighted = H * edge_weight.unsqueeze(0)
        node_degree = H_weighted.sum(dim=1).clamp_min(1.0)
        msg = (H_weighted @ edge_feat) / node_degree.unsqueeze(-1)
        msg = torch.relu(self.Wv(msg))
        # Residual propagation preserves prototype identity while adding context.
        scale = torch.clamp(self.propagation_scale, 0.0, 1.0)
        out = F.layer_norm(V_all + scale * msg, (self.hidden_dim,))
        return out[idx]

    def retrieve(self, h_t):
        idx = self.active_indices
        Vp = self.propagate()
        if self.normalize_retrieval:
            # Optional stable cosine retrieval.  The paper formula uses h^T v;
            # this branch is kept as a fallback for very unstable runs.
            h = F.normalize(h_t, dim=-1, eps=1e-6)
            v = F.normalize(Vp, dim=-1, eps=1e-6)
            scores = h @ v.t()
        else:
            # Paper-aligned attention score: h_t^T v'_k / tau over all memories.
            scores = h_t @ Vp.t()
        scores = scores / max(float(self.temperature), 1e-6)
        if self.utility_retrieval_weight > 0 and self.utility.numel() > 0:
            util = self.utility[idx].to(device=h_t.device, dtype=h_t.dtype).clamp(0.0, 1.0)
            # DML utility should bias retrieval only gently; it should not override
            # content similarity.  Centering at 0.5 keeps initialized memories neutral.
            scores = scores + self.utility_retrieval_weight * (util - 0.5).view(1, -1)
        topk = int(self.retrieve_topk or 0)
        if topk > 0 and topk < scores.shape[-1]:
            keep_idx = torch.topk(scores, k=topk, dim=-1).indices
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask.scatter_(dim=-1, index=keep_idx, value=False)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        alpha_active = torch.softmax(scores, dim=-1)
        m_t = alpha_active @ Vp
        if alpha_active.shape[-1] > 1:
            entropy = -(alpha_active * torch.log(alpha_active.clamp_min(1e-8))).sum(dim=-1)
            max_entropy = torch.log(torch.tensor(float(alpha_active.shape[-1]), device=h_t.device, dtype=h_t.dtype))
            confidence = (1.0 - entropy / max_entropy.clamp_min(1e-6)).clamp(0.0, 1.0)
        else:
            confidence = torch.ones(h_t.shape[0], device=h_t.device, dtype=h_t.dtype)
        forecast_residual = None
        if self.forecast_residuals.numel() > 0:
            # [B,K] x [K,T,N,C] -> [B,T,N,C].  This is the actual long-term
            # memory reuse path: each retrieved prototype contributes its
            # average future residual learned from the training cluster.
            fr = self.forecast_residuals[idx].to(device=h_t.device, dtype=h_t.dtype)
            forecast_residual = torch.einsum('bk,ktnc->btnc', alpha_active, fr)
            conf_scale = self.residual_confidence_floor + (1.0 - self.residual_confidence_floor) * confidence
            conf_scale = conf_scale.clamp(0.0, 1.0).pow(self.residual_confidence_power)
            forecast_residual = forecast_residual * conf_scale.view(-1, 1, 1, 1)
        alpha_full = torch.zeros(h_t.shape[0], self.max_memory_size, device=h_t.device, dtype=h_t.dtype)
        alpha_full[:, idx] = alpha_active
        return m_t, alpha_full, Vp, idx, forecast_residual, confidence

    def semantic_regularization(self):
        idx = self.active_indices
        k = len(idx)
        if k < 2:
            return self.prototypes.sum() * 0.0
        V = F.normalize(self.prototypes[idx], dim=-1)
        S = F.normalize(self.semantic_proj(self.semantic_embeddings[idx].to(V.device, V.dtype)), dim=-1)
        sim_v = V @ V.t()
        sim_s = S @ S.t()
        triu = torch.triu_indices(k, k, offset=1, device=V.device)
        pair_scores = sim_v[triu[0], triu[1]].abs()
        m = min(self.sr_pairs, pair_scores.numel())
        if m <= 0:
            return self.prototypes.sum() * 0.0
        top = torch.topk(pair_scores, k=m).indices
        i = triu[0][top]
        j = triu[1][top]
        return torch.mean(torch.abs(sim_v[i, j] - sim_s[i, j]))

    @torch.no_grad()
    def update_access(self, alpha_full, epoch: int):
        contrib = alpha_full.mean(dim=0)
        active_mean = 1.0 / max(1, self.active_k)
        used = contrib > (0.5 * active_mean)
        self.access_count += contrib.detach()
        self.last_access[used] = epoch

    @torch.no_grad()
    def add_memory(self, proto: torch.Tensor, semantic: Optional[torch.Tensor] = None,
                   tag: str = "new unseen pattern", description: str = "Created online for an unseen traffic pattern.",
                   last_access_epoch: int = 0, forecast_residual: Optional[torch.Tensor] = None):
        inactive = torch.where(~self.active_mask)[0]
        if len(inactive) == 0:
            return False
        pos = int(inactive[0])
        self.prototypes.data[pos].copy_(proto.detach().to(self.prototypes.device))
        self.active_mask[pos] = True
        self.utility[pos] = 0.5
        self.last_access[pos] = int(last_access_epoch)
        self.access_count[pos] = 0
        self.core_mask[pos] = False
        if semantic is None:
            semantic = torch.zeros(self.semantic_dim, device=self.semantic_embeddings.device)
            semantic[0] = 1.0
        self.semantic_embeddings[pos].copy_(semantic.to(self.semantic_embeddings.device))
        self.tags[pos] = tag
        self.descriptions[pos] = description
        if forecast_residual is not None:
            fr = forecast_residual.detach().to(device=self.prototypes.device, dtype=self.prototypes.dtype)
            if self.forecast_residuals.numel() == 0:
                shape = (self.max_memory_size,) + tuple(fr.shape)
                self.forecast_residuals = torch.zeros(shape, device=self.prototypes.device, dtype=self.prototypes.dtype)
            if tuple(self.forecast_residuals.shape[1:]) == tuple(fr.shape):
                self.forecast_residuals[pos].copy_(fr)
                self.residual_counts[pos] = 1.0
        self.rebuild_hypergraph()
        return True

    @torch.no_grad()
    def deactivate(self, remove_idx):
        if len(remove_idx) == 0:
            return
        self.active_mask[remove_idx] = False
        self.utility[remove_idx] = 0.0
        self.core_mask[remove_idx] = False
        self.rebuild_hypergraph()

    def scale_core_gradients(self, scale=0.1):
        if self.prototypes.grad is None:
            return
        core = self.core_mask.to(self.prototypes.grad.device)
        self.prototypes.grad[core] *= scale
