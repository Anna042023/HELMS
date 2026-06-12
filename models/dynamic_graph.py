import torch
import torch.nn.functional as F


class DynamicGraphConstructor:
    """Dynamic/static functional graph from a traffic window.

    The original V25 code always executed the dynamic branch (B x N x N cdist)
    even when config.yaml set model.use_static_graph=true.  That made every
    forward pass expensive, especially on PEMS07/PEMS-BAY.  This version honors
    use_static_graph: when a static adjacency is supplied, it builds one sparse
    top-k normalized graph once and only expands it to the current batch.

    Set use_static_graph=false in config.yaml to recover the original dynamic
    DTW/Granger-like graph construction.
    """
    def __init__(self, topk=12, dynamic_alpha=0.5, static_alpha=0.3, use_static_graph=False):
        self.topk = int(topk)
        self.dynamic_alpha = float(dynamic_alpha)
        self.static_alpha = float(static_alpha)
        self.use_static_graph = bool(use_static_graph)
        self._static_cache = {}

    def _static_topk_adj(self, static_adj: torch.Tensor, batch_size: int, dtype, device) -> torch.Tensor:
        s = static_adj.to(device=device, dtype=dtype)
        N = s.shape[0]
        key = (N, self.topk, str(device), str(dtype), int(s.data_ptr()))
        cached = self._static_cache.get(key)
        if cached is None or cached.device != device or cached.dtype != dtype:
            score = s.clone()
            # If the file stores distances or binary connectivity, top-k still
            # preserves the strongest local neighbors.  Add self loops explicitly.
            eye = torch.eye(N, device=device, dtype=torch.bool)
            score = score.masked_fill(eye, float("-inf"))
            k = min(max(1, self.topk), max(1, N - 1))
            top_idx = torch.topk(score, k=k, dim=-1).indices
            adj = torch.zeros(N, N, device=device, dtype=dtype)
            adj.scatter_(dim=-1, index=top_idx, value=1.0)
            adj = adj.masked_fill(eye, 1.0)
            adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
            self._static_cache[key] = adj
            cached = adj
        return cached.unsqueeze(0).expand(batch_size, -1, -1)

    @torch.no_grad()
    def __call__(self, x, static_adj=None):
        # x: [B,L,N,C]
        B, L, N, C = x.shape
        if self.use_static_graph and static_adj is not None:
            return self._static_topk_adj(static_adj, B, x.dtype, x.device)

        series = x[..., 0].transpose(1, 2).contiguous()  # [B,N,L]
        series = series - series.mean(dim=-1, keepdim=True)
        normed = F.normalize(series, dim=-1, eps=1e-6)
        corr = torch.bmm(normed, normed.transpose(1, 2))  # [B,N,N]

        # DTW-like: close shapes get high score. Use cdist on normalized series.
        dist = torch.cdist(normed, normed, p=2)
        dtw_like = -dist

        if L > 1:
            cause_src = F.normalize(series[:, :, :-1], dim=-1, eps=1e-6)
            cause_dst = F.normalize(series[:, :, 1:], dim=-1, eps=1e-6)
            granger_like = torch.bmm(cause_src, cause_dst.transpose(1, 2))
        else:
            granger_like = corr

        score = self.dynamic_alpha * dtw_like + (1.0 - self.dynamic_alpha) * granger_like
        score = score + 0.2 * corr
        if static_adj is not None:
            s = static_adj.to(score.device, score.dtype)
            score = score + self.static_alpha * s.unsqueeze(0)
        eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
        score = score.masked_fill(eye, float("-inf"))
        k = min(self.topk, max(1, N - 1))
        top_idx = torch.topk(score, k=k, dim=-1).indices
        adj = torch.zeros(B, N, N, device=x.device, dtype=x.dtype)
        adj.scatter_(dim=-1, index=top_idx, value=1.0)
        adj = adj.masked_fill(eye, 1.0)
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return adj
