import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_graph import DynamicGraphConstructor


class SparseWeightedGraphConv(nn.Module):
    """Memory-safe graph message passing on the already sparsified top-k graph.

    The earlier dense attention implementation built [B*L, heads, N, N]
    score tensors.  On PeMS07 (N=883, B=16, L=12) this easily exceeds 24GB
    during warmup backward.  This layer keeps the paper's dynamic graph
    encoder idea, but performs message passing only over the top-k neighbours
    contained in the dynamic adjacency.  Memory is O(B*L*N*k*D), not O(B*L*N^2).
    """
    def __init__(self, hidden_dim: int, topk: int = 16, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.topk = int(max(1, topk))
        self.lin_msg = nn.Linear(hidden_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def _message(self, x3: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x3: [B,N,D], adj: [B,N,N]
        B, N, D = x3.shape
        k = min(self.topk, N)
        vals, idx = torch.topk(adj, k=k, dim=-1)
        vals = vals / vals.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        idx_flat = idx.reshape(B, N * k)
        neigh = torch.gather(x3, 1, idx_flat.unsqueeze(-1).expand(-1, -1, D)).view(B, N, k, D)
        msg = (neigh * vals.unsqueeze(-1)).sum(dim=2)
        out = self.lin_out(F.gelu(self.lin_msg(msg)))
        return self.norm(x3 + self.dropout(out))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x can be [B,N,D] or [B,L,N,D]; adj is [B,N,N].
        if x.dim() == 3:
            return self._message(x, adj)
        B, L, N, D = x.shape
        k = min(self.topk, N)
        vals, idx = torch.topk(adj, k=k, dim=-1)
        vals = vals / vals.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        src = x.reshape(B * L, N, D)
        idx2 = idx.unsqueeze(1).expand(B, L, N, k).reshape(B * L, N * k)
        vals2 = vals.unsqueeze(1).expand(B, L, N, k).reshape(B * L, N, k)
        neigh = torch.gather(src, 1, idx2.unsqueeze(-1).expand(-1, -1, D)).view(B * L, N, k, D)
        msg = (neigh * vals2.unsqueeze(-1)).sum(dim=2).view(B, L, N, D)
        out = self.lin_out(F.gelu(self.lin_msg(msg)))
        return self.norm(x + self.dropout(out))


class DilatedCausalTemporalConv(nn.Module):
    def __init__(self, hidden_dim, dilation=1, dropout=0.1):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = 3
        self.conv_filter = nn.Conv1d(hidden_dim, hidden_dim, self.kernel_size, dilation=dilation)
        self.conv_gate = nn.Conv1d(hidden_dim, hidden_dim, self.kernel_size, dilation=dilation)
        self.out = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B,L,N,D]
        B, L, N, D = x.shape
        residual = x
        z = x.permute(0, 2, 3, 1).reshape(B * N, D, L)
        pad = (self.kernel_size - 1) * self.dilation
        zpad = F.pad(z, (pad, 0))
        f = torch.tanh(self.conv_filter(zpad))
        g = torch.sigmoid(self.conv_gate(zpad))
        out = self.out(f * g)
        out = out[:, :, -L:]
        out = out.reshape(B, N, D, L).permute(0, 3, 1, 2)
        out = self.dropout(out)
        return self.norm(residual + out)


class STBlock(nn.Module):
    def __init__(self, hidden_dim, heads=4, dilation=1, dropout=0.1, graph_topk=16):
        super().__init__()
        # heads kept in signature for compatibility; sparse layer is single-stream.
        self.graph_mp = SparseWeightedGraphConv(hidden_dim, topk=graph_topk, dropout=dropout)
        self.temporal = DilatedCausalTemporalConv(hidden_dim, dilation=dilation, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x [B,L,N,D], adj [B,N,N].  No dense [B*L,N,N] expansion.
        xt = self.graph_mp(x, adj)
        xt = self.temporal(xt)
        xt = self.norm(xt + self.dropout(self.ffn(xt)))
        return xt


class SpatioTemporalEncoder(nn.Module):
    """Eq. (3)-(4): node features + dynamic graph ST-GNN -> H_t, h_t."""
    def __init__(self, num_nodes, input_dim=1, hidden_dim=64, node_embed_dim=16,
                 time_feat_dim=4, time_embed_dim=16, num_layers=3, heads=4,
                 dropout=0.1, dynamic_topk=12, dynamic_alpha=0.5, static_alpha=0.3, use_static_graph=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.node_emb = nn.Parameter(torch.randn(num_nodes, node_embed_dim) * 0.02)
        self.time_proj = nn.Linear(time_feat_dim, time_embed_dim)
        self.input_proj = nn.Linear(input_dim + node_embed_dim + time_embed_dim, hidden_dim)
        self.graph_constructor = DynamicGraphConstructor(dynamic_topk, dynamic_alpha, static_alpha, use_static_graph=use_static_graph)
        graph_topk = max(2, int(dynamic_topk) + 1)  # include self loop
        self.blocks = nn.ModuleList([
            STBlock(hidden_dim, heads=heads, dilation=2 ** i, dropout=dropout, graph_topk=graph_topk)
            for i in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def build_node_features(self, x, time_feat):
        # x [B,L,N,C]. time_feat may be [B,L,4] or [B,L+T,4];
        # only the historical part is used by the causal encoder.
        B, L, N, C = x.shape
        if time_feat.shape[1] != L:
            time_feat = time_feat[:, :L, :]
        t_emb = self.time_proj(time_feat).unsqueeze(2).expand(B, L, N, -1)
        n_emb = self.node_emb.unsqueeze(0).unsqueeze(0).expand(B, L, N, -1)
        return self.input_proj(torch.cat([x, t_emb, n_emb], dim=-1))

    def forward(self, x, time_feat, static_adj=None):
        adj = self.graph_constructor(x, static_adj=static_adj)
        h = self.build_node_features(x, time_feat)
        for block in self.blocks:
            h = block(h, adj)
        H_t = self.out_norm(h[:, -1])
        h_t = H_t.mean(dim=1)
        return H_t, h_t, adj
