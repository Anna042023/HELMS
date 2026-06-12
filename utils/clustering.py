from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans, SpectralClustering


class PrototypeAutoEncoder(nn.Module):
    def __init__(self, dim, latent_dim=None):
        super().__init__()
        latent_dim = latent_dim or max(8, dim // 2)
        self.encoder = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x):
        z = self.encoder(x)
        rec = self.decoder(z)
        return z, rec


@torch.no_grad()
def _centers_from_labels(h: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    centers = []
    global_mean = h.mean(axis=0)
    for i in range(k):
        mask = labels == i
        if mask.any():
            centers.append(h[mask].mean(axis=0))
        else:
            centers.append(global_mean + np.random.normal(0, 1e-3, size=global_mean.shape))
    return np.stack(centers).astype(np.float32)


def fit_autoencoder(states: torch.Tensor, epochs=20, batch_size=512, lr=1e-3, device="cpu") -> Tuple[torch.Tensor, PrototypeAutoEncoder]:
    states = states.detach().float()
    dim = states.shape[-1]
    ae = PrototypeAutoEncoder(dim).to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=1e-4)
    n = states.shape[0]
    for _ in range(max(1, epochs)):
        perm = torch.randperm(n)
        for st in range(0, n, batch_size):
            idx = perm[st:st + batch_size]
            x = states[idx].to(device)
            _, rec = ae(x)
            loss = torch.mean((rec - x) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        z, _ = ae(states.to(device))
    return z.detach().cpu(), ae.cpu()


def cluster_prototypes(states: torch.Tensor, k: int, method="spectral", ae_epochs=20, device="cpu"):
    """Autoencoder + clustering for HMC memory prototypes.

    The paper uses spectral clustering after a small autoencoder.  For large
    initialization pools, use config.init_sample_size<=3000 to keep spectral
    clustering affordable; otherwise the function falls back to MiniBatchKMeans
    instead of crashing.
    """
    n = states.shape[0]
    k = min(k, max(2, n))
    z, ae = fit_autoencoder(states, epochs=ae_epochs, device=device)
    z_np = z.numpy()
    h_np = states.detach().cpu().numpy()
    if method == "spectral" and n <= 3500:
        n_neighbors = min(10, max(1, n - 1))
        clustering = SpectralClustering(
            n_clusters=k, affinity="nearest_neighbors", n_neighbors=n_neighbors,
            assign_labels="kmeans", random_state=2026
        )
        labels = clustering.fit_predict(z_np)
        centers = _centers_from_labels(h_np, labels, k)
    else:
        if method == "spectral":
            print(f"[WARN] spectral clustering skipped because n={n} is large; use init_sample_size<=3500 for exact paper setting.")
        km = MiniBatchKMeans(n_clusters=k, batch_size=min(2048, max(256, n)), random_state=2026, n_init=10)
        labels = km.fit_predict(z_np)
        centers = _centers_from_labels(h_np, labels, k)
    return torch.tensor(centers, dtype=torch.float32), torch.tensor(labels, dtype=torch.long), ae
