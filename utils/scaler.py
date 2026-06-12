import numpy as np
import torch


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

    def inverse_transform_torch(self, data: torch.Tensor):
        mean = torch.as_tensor(self.mean, dtype=data.dtype, device=data.device)
        std = torch.as_tensor(self.std, dtype=data.dtype, device=data.device)

        # If the model predicts only the first channel while the input contains
        # multiple channels, inverse-transform with the matching target channel
        # statistics only.  Without this slice, PyTorch broadcasting expands a
        # [B,T,N,1] prediction to [B,T,N,C], which corrupts metrics and
        # calibration shapes.
        if data.dim() >= 1 and mean.shape[-1] != data.shape[-1]:
            mean = mean[..., :data.shape[-1]]
            std = std[..., :data.shape[-1]]

        while mean.dim() < data.dim():
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return data * std + mean
