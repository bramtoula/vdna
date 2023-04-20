from typing import List

import numpy as np
import torch
from scipy.linalg import sqrtm


def histogram_per_channel(data: torch.Tensor, hist_nb_bins: int, hist_range: List[float]) -> torch.Tensor:
    # Takes array of size (B,C,H,W) and returns histogram counts of values in specified dims. out shape is (C,bin_number)
    # For a single element in the batch, we can afford to handle all channels simultaneously. Memory explodes if we have more elements.

    # Make sure values are in range
    data = torch.clamp(data, hist_range[0], hist_range[1])

    # For small number of activations, we can afford to do it all at once, but otherwise memory explodes
    if data.shape[0] * data.shape[1] * data.shape[2] * data.shape[3] * hist_nb_bins < 2e6:
        bin_edges = torch.linspace(hist_range[0], hist_range[1], steps=hist_nb_bins + 1).to(data.device)
        bin_edges_left = bin_edges[:-1]
        bin_edges_right = bin_edges[1:]
        data_expanded = data.unsqueeze(-1)
        out = torch.logical_and(data_expanded >= bin_edges_left, data_expanded <= bin_edges_right).sum((0, 2, 3))

    # For more activations, we just do it channel by channel
    else:
        out = torch.zeros((data.shape[1], hist_nb_bins), dtype=torch.long).to(data.device)
        for c in range(data.shape[1]):
            out[c] = torch.histc(
                data[:, c, :, :].flatten(),
                bins=hist_nb_bins,
                min=hist_range[0],
                max=hist_range[1],
            )
    return out


def earth_movers_distance(hist1: torch.Tensor, hist2: torch.Tensor) -> torch.Tensor:
    # Expects histograms of same shape. Will normalise them.
    # Each row is a histogram. The result has the EMD for each row comparison.
    hist1 = hist1 / torch.sum(hist1, dim=1, dtype=torch.double, keepdim=True)
    hist2 = hist2 / torch.sum(hist2, dim=1, dtype=torch.double, keepdim=True)
    diff = hist1 - hist2
    y = torch.cumsum(diff, dim=1)
    return torch.sum(torch.abs(y), dim=1)


# Adapted from https://github.com/bioinf-jku/TTUR/blob/master/fid.py
# Stable version by Dougal J. Sutherland
def frechet_distance_multidim(
    mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    device = mu1.device
    mu1 = mu1.cpu().numpy()
    mu2 = mu2.cpu().numpy()
    sigma1 = sigma1.cpu().numpy()
    sigma2 = sigma2.cpu().numpy()

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; " "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=5e-3):
            m = np.max(np.abs(covmean.imag))
            print("WARNING! Imaginary component {} in FD computation".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fd = torch.tensor(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean).to(device)
    return fd


def frechet_distance_1d(mu1: torch.Tensor, var1: torch.Tensor, mu2: torch.Tensor, var2: torch.Tensor) -> torch.Tensor:
    # Expects mus and sigmas to be numpy arrays where each row will lead to a frechet distance
    return torch.square(mu1 - mu2) + var1 + var2 - 2 * torch.sqrt(var1 * var2)
