from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch

from .vdna_base import VDNA


def _get_gaussian_params(features: torch.Tensor) -> Dict[str, torch.Tensor]:
    features = features.view(features.shape[0], -1).to(torch.float64)
    mu = torch.mean(features, dim=0)
    if len(features) > 1:
        sigma = torch.cov(features.T)
    else:
        sigma = torch.zeros((features.shape[1], features.shape[1])).to(features.device)
    return {"mu": mu, "sigma": sigma}


class VDNALayerGauss(VDNA):
    def __init__(self):
        super().__init__()
        self.type = "layer-gaussian"
        self.name = "layer-gaussian"
        self.data = {}

    def _set_extraction_settings(self, feat_extractor):
        feat_extractor.extraction_settings.average_feats_spatially = True
        return feat_extractor

    def _get_vdna_metadata(self) -> dict:
        return {}

    def _save_dist_data(self, file_path: Union[str, Path]):
        file_path = Path(file_path).with_suffix(".npz")
        data = {}
        for layer in self.data:
            data[layer] = {}
            data["mu-" + layer] = self.data[layer]["mu"].cpu().numpy().astype(np.float64)
            data["sigma-" + layer] = self.data[layer]["sigma"].cpu().numpy().astype(np.float64)
        np.savez_compressed(file_path, **data)

    def _load_dist_data(self, dist_metadata: Dict, file_path: Union[str, Path], device: str):
        file_path = Path(file_path).with_suffix(".npz")
        loaded_data = np.load(file_path)
        self.data = {}
        for layer in self.neurons_list:
            self.data[layer] = {}
            self.data[layer]["mu"] = torch.from_numpy(loaded_data["mu-" + layer]).to(device)
            self.data[layer]["sigma"] = torch.from_numpy(loaded_data["sigma-" + layer]).to(device)

    def _fit_distribution(self, features_dict):
        self.data = {}
        for layer in features_dict:
            self.data[layer] = _get_gaussian_params(features_dict[layer])

    def get_neuron_dist(self, layer_name: str, neuron_idx: int):
        return {
            "mu": self.data[layer_name]["mu"][neuron_idx],
            "sigma": self.data[layer_name]["sigma"][neuron_idx][neuron_idx],
        }

    def get_all_neurons_in_layer_dist(self, layer_name: str):
        return {"mu": self.data[layer_name]["mu"], "sigma": self.data[layer_name]["sigma"]}

    def get_all_neurons_dists(self):
        return self.data
