import logging
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch

from ..networks import FeatureExtractionModel
from .vdna_base import VDNA


class VDNAFeatures(VDNA):
    def __init__(self):
        super().__init__()
        self.type = "features"
        self.name = "features"
        self.data = {}
        logging.warning(
            "This is pretty inefficient as all raw features are kept. We currently do not have metrics to deal with raw features."
        )

    def _set_extraction_settings(self, feat_extractor: FeatureExtractionModel) -> FeatureExtractionModel:
        return feat_extractor

    def fit_distribution(self, features_dict: Dict[str, torch.Tensor]):
        self.data = features_dict

    def _get_vdna_metadata(self) -> dict:
        return {}

    def _save_dist_data(self, file_path: Union[str, Path]):
        file_path = Path(file_path).with_suffix(".npz")
        data = {}
        for layer in self.data:
            data[layer] = self.data[layer].cpu().numpy().astype(np.float32)
        np.savez_compressed(file_path, **data)

    def _load_dist_data(self, dist_metadata: Dict, file_path: Union[str, Path], device: str):
        file_path = Path(file_path).with_suffix(".npz")
        loaded_data = np.load(file_path)
        self.data = {}
        for layer in self.neurons_list:
            self.data[layer] = torch.from_numpy(loaded_data[layer]).to(device)

    def get_neuron_dist(self, layer_name: str, neuron_idx: int) -> torch.Tensor:
        return self.data[layer_name][:, neuron_idx]

    def get_all_neurons_in_layer_dist(self, layer_name: str) -> torch.Tensor:
        return self.data[layer_name]

    def get_all_neurons_dists(self) -> Dict[str, torch.Tensor]:
        return self.data
