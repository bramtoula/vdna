from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch

from .vdna_base import VDNA


class VDNAActivationRanges(VDNA):
    def __init__(self):
        super().__init__()
        self.type = "activation-ranges"
        self.name = "activation-ranges"
        self.data = {}

    def _set_extraction_settings(self, feat_extractor):
        feat_extractor.extraction_settings.keep_only_min_max = True
        return feat_extractor

    def fit_distribution(self, features_dict):
        self.data = {}
        for layer in features_dict:
            self.data[layer] = {}
            # Get minimum over spatial dimensions which are dims 2 and 3. Minimum is stored in the first channel of the last dimension
            self.data[layer]["min"] = torch.min(
                features_dict[layer][..., 0].reshape(1, features_dict[layer].shape[1], -1), dim=2
            ).values.squeeze()
            # Get maximum over spatial dimensions which are dims 2 and 3. Maximum is stored in the second channel of the last dimension
            self.data[layer]["max"] = torch.max(
                features_dict[layer][..., 1].reshape(1, features_dict[layer].shape[1], -1), dim=2
            ).values.squeeze()


    def _get_vdna_metadata(self) -> dict:
        return {}

    def _save_dist_data(self, file_path: Union[str, Path]):
        file_path = Path(file_path).with_suffix(".npz")
        data = {}
        for layer in self.data:
            data[layer] = {}
            data["min-" + layer] = self.data[layer]["min"].cpu().numpy().astype(np.float32)
            data["max-" + layer] = self.data[layer]["max"].cpu().numpy().astype(np.float32)
        np.savez_compressed(file_path, **data)

    def _load_dist_data(self, dist_metadata: Dict, file_path: Union[str, Path], device: str):
        file_path = Path(file_path).with_suffix(".npz")
        loaded_data = np.load(file_path)
        self.data = {}
        for layer in self.neurons_list:
            self.data[layer] = {}
            self.data[layer]["min"] = torch.from_numpy(loaded_data["min-" + layer]).to(device)
            self.data[layer]["max"] = torch.from_numpy(loaded_data["max-" + layer]).to(device)

    def get_neuron_dist(self, layer_name: str, neuron_idx: int):
        return {
            "min": self.data[layer_name]["min"][neuron_idx],
            "max": self.data[layer_name]["max"][neuron_idx],
        }

    def get_all_neurons_in_layer_dist(self, layer_name: str):
        min = self.data[layer_name]["min"]
        max = self.data[layer_name]["max"]
        return {"min": min, "max": max}

    def get_all_neurons_dists(self):
        mins = []
        maxs = []
        for layer_name in self.data:
            mins.append(self.data[layer_name]["min"])
            maxs.append(self.data[layer_name]["max"])
        return {"min": torch.cat(mins), "max": torch.cat(maxs)}

    def __add__(self, other):
        new_vdna = VDNAActivationRanges()
        new_vdna = self._common_before_add(other, new_vdna)

        for layer in self.data:
            new_vdna.data[layer] = {}
            new_vdna.data[layer]["min"] = torch.min(self.data[layer]["min"], other.data[layer]["min"])
            new_vdna.data[layer]["max"] = torch.max(self.data[layer]["max"], other.data[layer]["max"])
        return new_vdna
