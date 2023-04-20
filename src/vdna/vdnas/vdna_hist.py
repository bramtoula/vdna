from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch

from ..networks import FeatureExtractionModel
from .vdna_base import VDNA


class VDNAHist(VDNA):
    def __init__(self, hist_nb_bins: int = 2000):
        super().__init__()
        self.type = "histogram"
        self.name = "histogram-" + str(hist_nb_bins)
        self.hist_nb_bins = hist_nb_bins
        self.data = {}

    def _set_extraction_settings(self, feat_extractor: FeatureExtractionModel) -> FeatureExtractionModel:
        feat_extractor.extraction_settings.accumulate_spatial_feats_in_hist = True
        feat_extractor.extraction_settings.accumulate_sample_feats_in_hist = True
        feat_extractor.extraction_settings.hist_nb_bins = self.hist_nb_bins
        feat_extractor.extraction_settings.normalise_feats = True
        return feat_extractor

    def _fit_distribution(self, features_dict: Dict[str, torch.Tensor]):
        # Already put in histograms during processing thanks to the extraction_settings
        self.data = features_dict

    def _get_vdna_metadata(self) -> dict:
        return {"hist_nb_bins": self.hist_nb_bins}

    def _save_dist_data(self, file_path: Union[str, Path]):
        file_path = Path(file_path).with_suffix(".npz")
        data = {}
        for layer in self.data:
            data[layer] = self.data[layer].cpu().numpy().astype(np.int32)
        np.savez_compressed(file_path, **data)

    def _load_dist_data(self, dist_metadata: Dict, file_path: Union[str, Path], device: str):
        file_path = Path(file_path).with_suffix(".npz")
        self.hist_nb_bins = dist_metadata["hist_nb_bins"]
        loaded_data = np.load(file_path)
        self.data = {}
        for layer in loaded_data:
            self.data[layer] = torch.from_numpy(loaded_data[layer]).to(device)

    def get_neuron_dist(self, layer_name: str, neuron_idx: int) -> torch.Tensor:
        return self.data[layer_name][neuron_idx].reshape(1, -1)

    def get_all_neurons_in_layer_dist(self, layer_name: str) -> torch.Tensor:
        return self.data[layer_name]

    def get_all_neurons_dists(self) -> torch.Tensor:
        histograms = []
        for layer in self.data:
            histograms.append(self.get_all_neurons_in_layer_dist(layer))
        return torch.cat(histograms)
