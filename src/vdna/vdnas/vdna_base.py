import json
from pathlib import Path
from typing import Dict, Union

import numpy as np

from ..networks import FeatureExtractionModel
from ..utils.io import get_saving_metadata
from ..utils.settings import DataSettings, ExtractionSettings


class VDNA:
    def __init__(self):
        self.type = "NotImplemented"
        self.name = "NotImplemented"
        self.data = {}
        self.num_images = 0
        self.data_settings_used = DataSettings()
        self.extraction_settings_used = ExtractionSettings()
        self.feature_extractor_name = "NotFilled"
        self.loaded_from_path = "NotLoaded"
        self.neurons_list = {"NotFilled": 0}
        self.device = "cpu"

    def _set_extraction_settings(self, feat_extractor: FeatureExtractionModel):
        raise NotImplementedError

    def _fit_distribution(self, features_dict: Dict):
        raise NotImplementedError

    def _get_vdna_metadata(self) -> dict:
        raise NotImplementedError

    def _save_metadata(self, file_path: Union[str, Path]):
        file_path = Path(file_path)
        file_path = file_path.with_suffix(".json")
        metadata = get_saving_metadata()
        metadata["distribution"] = self._get_vdna_metadata()
        metadata["type"] = self.type
        metadata["name"] = self.name
        metadata["data_settings"] = self.data_settings_used.__dict__
        if isinstance(metadata["data_settings"]["source"], list) and isinstance(
            metadata["data_settings"]["source"][0], np.ndarray
        ):
            metadata["data_settings"]["source"] = "Provided NumPy Arrays - not saved in metadata"
        metadata["extraction_settings"] = self.extraction_settings_used.__dict__
        metadata["num_images"] = self.num_images
        metadata["feature_extractor_name"] = self.feature_extractor_name
        metadata["neurons_list"] = self.neurons_list

        with open(file_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def _save_dist_data(self, file_path: Union[str, Path]):
        raise NotImplementedError

    def _load_metadata(self, file_path: Union[str, Path]):
        file_path = Path(file_path).with_suffix(".json")
        metadata = json.load(open(file_path))
        self.type = metadata["type"]
        self.name = metadata["name"]
        self.data_settings_used = DataSettings(**metadata["data_settings"])
        self.extraction_settings_used = ExtractionSettings(**metadata["extraction_settings"])
        self.num_images = metadata["num_images"]
        self.feature_extractor_name = metadata["feature_extractor_name"]
        self.neurons_list = metadata["neurons_list"]
        return metadata["distribution"]

    def _load_dist_data(self, dist_metadata: Dict, file_path: Union[str, Path], device: str):
        raise NotImplementedError

    def fill_vdna(self, feature_extractor: FeatureExtractionModel, data_settings: DataSettings):
        feat_extractor = self._set_extraction_settings(feature_extractor)
        self.device = str(feature_extractor.extraction_settings.device)
        self.extraction_settings_used = feat_extractor.extraction_settings
        self.data_settings_used = data_settings
        self.feature_extractor_name = feat_extractor.name
        self.neurons_list = feat_extractor.network_settings.neurons_per_layer
        features_dict, self.num_images, sample_images = feat_extractor.get_data_features(data_settings)

        self._fit_distribution(features_dict)
        return sample_images

    def load(self, file_path: Union[str, Path], device: str = "cpu"):
        dist_metadata = self._load_metadata(file_path)
        self._load_dist_data(dist_metadata, file_path, device)
        self.loaded_from_path = str(file_path)
        self.device = device

    def save(self, file_path: Union[str, Path]):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        self._save_metadata(file_path)
        self._save_dist_data(file_path)

    def get_neuron_dist(self, layer_name: str, neuron_idx: int):
        raise NotImplementedError

    def get_all_neurons_in_layer_dist(self, layer_name: str):
        raise NotImplementedError

    def get_all_neurons_dists(self):
        raise NotImplementedError
