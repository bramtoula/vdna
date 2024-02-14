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
        self.data_settings_used = [DataSettings()]
        self.extraction_settings_used = [ExtractionSettings()]
        self.feature_extractor_name = "NotFilled"
        self.loaded_from_path = "NotLoaded"
        self.neurons_list = {"NotFilled": 0}
        self.device = "cpu"

    def _set_extraction_settings(self, feat_extractor: FeatureExtractionModel):
        raise NotImplementedError

    def fit_distribution(self, features_dict: Dict):
        raise NotImplementedError

    def _get_vdna_metadata(self) -> Dict:
        raise NotImplementedError

    def _save_metadata(self, file_path: Union[str, Path]):
        file_path = Path(file_path)
        file_path = file_path.with_suffix(".json")
        metadata = get_saving_metadata()
        metadata["distribution"] = self._get_vdna_metadata()
        metadata["type"] = self.type
        metadata["name"] = self.name
        metadata["data_settings"] = [data.__dict__ for data in self.data_settings_used]
        for i in range(len(metadata["data_settings"])):
            if isinstance(metadata["data_settings"][i]["source"], list) and isinstance(
                metadata["data_settings"][i]["source"][0], np.ndarray
            ):
                metadata["data_settings"][i]["source"] = "Provided NumPy Arrays - not saved in metadata"
        metadata["extraction_settings"] = [data.__dict__ for data in self.extraction_settings_used]
        metadata["num_images"] = self.num_images
        metadata["feature_extractor_name"] = self.feature_extractor_name
        metadata["neurons_list"] = self.neurons_list

        with open(file_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def _common_before_add(self, other, new_vdna):
        assert self.type == other.type
        assert self.name == other.name
        assert self.feature_extractor_name == other.feature_extractor_name
        assert self.neurons_list == other.neurons_list
        assert self.device == other.device
        new_vdna.device = self.device
        new_vdna.feature_extractor_name = self.feature_extractor_name
        new_vdna.neurons_list = self.neurons_list
        new_vdna.num_images = self.num_images + other.num_images
        new_vdna.data_settings_used = self.data_settings_used + other.data_settings_used
        new_vdna.extraction_settings_used = self.extraction_settings_used + other.extraction_settings_used
        return new_vdna

    def __add__(self, other):
        raise NotImplementedError

    def __iadd__(self, other):
        return self.__add__(other)

    def _save_dist_data(self, file_path: Union[str, Path]):
        raise NotImplementedError

    def _load_metadata(self, file_path: Union[str, Path]):
        file_path = Path(file_path).with_suffix(".json")
        metadata = json.load(open(file_path))
        self.type = metadata["type"]
        self.name = metadata["name"]
        if not isinstance(metadata["data_settings"], list):  # legacy for when data_settings was not a list
            metadata["data_settings"] = [metadata["data_settings"]]
        self.data_settings_used = [DataSettings(**data) for data in metadata["data_settings"]]
        if not isinstance(metadata["extraction_settings"], list):  # legacy for when extraction_settings was not a list
            metadata["extraction_settings"] = [metadata["extraction_settings"]]
        self.extraction_settings_used = [ExtractionSettings(**data) for data in metadata["extraction_settings"]]
        self.num_images = metadata["num_images"]
        self.feature_extractor_name = metadata["feature_extractor_name"]
        self.neurons_list = metadata["neurons_list"]
        return metadata["distribution"]

    def _load_dist_data(self, dist_metadata: Dict, file_path: Union[str, Path], device: str):
        raise NotImplementedError

    def prepare_settings(self, feature_extractor: FeatureExtractionModel, data_settings: DataSettings):
        feat_extractor = self._set_extraction_settings(feature_extractor)
        self.device = str(feature_extractor.extraction_settings.device)
        self.extraction_settings_used = [feat_extractor.extraction_settings]
        self.data_settings_used = [data_settings]
        self.feature_extractor_name = feat_extractor.name
        self.neurons_list = feat_extractor.network_settings.neurons_per_layer
        return feat_extractor

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
