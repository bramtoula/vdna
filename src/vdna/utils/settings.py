from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Union

import numpy as np


@dataclass
class DataSettings:
    source: Union[str, List[str], List[np.ndarray]] = "NotFilled"
    crop_to_square_pre_resize: str = "none"
    resize_mode: str = "clean"
    custom_fn_resize: Union[None, Callable] = None
    custom_np_image_tranform: Union[None, Callable] = None
    custom_pil_image_tranform: Union[None, Callable] = None
    shuffle_files: bool = True
    num_images: int = -1


@dataclass
class NetworkSettings:
    norm_mean: List = field(default_factory=lambda: [0.0, 0.0, 0.0])
    norm_std: List = field(default_factory=lambda: [1.0, 1.0, 1.0])
    neurons_per_layer: Dict = field(default_factory=dict)
    expected_size: Tuple = (None, None)
    name: str = ""


@dataclass
class ExtractionSettings:
    average_feats_spatially: bool = False
    accumulate_spatial_feats_in_hist: bool = False
    accumulate_sample_feats_in_hist: bool = False
    keep_only_min_max: bool = False
    normalise_feats: bool = False
    range_scale_for_norm_params: float = 1.2
    hist_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    hist_nb_bins: int = 0
    hist_channel_batch_size: int = 10
    num_workers: int = 12
    batch_size: int = 64
    device: str = "cuda:0"
    description: str = ""
    verbose: bool = True
    n_sample_images: int = 10
    sample_images_folder: str = "sample_images"
    seed: int = 0
    hub_repo: str = "bramtoula/visual-dna-models"
