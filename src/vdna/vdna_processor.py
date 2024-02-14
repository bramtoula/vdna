import copy
import json
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from .networks import FeatureExtractionModel, get_feature_extractor
from .utils.io import save_images
from .utils.settings import DataSettings, ExtractionSettings
from .vdnas import VDNA, get_vdna


def load_vdna_from_files(file_path: Union[str, Path], device: str = "cpu"):
    """
    Load a VDNA object from a file path.

    Args:
        file_path (str or Path): Path to the files containing the VDNA. This should only be the root name without extensions.
        device (str): Device to load the VDNA onto. Default is "cpu".

    Returns:
        VDNA: A VDNA object.

    """
    metadata_file_path = Path(file_path).with_suffix(".json")
    metadata = json.load(open(metadata_file_path))
    dist_name = metadata["name"]
    vdna = get_vdna(dist_name)
    vdna.load(file_path, device=device)
    return vdna


def load_vdna_from_hub(repo_id: str, file_path: str, device: str = "cpu", repo_type: str = "dataset"):
    """
    Load a VDNA object from the Hugging Face Model Hub.

    Args:
        repo_id (str): ID of the repository on the HF Model Hub.
        file_path (str): Path to the files containing the VDNA. This should only be the root name without extensions.
        device (str): Device to load the VDNA onto. Default is "cpu".
        repo_type (str): Type of repository on the Model Hub. Default is "dataset".

    Returns:
        VDNA: A VDNA object.

    """
    metadata_file_path = hf_hub_download(repo_id=repo_id, filename=file_path + ".json", repo_type=repo_type)
    hf_hub_download(repo_id=repo_id, filename=file_path + ".npz", repo_type=repo_type)
    common_file_path = metadata_file_path[:-5]
    metadata = json.load(open(common_file_path + ".json"))
    dist_name = metadata["name"]
    vdna = get_vdna(dist_name)
    vdna.load(common_file_path, device=device)
    return vdna


class VDNAProcessor:
    def __init__(self):
        self.last_extraction_settings_used = ExtractionSettings()
        self.data_settings = DataSettings()
        self.feat_extractor = FeatureExtractionModel()

    def make_vdna(
        self,
        source: Union[str, np.ndarray, List[str], List[np.ndarray]],
        num_images: int = -1,  # Use all images
        shuffle_files: bool = False,
        feat_extractor_name: str = "dinov2_base_224",
        distribution_name: str = "histogram-1000",
        return_one_vdna_per_im: bool = False,
        seed: int = 0,
        batch_size: int = 64,
        device: str = "cuda:0",
        verbose: bool = True,
        num_workers: int = 12,
        save_sample_images: Optional[Union[str, Path]] = None,
        n_sample_images: int = 0,
        crop_to_square_pre_resize: str = "none",
    ) -> Union[VDNA, List[VDNA]]:
        """
        Generates a VDNA (Visual DNA) for a given set of images or path to a directory containing images.

        Args:
            source (Union[str, np.ndarray, List[str], List[np.ndarray]]): The source of images to process. Can be a list of image numpy arrays, a path to an image or to a directory which will be recursively searched, a list of image paths, or a .txt file with each image path in a line.
            num_images (int): The maximum number of images to process. If -1, all files will be processed. Defaults to -1.
            shuffle_files (bool): Whether or not to shuffle the files before processing. Defaults to False.
            feat_extractor_name (str): The name of the feature extractor to use. Defaults to "mugs_vit_base".
            distribution_name (str): The name of the distribution to use for generating the VDNA. Defaults to "histogram-1000".
            return_one_vdna_per_im (bool): Whether or not to return a list of VDNAs, one per image in the source. Image activations are not accumulated with this setting. Defaults to False.
            seed (int): The random seed to use. Defaults to 0.
            batch_size (int): The batch size to use for feature extractor inference. Defaults to 64.
            device (str): The device to use for processing. Defaults to "cuda:0".
            verbose (bool): Whether or not to print progress messages during processing. Defaults to True.
            num_workers (int): The number of worker processes to use for processing. Defaults to 12.
            save_sample_images (Optional[Union[str, Path]]): The path to save sample images after processing. If None, no images will be saved. Defaults to None.
            n_sample_images (int): The number of sample images to save. Defaults to 5.
            crop_to_square_pre_resize (str): Whether or not to crop the images to a square before resizing. Possible values are "none", "center" and "random". Defaults to "none".

        Returns:
            Union[VDNA, List[VDNA]]: A single or list of VDNA (Visual DNA) object(s) that represents the processed images.
        """
        data_settings = DataSettings(
            source=source,
            num_images=num_images,
            shuffle_files=shuffle_files,
            crop_to_square_pre_resize=crop_to_square_pre_resize,
        )
        extraction_settings = ExtractionSettings(
            device=device,
            batch_size=batch_size,
            seed=seed,
            n_sample_images=n_sample_images,
            verbose=verbose,
            num_workers=num_workers,
            keep_sample_feats_separate=return_one_vdna_per_im,
        )
        if feat_extractor_name != self.feat_extractor.name or extraction_settings != self.last_extraction_settings_used:
            self.feat_extractor = get_feature_extractor(feat_extractor_name, extraction_settings)

            # Compile if we have torch version >= 2.0
            if torch.__version__ >= "2.0" and sys.version_info < (3, 11):
                self.feat_extractor = torch.compile(self.feat_extractor)

        if return_one_vdna_per_im:
            assert isinstance(source, List), "Expects a list as a source to return one VDNA per image."

        self.last_extraction_settings_used = extraction_settings

        vdna = get_vdna(distribution_name)
        feat_extractor = vdna.prepare_settings(feature_extractor=self.feat_extractor, data_settings=data_settings)
        features_dict, self.num_images, sample_images = feat_extractor.get_data_features(data_settings)

        if return_one_vdna_per_im:
            vdnas = []
            for i, feat_dict in enumerate(features_dict):
                new_vdna = copy.deepcopy(vdna)
                new_vdna.data_settings_used[0].source = source[i]
                new_vdna.data_settings_used[0].num_images = 1
                new_vdna.fit_distribution(feat_dict)
                vdnas.append(new_vdna)
        else:
            vdna.fit_distribution(features_dict)

        if save_sample_images is not None:
            save_images(save_sample_images, sample_images)

        if return_one_vdna_per_im:
            return vdnas
        else:
            return vdna
