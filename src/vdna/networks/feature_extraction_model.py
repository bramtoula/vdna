import logging
import os
import random
import zipfile
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..utils.im import IM_EXTENSIONS, ResizeDataset, denormalise_tensors
from ..utils.settings import ExtractionSettings, NetworkSettings
from ..utils.stats import histogram_per_channel


def get_min_max_features(acc_feats, feats, layer):
    with torch.no_grad():
        # Keep only min and max values in 2 tensor channels of the last dimension. Min and max values are taken over the batch
        if layer in acc_feats:
            # Get min and max over batch
            b_min_max = torch.stack(
                (
                    torch.min(feats[layer], dim=0, keepdim=True).values,
                    torch.max(feats[layer], dim=0, keepdim=True).values,
                ),
                dim=-1,
            )
            # Keep min max between previous data and current batch
            acc_feats[layer] = torch.stack(
                (
                    torch.min(acc_feats[layer][..., 0], b_min_max[..., 0]),
                    torch.max(acc_feats[layer][..., 1], b_min_max[..., 1]),
                ),
                dim=-1,
            )
        else:
            # If first batch, then just keep the histogram
            acc_feats[layer] = torch.stack(
                (
                    torch.min(feats[layer], dim=0, keepdim=True).values,
                    torch.max(feats[layer], dim=0, keepdim=True).values,
                ),
                dim=-1,
            )
    return acc_feats


def get_pre_hist_norm_params_from_min_max(
    min_max_per_neuron: Dict[str, Dict[str, Dict[int, float]]], range_scale: float, device: str = "cpu"
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Get the parameters for normalisation of the histogram of the neuron activations.

    Args:
        min_max_per_neuron (dict): Dictionary with the min and max activation for each neuron.
        range_scale (float): Scaling to apply to the range boundaries before finding normalisation parameters.

    Returns:
        dict: Dictionary with the mean and std tensors to apply to each layer
    """
    norm_means_per_layer = {}
    norm_stds_per_layer = {}
    for layer in min_max_per_neuron["mins_per_neuron"]:
        min_activations = []
        max_activations = []
        for neuron in min_max_per_neuron["mins_per_neuron"][layer]:
            min_activations.append(min_max_per_neuron["mins_per_neuron"][layer][neuron] * range_scale)
            max_activations.append(min_max_per_neuron["maxs_per_neuron"][layer][neuron] * range_scale)
        min_activations = torch.DoubleTensor(min_activations)
        max_activations = torch.DoubleTensor(max_activations)
        mean = (min_activations + max_activations) / 2
        std = (max_activations - min_activations) / 2 + 1e-8
        norm_means_per_layer[layer] = mean.reshape(1, -1, 1, 1).to(device)
        norm_stds_per_layer[layer] = std.reshape(1, -1, 1, 1).to(device)
    return norm_means_per_layer, norm_stds_per_layer


class FeatureExtractionModel(nn.Module):
    def __init__(
        self,
        network_settings: NetworkSettings = NetworkSettings(),
        extraction_settings: ExtractionSettings = ExtractionSettings(),
        activation_ranges_per_neuron={"mins_per_neuron": {}, "maxs_per_neuron": {}},
    ):
        super(FeatureExtractionModel, self).__init__()
        self.network_settings = network_settings
        self.extraction_settings = extraction_settings
        self.norm_means_per_layer, self.norm_stds_per_layer = get_pre_hist_norm_params_from_min_max(
            activation_ranges_per_neuron,
            self.extraction_settings.range_scale_for_norm_params,
            device=self.extraction_settings.device,
        )
        self.name = "not_set"

    def get_features(self, batch):
        raise NotImplementedError

    """
    Compute the features for a batch of images. Should return a dict with features at each layer.
    """

    def get_batch_features(self, batch, device):
        return self.get_features(batch.to(device))

    def get_dataloader(self, data_settings):
        # Check if data_settings.source is a list of np arrays
        if isinstance(data_settings.source, List) and isinstance(data_settings.source[0], np.ndarray):
            images = data_settings.source
            l_files = None
        else:
            images = None
            l_files = self.get_files_list(data_settings)

        dataset = ResizeDataset(
            l_files,
            images,
            crop_to_square_pre_resize=data_settings.crop_to_square_pre_resize,
            size=self.network_settings.expected_size,
            resize_mode=data_settings.resize_mode,
            norm_mean=self.network_settings.norm_mean,
            norm_std=self.network_settings.norm_std,
        )
        if data_settings.custom_np_image_tranform is not None:
            dataset.custom_np_image_tranform = data_settings.custom_np_image_tranform
        if data_settings.custom_pil_image_tranform is not None:
            dataset.custom_pil_image_tranform = data_settings.custom_pil_image_tranform
        if data_settings.custom_fn_resize is not None:
            dataset.fn_resize = data_settings.custom_fn_resize

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.extraction_settings.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.extraction_settings.num_workers,
        )

        return dataloader

    def get_data_features(self, data_settings):
        dataloader = self.get_dataloader(data_settings)
        # wrap the images in a dataloader for parallelizing the resize operation
        if (
            not self.extraction_settings.average_feats_spatially
            and not self.extraction_settings.accumulate_spatial_feats_in_hist
            and not self.extraction_settings.keep_only_min_max
        ):
            logging.warning(
                "Will try to accumulate all features from dataset with no spatial reduction, expect huge RAM usage if using many images!"
            )

        device = torch.device(self.extraction_settings.device)

        # collect all features
        acc_feats = {}
        if self.extraction_settings.verbose:
            pbar = tqdm(dataloader, desc=self.extraction_settings.description)
        else:
            pbar = dataloader

        for batch in pbar:
            with torch.no_grad():
                feats = self.get_batch_features(batch, device)

            if self.extraction_settings.average_feats_spatially:
                for layer in feats:
                    feats[layer] = torch.mean(feats[layer], dim=(2, 3), keepdim=True)

            if self.extraction_settings.normalise_feats:
                for layer in feats:
                    feats[layer] = (feats[layer] - self.norm_means_per_layer[layer]) / self.norm_stds_per_layer[layer]

            if (
                self.extraction_settings.accumulate_spatial_feats_in_hist
                or self.extraction_settings.accumulate_sample_feats_in_hist
            ):
                for layer in feats:
                    feats[layer] = histogram_per_channel(
                        feats[layer],
                        hist_nb_bins=self.extraction_settings.hist_nb_bins,
                        hist_range=self.extraction_settings.hist_range,
                    )

            for layer in feats:
                if (
                    not self.extraction_settings.accumulate_sample_feats_in_hist
                    and not self.extraction_settings.keep_only_min_max
                ):
                    # Keep all features for each batch in a list
                    acc_feats[layer] = acc_feats.get(layer, []) + [feats[layer]]
                elif self.extraction_settings.keep_only_min_max:
                    # Keep only min and max features over all samples
                    acc_feats = get_min_max_features(acc_feats, feats, layer)
                else:
                    # Accumulate histograms
                    if layer in acc_feats:
                        # Sum the histograms over all samples
                        acc_feats[layer] += feats[layer]
                    else:
                        # If first batch, then just keep the histogram
                        acc_feats[layer] = feats[layer]

        if (
            not self.extraction_settings.accumulate_sample_feats_in_hist
            and not self.extraction_settings.keep_only_min_max
        ):
            acc_feats = {layer: torch.cat(acc_feats[layer]) for layer in acc_feats}

        dataset = dataloader.dataset

        n_sample_ims = min(len(dataset), self.extraction_settings.n_sample_images)

        sample_images = [dataset[i] for i in random.sample(range(len(dataset)), n_sample_ims)]
        sample_images = denormalise_tensors(
            sample_images, self.network_settings.norm_mean, self.network_settings.norm_std
        )
        return acc_feats, len(dataset), sample_images

    def get_files_list(self, data_settings):
        # get all relevant files in the dataset
        if isinstance(data_settings.source, List):
            # Check that all files exists and are images using pathlib
            for file in data_settings.source:
                assert Path(file).is_file(), f"File {file} does not exist"
                assert file.split(".")[-1] in IM_EXTENSIONS, f"File {file} is not an image"
            files = data_settings.source
        elif data_settings.source.split(".")[-1] in IM_EXTENSIONS:
            files = [data_settings.source]
        elif data_settings.source.split(".")[-1] == "txt":
            files = []
            # Read the text file
            with open(data_settings.source, "r") as f:
                for line in f:
                    files.append(line.strip())
                    # If path is relative, make it absolute using the directory of the text file
                    if not os.path.isabs(files[-1]):
                        files[-1] = os.path.join(os.path.abspath(os.path.dirname(data_settings.source)), files[-1])
                    # Check if the file is an image
                    assert files[-1].split(".")[-1] in IM_EXTENSIONS, f"File {files[-1]} is not an image"
                    # Check if the file exists
                    assert os.path.exists(files[-1]), f"File {files[-1]} does not exist"
            # Sort files
            files = sorted(files)
        elif ".zip" in data_settings.source:
            files = list(set(zipfile.ZipFile(data_settings.source).namelist()))
            # remove the non-image files inside the zip
            files = [x for x in files if os.path.splitext(x)[1].lower()[1:] in IM_EXTENSIONS]
        else:
            files = sorted(
                [
                    file
                    for ext in IM_EXTENSIONS
                    for file in glob(os.path.join(data_settings.source, f"**/*.{ext}"), recursive=True)
                ]
            )

        if self.extraction_settings.verbose:
            print(f"Found {len(files)} images in the provided source")
        # use a subset number of files if needed
        if data_settings.num_images > 0 and data_settings.num_images < len(files):
            if data_settings.shuffle_files:
                random.seed(self.extraction_settings.seed)
                random.shuffle(files)
            files = files[: data_settings.num_images]
        if self.extraction_settings.verbose:
            print(f"Using {len(files)} images")
        return files

    def forward(self, x):
        self.get_features(x)
