import shutil
from pathlib import Path
from typing import Union

import numpy as np
import pytest
import torch

from vdna import EMD, FD, NFD, VDNA, VDNAProcessor, load_vdna_from_files


def check_same_dist_data(
    d1: Union[dict, list, np.ndarray, float, int, torch.Tensor],
    d2: Union[dict, list, np.ndarray, float, int, torch.Tensor],
):
    assert type(d1) == type(d2)
    # Check if dict
    if isinstance(d1, dict) and isinstance(d2, dict):
        assert d1.keys() == d2.keys()
        for k in d1.keys():
            check_same_dist_data(d1[k], d2[k])
    # Check if list
    elif isinstance(d1, list) and isinstance(d2, list):
        check_same_dist_data(np.array(d1), np.array(d2))
    # Check if numpy array
    elif isinstance(d1, np.ndarray) or isinstance(d1, float) or isinstance(d1, int):
        assert torch.equal(torch.Tensor(d1), torch.Tensor(d2))


def check_same_data(v1: VDNA, v2: VDNA):
    assert v1.type == v2.type
    assert v1.name == v2.name
    check_same_dist_data(v1.data, v2.data)
    assert v1.extraction_settings_used == v2.extraction_settings_used
    assert v1.data_settings_used == v2.data_settings_used
    assert v1.feature_extractor_name == v2.feature_extractor_name


def check_save_load(vdna: VDNA, tol) -> VDNA:
    curr_dir = Path(__file__).parent.resolve()

    tmp_name = "tmp_" + vdna.name + "_" + vdna.feature_extractor_name

    save_path = curr_dir / "test_data" / tmp_name
    vdna.save(save_path)
    vdna_loaded = load_vdna_from_files(save_path, device=vdna.device)

    check_same_data(vdna, vdna_loaded)

    # Delete the files with all extensions
    for ext in [".json", ".npy", ".npz"]:
        if (save_path.with_suffix(ext)).exists():
            (save_path.with_suffix(ext)).unlink()

    assert abs(compare_vdnas(vdna, vdna_loaded)) < tol

    return vdna_loaded


def check_valid_results_neuron_wise_approaches(
    first_layer_name,
    result_all,
    result_neuron_wise,
    result_first_layer,
    result_first_layer_first_neuron,
    acceptable_difference: float = 1e-3,
):
    all_neurons = []
    all_neurons_first_layer = result_neuron_wise[first_layer_name]
    for layer in result_neuron_wise:
        all_neurons.append(result_neuron_wise[layer])
    all_neurons = torch.cat(all_neurons)

    assert (
        torch.abs(result_neuron_wise[first_layer_name][0] - result_first_layer_first_neuron) < acceptable_difference
    ), "First neuron of first layer should be the same as the first neuron of the first layer when using neuron index"
    assert (
        torch.abs(torch.mean(all_neurons) - result_all) < acceptable_difference
    ), "Average of all neurons should be the same as not specifying a layer or neuron index"
    assert (
        torch.abs(torch.mean(all_neurons_first_layer) - result_first_layer) < acceptable_difference
    ), "Average of all neurons in first layer should be the same as specifying the first layer"


def compare_vdnas(v1: VDNA, v2: VDNA) -> float:
    min_diff_all = 1e8
    neurons_list = v1.neurons_list
    first_layer_name = list(neurons_list.keys())[0]

    if v1.type in ["gaussian", "layer-gaussian"]:
        result_all = NFD(v1, v2)
        result_neuron_wise = NFD(v1, v2, return_neuron_wise=True)
        result_first_layer = NFD(v1, v2, use_neurons_from_layer=first_layer_name)
        result_first_layer_first_neuron = NFD(v1, v2, use_neurons_from_layer=first_layer_name, use_neuron_index=0)
        check_valid_results_neuron_wise_approaches(
            first_layer_name, result_all, result_neuron_wise, result_first_layer, result_first_layer_first_neuron
        )
        min_diff = min(min_diff_all, result_all.item())

    if v1.type == "histogram":
        result_all = EMD(v1, v2)
        result_neuron_wise = EMD(v1, v2, return_neuron_wise=True)
        result_first_layer = EMD(v1, v2, use_neurons_from_layer=first_layer_name)
        result_first_layer_first_neuron = EMD(v1, v2, use_neurons_from_layer=first_layer_name, use_neuron_index=0)
        check_valid_results_neuron_wise_approaches(
            first_layer_name, result_all, result_neuron_wise, result_first_layer, result_first_layer_first_neuron
        )
        min_diff = min(min_diff_all, result_all.item())

    if v1.type == "layer-gaussian":
        result_all = FD(v1, v2)
        result_neuron_wise = FD(v1, v2, return_neuron_wise=True)
        result_first_layer = FD(v1, v2, use_neurons_from_layer=first_layer_name)
        result_first_layer_first_neuron = FD(v1, v2, use_neurons_from_layer=first_layer_name, use_neuron_index=0)

        min_diff = min(min_diff_all, result_all.item())

    return min_diff


def get_test_vdnas(distribution_name, feat_extractor):
    test_arrays = [np.zeros((100, 100, 3)), np.ones((250, 70, 3))]
    vdna_proc = VDNAProcessor()

    v1 = vdna_proc.make_vdna(
        distribution_name=distribution_name,
        feat_extractor_name=feat_extractor,
        source="tests/test_data/multiple_images",
        batch_size=2,
    )
    v2 = vdna_proc.make_vdna(
        distribution_name=distribution_name, feat_extractor_name=feat_extractor, source="tests/test_data/index.txt"
    )
    v3 = vdna_proc.make_vdna(
        distribution_name=distribution_name, feat_extractor_name=feat_extractor, source=test_arrays
    )
    v4 = vdna_proc.make_vdna(
        distribution_name=distribution_name, feat_extractor_name=feat_extractor, source="tests/test_data/single_image"
    )

    return v1, v2, v3, v4


@pytest.mark.parametrize(
    "distribution_name",
    [
        "gaussian",
        "layer-gaussian",
        "histogram-1000",
        "histogram-50",
    ],
)
@pytest.mark.parametrize(
    "feat_extractor",
    [
        "mugs_vit_base",
        "mugs_vit_large",
        "vgg16",
        "inception",
        "dino_resnet50",
        "dino_vit_base",
        "rand_resnet50",
        "clip_im_vit_b16",
    ],
)
def test_get_dists(distribution_name, feat_extractor):
    v1, v2, v3, v4 = get_test_vdnas(distribution_name, feat_extractor)
    neurons_list = v1.neurons_list
    for layer in neurons_list:
        v1.get_all_neurons_in_layer_dist(layer)
        v2.get_all_neurons_in_layer_dist(layer)
        v3.get_all_neurons_in_layer_dist(layer)
        v4.get_all_neurons_in_layer_dist(layer)
        for neuron_id in range(neurons_list[layer]):
            v1.get_neuron_dist(layer, neuron_id)
            v2.get_neuron_dist(layer, neuron_id)
            v3.get_neuron_dist(layer, neuron_id)
            v4.get_neuron_dist(layer, neuron_id)

    v1.get_all_neurons_dists()
    v2.get_all_neurons_dists()
    v3.get_all_neurons_dists()
    v4.get_all_neurons_dists()


@pytest.mark.parametrize(
    "distribution_name",
    [
        "gaussian",
        "layer-gaussian",
        "histogram-1000",
        "histogram-50",
    ],
)
@pytest.mark.parametrize(
    "feat_extractor",
    [
        "mugs_vit_base",
        "mugs_vit_large",
        "vgg16",
        "inception",
        "dino_resnet50",
        "dino_vit_base",
        "rand_resnet50",
        "clip_im_vit_b16",
    ],
)
def test_comp_vdnas(distribution_name, feat_extractor, tol=1e-3):
    v1, v2, v3, v4 = get_test_vdnas(distribution_name, feat_extractor)

    assert compare_vdnas(v1, v2) > 0.0
    assert compare_vdnas(v1, v3) > 0.0
    assert compare_vdnas(v1, v4) > 0.0
    assert compare_vdnas(v2, v3) > 0.0
    assert compare_vdnas(v2, v4) > 0.0
    assert compare_vdnas(v3, v4) > 0.0

    assert compare_vdnas(v1, v1) <= tol
    assert compare_vdnas(v2, v2) <= tol
    assert compare_vdnas(v3, v3) <= tol
    assert compare_vdnas(v4, v4) <= tol


@pytest.mark.parametrize(
    "feat_extractor",
    [
        "mugs_vit_base",
        "mugs_vit_large",
        "vgg16",
        "inception",
        "dino_resnet50",
        "dino_vit_base",
        "rand_resnet50",
        "clip_im_vit_b16",
    ],
)
def test_layer_gauss_conversion(feat_extractor):
    vdna_proc = VDNAProcessor()

    v1 = vdna_proc.make_vdna(
        distribution_name="gaussian",
        feat_extractor_name=feat_extractor,
        source="tests/test_data/multiple_images",
    )

    v2 = vdna_proc.make_vdna(
        distribution_name="layer-gaussian",
        feat_extractor_name=feat_extractor,
        source="tests/test_data/multiple_images",
    )

    assert NFD(v1, v2) <= 1e-6


@pytest.mark.parametrize(
    "distribution_name",
    [
        "gaussian",
        "layer-gaussian",
        "histogram-1000",
        "histogram-50",
    ],
)
@pytest.mark.parametrize(
    "feat_extractor",
    [
        "mugs_vit_base",
        "mugs_vit_large",
        "vgg16",
        "inception",
        "dino_resnet50",
        "dino_vit_base",
        "rand_resnet50",
        "clip_im_vit_b16",
    ],
)
def test_vdnas_sources(distribution_name, feat_extractor):
    vdna_proc = VDNAProcessor()

    # From a directory
    vdna = vdna_proc.make_vdna(
        distribution_name=distribution_name,
        feat_extractor_name=feat_extractor,
        source="tests/test_data/multiple_images",
    )
    check_save_load(vdna, 1e-2)

    # From a list of np arrays
    images = [
        np.random.rand(448, 224, 3),
        np.random.rand(224, 448, 3),
        np.random.rand(448, 448, 3),
    ]
    vdna = vdna_proc.make_vdna(
        distribution_name=distribution_name,
        feat_extractor_name=feat_extractor,
        source=images,
    )
    check_save_load(vdna, 1e-2)

    # From a list with a single np array
    images = [np.random.rand(448, 224, 3)]
    vdna = vdna_proc.make_vdna(
        distribution_name=distribution_name,
        feat_extractor_name=feat_extractor,
        source=images,
    )
    check_save_load(vdna, 1e-2)

    # From a list of string paths
    images = ["tests/test_data/multiple_images/Lenna_(test_image).png", "tests/test_data/multiple_images/4.1.07.tiff"]
    vdna = vdna_proc.make_vdna(
        distribution_name=distribution_name,
        feat_extractor_name=feat_extractor,
        source=images,
    )
    check_save_load(vdna, 1e-2)

    # From a path to an image
    vdna = vdna_proc.make_vdna(
        distribution_name=distribution_name,
        feat_extractor_name=feat_extractor,
        source="tests/test_data/multiple_images/Lenna_(test_image).png",
    )
    check_save_load(vdna, 1e-2)


@pytest.mark.parametrize(
    "distribution_name",
    [
        "gaussian",
        "layer-gaussian",
        "histogram-1000",
        "histogram-50",
    ],
)
@pytest.mark.parametrize(
    "feat_extractor",
    [
        "mugs_vit_base",
        "mugs_vit_large",
        "vgg16",
        "inception",
        "dino_resnet50",
        "dino_vit_base",
        "rand_resnet50",
        "clip_im_vit_b16",
    ],
)
def test_make_vdnas_options(distribution_name, feat_extractor):
    vdna_proc = VDNAProcessor()

    vdna_proc.make_vdna(
        distribution_name=distribution_name,
        feat_extractor_name=feat_extractor,
        source="tests/test_data/multiple_images",
        verbose=False,
    )

    vdna_proc.make_vdna(
        distribution_name=distribution_name,
        feat_extractor_name=feat_extractor,
        source="tests/test_data/multiple_images",
        num_images=2,
        shuffle_files=True,
    )

    vdna_proc.make_vdna(
        distribution_name=distribution_name,
        feat_extractor_name=feat_extractor,
        source="tests/test_data/multiple_images",
        seed=42,
    )

    vdna_proc.make_vdna(
        distribution_name=distribution_name,
        feat_extractor_name=feat_extractor,
        source="tests/test_data/multiple_images",
        num_workers=4,
    )

    curr_dir = Path(__file__).parent.resolve()
    tmp_name = "tmp_" + distribution_name + "_" + feat_extractor
    save_path = curr_dir / "test_data" / "sample_ims" / tmp_name

    vdna_proc.make_vdna(
        distribution_name=distribution_name,
        feat_extractor_name=feat_extractor,
        source="tests/test_data/multiple_images",
        save_sample_images=save_path,
    )

    shutil.rmtree(save_path)

    test_arrays = [np.zeros((100, 100, 3)), np.ones((250, 70, 3))]

    vdna_proc.make_vdna(
        distribution_name=distribution_name,
        feat_extractor_name=feat_extractor,
        source=test_arrays,
        crop_to_square_pre_resize="random",
    )

    vdna_proc.make_vdna(
        distribution_name=distribution_name,
        feat_extractor_name=feat_extractor,
        source=test_arrays,
        crop_to_square_pre_resize="center",
    )


if __name__ == "__main__":
    dist = "layer-gaussian"
    feat_extractor = "mugs_vit_base"
    test_comp_vdnas(dist, feat_extractor)
    test_get_dists(dist, feat_extractor)
    test_layer_gauss_conversion(feat_extractor)
    test_make_vdnas_options(dist, feat_extractor)
    test_vdnas_sources(dist, feat_extractor)
