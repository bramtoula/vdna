from pathlib import Path
from typing import Union

import numpy as np
import torch

from vdna import EMD, FD, NFD, VDNA, VDNAProcessor, load_vdna_from_files


def check_same_dist_data(
    d1: Union[dict, list, np.ndarray, float, int, torch.Tensor],
    d2: Union[dict, list, np.ndarray, float, int, torch.Tensor],
    tol=1e-5,
) -> bool:
    if type(d1) != type(d2):
        return False
    # Check if dict
    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            return False
        for k in d1.keys():
            if not check_same_dist_data(d1[k], d2[k]):
                return False
        return True
    # Check if list
    elif isinstance(d1, list) and isinstance(d2, list):
        return check_same_dist_data(np.array(d1), np.array(d2))
    # Check if numpy array or float or int
    elif isinstance(d1, np.ndarray):
        return torch.isclose(torch.from_numpy(d1), torch.from_numpy(d2), atol=tol).all()
    elif isinstance(d1, float) or isinstance(d1, int):
        return torch.isclose(torch.tensor(d1), torch.tensor(d2), atol=tol).all()
    elif isinstance(d1, torch.Tensor):
        return torch.isclose(d1, d2, atol=tol).all()
    else:
        return False


def check_same_data(v1: VDNA, v2: VDNA):
    assert v1.type == v2.type
    assert v1.name == v2.name
    assert check_same_dist_data(v1.data, v2.data)
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
    min_diff = 1e8
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
        min_diff = min(min_diff, result_all.item())

    if v1.type == "histogram":
        result_all = EMD(v1, v2)
        result_neuron_wise = EMD(v1, v2, return_neuron_wise=True)
        result_first_layer = EMD(v1, v2, use_neurons_from_layer=first_layer_name)
        result_first_layer_first_neuron = EMD(v1, v2, use_neurons_from_layer=first_layer_name, use_neuron_index=0)
        check_valid_results_neuron_wise_approaches(
            first_layer_name, result_all, result_neuron_wise, result_first_layer, result_first_layer_first_neuron
        )
        min_diff = min(min_diff, result_all.item())

    if v1.type == "layer-gaussian":
        result_all = FD(v1, v2)
        result_neuron_wise = FD(v1, v2, return_neuron_wise=True)
        result_first_layer = FD(v1, v2, use_neurons_from_layer=first_layer_name)
        result_first_layer_first_neuron = FD(v1, v2, use_neurons_from_layer=first_layer_name, use_neuron_index=0)

        min_diff = min(min_diff, result_all.item())

    if v1.type == "activation-ranges":
        # No comparison function for activation ranges currently
        if check_same_dist_data(v1.data, v2.data):
            min_diff = 0.0
        else:
            min_diff = 1e8
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
