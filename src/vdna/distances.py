from typing import Dict, Optional, Union

import torch

from .utils.stats import earth_movers_distance, frechet_distance_1d, frechet_distance_multidim
from .utils.utils import convert_gaussian_to_neuron_gaussian
from .vdnas.vdna_base import VDNA
from .vdnas.vdna_gauss import VDNAGauss
from .vdnas.vdna_hist import VDNAHist
from .vdnas.vdna_layer_gauss import VDNALayerGauss


def common_check_vdna_comps(
    vdna1: VDNA,
    vdna2: VDNA,
    use_neurons_from_layer: Optional[str],
    use_neuron_index: Optional[int],
    return_neuron_wise: bool,
):
    """
    Check that two VDNAs have the same feature extractor, are on the same device,
    and that the specified layer and neuron index are valid.

    Args:
        vdna1 (VDNA): The first VDNA.
        vdna2 (VDNA): The second VDNA.
        use_neurons_from_layer (str or None): The name of the layer to use neurons from,
            or None if not using a specific layer.
        use_neuron_index (int or None): The index of the neuron to use in the specified layer,
            or None if not using a specific neuron.
        return_neuron_wise (bool): Whether to return neuron-wise distance.

    Raises:
        AssertionError: If the feature extractors are not the same, the specified
            neuron index is out of bounds, or the VDNAs are on different devices.
        ValueError: If `return_neuron_wise` is True and `use_neuron_index` is not None.
    """
    assert vdna1.feature_extractor_name == vdna2.feature_extractor_name, "Feature extractors must be the same"
    assert (
        use_neuron_index is not None and use_neurons_from_layer is not None
    ) or use_neuron_index is None, "If specifying neuron index, you should also specify layer name"
    if use_neurons_from_layer:
        assert use_neurons_from_layer in vdna1.neurons_list, "Layer name not found in VDNA neurons"

    if use_neuron_index and use_neurons_from_layer:
        assert use_neuron_index < vdna1.neurons_list[use_neurons_from_layer], (
            "Neuron index not found in VDNA neurons for layer"
            + use_neurons_from_layer
            + " which has "
            + str(len(vdna1.neurons_list[use_neurons_from_layer]))
            + " neurons"
        )

    if return_neuron_wise and use_neuron_index is not None:
        raise ValueError("Cannot return neuron-wise distance when specifying a specific neuron to use")

    assert vdna1.device == vdna2.device, "VDNAs must be on the same device"


def EMD(
    vdna1: VDNAHist,
    vdna2: VDNAHist,
    use_neurons_from_layer: Optional[str] = None,
    use_neuron_index: Optional[int] = None,
    return_neuron_wise: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Calculates the Earth Mover's Distance (EMD) between histograms of two VDNAs.

    Args:
        vdna1 (VDNAHist): The first VDNA with histograms.
        vdna2 (VDNAHist): The second VDNA with histograms.
        use_neurons_from_layer (str or None, optional): The name of the layer to use neurons from,
            or None if not using a specific layer. Defaults to None.
        use_neuron_index (int or None, optional): The index of the neuron to use in the specified layer,
            or None if not using a specific neuron. Defaults to None.
        return_neuron_wise (bool, optional): Whether to return neuron-wise distance. Defaults to False.

    Returns:
        torch.Tensor or Dict[str, torch.Tensor]: The EMD between the two VDNAs.
            If `return_neuron_wise` is True, returns a dictionary mapping neurons
            to EMDs.
            If `return_neuron_wise` is False, returns a single EMD value from the average of selected neurons.

    Raises:
        AssertionError: If the first VDNA is not a histogram, the second VDNA is not a histogram,
            the histograms do not have the same number of bins, or the specified neuron index
            is out of bounds.

    Example:
        >>> from vdna import load_vdna_from_files, EMD
        >>> # Load VDNAs
        >>> vdna1 = load_vdna_from_files("/path/to/vdna1")
        >>> vdna2 = load_vdna_from_files("/path/to/vdna2")
        >>> # Calculate EMD between all neurons in layer "block_0"
        >>> EMD(vdna1, vdna2, use_neurons_from_layer="block_0")
        >>> # Calculate EMD between neuron 0 in layer "block_0"
        >>> EMD(vdna1, vdna2, use_neurons_from_layer="block_0", use_neuron_index=0)
        >>> # Calculate EMD between all neurons in layer "block_0" and return neuron-wise distance
        >>> EMD(vdna1, vdna2, use_neurons_from_layer="block_0", return_neuron_wise=True)
    """
    assert vdna1.type == "histogram", "First VDNA must be a histogram for EMD"
    assert vdna2.type == "histogram", "Second VDNA must be a histogram for EMD"
    assert vdna1.hist_nb_bins == vdna2.hist_nb_bins, "Histograms must have the same number of bins"
    common_check_vdna_comps(vdna1, vdna2, use_neurons_from_layer, use_neuron_index, return_neuron_wise)

    if not return_neuron_wise:
        if use_neurons_from_layer:
            if use_neuron_index is not None:
                vdna1_hists = vdna1.get_neuron_dist(use_neurons_from_layer, use_neuron_index)
                vdna2_hists = vdna2.get_neuron_dist(use_neurons_from_layer, use_neuron_index)

            else:
                vdna1_hists = vdna1.get_all_neurons_in_layer_dist(use_neurons_from_layer)
                vdna2_hists = vdna2.get_all_neurons_in_layer_dist(use_neurons_from_layer)
        else:
            vdna1_hists = vdna1.get_all_neurons_dists()
            vdna2_hists = vdna2.get_all_neurons_dists()

        return torch.mean(earth_movers_distance(vdna1_hists, vdna2_hists))

    else:
        emds_per_neuron = {}
        layers_to_use = [use_neurons_from_layer] if use_neurons_from_layer else vdna1.neurons_list.keys()
        for layer in layers_to_use:
            vdna1_hists = vdna1.get_all_neurons_in_layer_dist(layer)
            vdna2_hists = vdna2.get_all_neurons_in_layer_dist(layer)
            emds_per_neuron[layer] = earth_movers_distance(vdna1_hists, vdna2_hists)
        return emds_per_neuron


def NFD(
    vdna1: Union[VDNAGauss, VDNALayerGauss],
    vdna2: Union[VDNAGauss, VDNALayerGauss],
    use_neurons_from_layer: Optional[str] = None,
    use_neuron_index: Optional[int] = None,
    return_neuron_wise: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Calculates the Neuron Fidelity Distance (NFD) between two VDNA Gaussian distributions.

    Args:
        vdna1 (VDNAGauss or VDNALayerGauss): The first VDNA with Gaussian distributions.
        vdna2 (VDNAGauss or VDNALayerGauss): The second VDNA with Gaussian distributions.
        use_neurons_from_layer (str or None, optional): The name of the layer to use neurons from,
            or None if not using a specific layer. Defaults to None.
        use_neuron_index (int or None, optional): The index of the neuron to use in the specified layer,
            or None if not using a specific neuron. Defaults to None.
        return_neuron_wise (bool, optional): Whether to return neuron-wise distance. Defaults to False.

    Returns:
        torch.Tensor or Dict[str, torch.Tensor]: The NFD between the two VDNAs.
            If `return_neuron_wise` is True, returns a dictionary mapping neuron indices
            to NFDs.
            If `return_neuron_wise` is False, returns a single EMD value from the average of selected neurons.

    Raises:
        AssertionError: If the first VDNA is not a Gaussian distribution, the second VDNA is not a
            Gaussian distribution, the specified neuron index is out of bounds, or the specified
            layer does not exist.

    Example:
        >>> from vdna import load_vdna_from_file, NFD
        >>> # Load VDNA Gaussian distributions
        >>> vdna1 = load_vdna_from_file("/path/to/vdna1")
        >>> vdna2 = load_vdna_from_file("/path/to/vdna2")
        >>> # Calculate NFD between all neurons in layer "block_0"
        >>> NFD(vdna1, vdna2, use_neurons_from_layer="block_0")
        >>> # Calculate NFD between neuron 0 in layer "block_0"
        >>> NFD(vdna1, vdna2, use_neurons_from_layer="block_0", use_neuron_index=0)
        >>> # Calculate NFD between all neurons in layer "block_0" and return neuron-wise distance
        >>> NFD(vdna1, vdna2, use_neurons_from_layer="block_0", return_neuron_wise=True)
    """
    assert vdna1.type in [
        "layer-gaussian",
        "gaussian",
    ], "First VDNA must use a Gaussian distribution for NFD"
    assert vdna2.type in [
        "layer-gaussian",
        "gaussian",
    ], "Second VDNA must use a Gaussian distribution for NFD"
    common_check_vdna_comps(vdna1, vdna2, use_neurons_from_layer, use_neuron_index, return_neuron_wise)

    # # If using gaussian distributions with full covariances, we just use about the variances
    if vdna1.type == "layer-gaussian":
        vdna1 = convert_gaussian_to_neuron_gaussian(vdna1)

    if vdna2.type == "layer-gaussian":
        vdna2 = convert_gaussian_to_neuron_gaussian(vdna2)

    if not return_neuron_wise:
        if use_neurons_from_layer:
            if use_neuron_index is not None:
                vdna1_dists = vdna1.get_neuron_dist(use_neurons_from_layer, use_neuron_index)
                vdna2_dists = vdna2.get_neuron_dist(use_neurons_from_layer, use_neuron_index)

            else:
                vdna1_dists = vdna1.get_all_neurons_in_layer_dist(use_neurons_from_layer)
                vdna2_dists = vdna2.get_all_neurons_in_layer_dist(use_neurons_from_layer)
        else:
            vdna1_dists = vdna1.get_all_neurons_dists()
            vdna2_dists = vdna2.get_all_neurons_dists()

        return torch.mean(
            frechet_distance_1d(vdna1_dists["mu"], vdna1_dists["var"], vdna2_dists["mu"], vdna2_dists["var"])
        )

    else:
        nfd_per_neuron = {}
        layers_to_use = [use_neurons_from_layer] if use_neurons_from_layer else vdna1.neurons_list.keys()
        for layer in layers_to_use:
            nfd_per_neuron[layer] = torch.zeros(vdna1.neurons_list[layer]).to(vdna1.device)
            for neuron_idx in range(vdna1.neurons_list[layer]):
                vdna1_dists = vdna1.get_neuron_dist(layer, neuron_idx)
                vdna2_dists = vdna2.get_neuron_dist(layer, neuron_idx)
                nfd_per_neuron[layer][neuron_idx] = frechet_distance_1d(
                    vdna1_dists["mu"], vdna1_dists["var"], vdna2_dists["mu"], vdna2_dists["var"]
                )
        return nfd_per_neuron


def FD(
    vdna1: VDNALayerGauss,
    vdna2: VDNALayerGauss,
    use_neurons_from_layer: Optional[str] = None,
    use_neuron_index: Optional[int] = None,
    return_neuron_wise: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Calculates the Frechet distance between two VDNA Gaussian distributions.

    Args:
        vdna1 (VDNALayerGauss): The first VDNA with layer-wise Gaussian distribution.
        vdna2 (VDNALayerGauss): The second VDNA with layer-wise Gaussian distribution.
        use_neurons_from_layer (str or None, optional): The name of the layer to use neurons from,
            or None if using all neurons. Defaults to None.
        use_neuron_index (int or None, optional): The index of the neuron to use in the specified layer,
            or None if not using a specific neuron. Defaults to None.
        return_neuron_wise (bool, optional): Whether to return neuron-wise distance. Defaults to False.

    Returns:
        torch.Tensor or Dict[str, torch.Tensor]: The Frechet distance between the two VDNAs.
            If return_neuron_wise is True, returns a dictionary with FDs for each layer.
            If return_neuron_wise is False, returns a scalar tensor averaged from al layers.

    Raises:
        AssertionError: If the VDNA distributions are not layer-gaussian.

    Examples:
        >>> from vdnalab import load_vdna_from_file, FD
        >>> vdna1 = load_vdna_from_file("/path/to/vdna1")
        >>> vdna2 = load_vdna_from_file("/path/to/vdna2")
        >>> # Calculate FD between all neurons
        >>> FD(vdna1, vdna2)
        >>> # Calculate FD between neuron 0 in layer "block_0"
        >>> FD(vdna1, vdna2, use_neurons_from_layer="block_0", use_neuron_index=0)
        >>> # Calculate FD between all neurons in layer "block_0" and return neuron-wise distance
        >>> FD(vdna1, vdna2, use_neurons_from_layer="block_0", return_neuron_wise=True)
    """
    assert vdna1.type == "layer-gaussian", "First VDNA must use a full-layer Gaussian distribution for FD"
    assert vdna2.type == "layer-gaussian", "Second VDNA must use a full-layer Gaussian distribution for FD"
    common_check_vdna_comps(vdna1, vdna2, use_neurons_from_layer, use_neuron_index, return_neuron_wise)

    if not return_neuron_wise:
        if use_neurons_from_layer:
            if use_neuron_index is not None:
                vdna1_dists = vdna1.get_neuron_dist(use_neurons_from_layer, use_neuron_index)
                vdna2_dists = vdna2.get_neuron_dist(use_neurons_from_layer, use_neuron_index)
            else:
                vdna1_dists = vdna1.get_all_neurons_in_layer_dist(use_neurons_from_layer)
                vdna2_dists = vdna2.get_all_neurons_in_layer_dist(use_neurons_from_layer)
            return torch.mean(
                frechet_distance_multidim(
                    vdna1_dists["mu"], vdna1_dists["sigma"], vdna2_dists["mu"], vdna2_dists["sigma"]
                )
            )

        else:
            vdna1_dists = vdna1.get_all_neurons_dists()
            vdna2_dists = vdna2.get_all_neurons_dists()
            all_fds = [
                frechet_distance_multidim(
                    vdna1_dists[layer]["mu"],
                    vdna1_dists[layer]["sigma"],
                    vdna2_dists[layer]["mu"],
                    vdna2_dists[layer]["sigma"],
                )
                for layer in vdna1.neurons_list.keys()
            ]
            return torch.mean(torch.Tensor(all_fds))
    else:
        fd_per_neuron = {}
        layers_to_use = [use_neurons_from_layer] if use_neurons_from_layer else vdna1.neurons_list.keys()
        for layer in layers_to_use:
            fd_per_neuron[layer] = torch.zeros(vdna1.neurons_list[layer]).to(vdna1.device)
            for neuron_idx in range(vdna1.neurons_list[layer]):
                vdna1_dists = vdna1.get_neuron_dist(layer, neuron_idx)
                vdna2_dists = vdna2.get_neuron_dist(layer, neuron_idx)
                fd_per_neuron[layer][neuron_idx] = frechet_distance_multidim(
                    vdna1_dists["mu"], vdna1_dists["sigma"], vdna2_dists["mu"], vdna2_dists["sigma"]
                )
        return fd_per_neuron
