from copy import deepcopy

import torch

from ..vdnas.vdna_gauss import VDNAGauss
from ..vdnas.vdna_layer_gauss import VDNALayerGauss

def convert_gaussian_to_neuron_gaussian(vdna: VDNALayerGauss) -> VDNAGauss:
    assert vdna.type == "layer-gaussian", "VDNA type must be layer-gaussian to be converted to gaussian"
    new_dist = VDNAGauss()
    new_dist.device = vdna.device
    new_dist.neurons_list = vdna.neurons_list.copy()
    new_dist.extraction_settings_used = deepcopy(vdna.extraction_settings_used)
    new_dist.data_settings_used = deepcopy(vdna.data_settings_used)
    new_dist.num_images = vdna.num_images
    new_dist.loaded_from_path = vdna.loaded_from_path
    new_dist.feature_extractor_name = vdna.feature_extractor_name
    for layer in vdna.data:
        mu = vdna.data[layer]["mu"]
        variances = torch.diag(vdna.data[layer]["sigma"])
        new_dist.data[layer] = {"mu": mu, "var": variances}
    return new_dist
