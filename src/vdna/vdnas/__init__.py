from .vdna_activation_ranges import VDNAActivationRanges
from .vdna_base import VDNA
from .vdna_feats import VDNAFeats
from .vdna_feats_spatial_avg import VDNAFeatsSpatialAvg
from .vdna_gauss import VDNAGauss
from .vdna_hist import VDNAHist
from .vdna_layer_gauss import VDNALayerGauss


def get_vdna(dist_name):
    if dist_name == "activation-ranges":
        return VDNAActivationRanges()
    if dist_name == "layer-gaussian":
        return VDNALayerGauss()
    elif dist_name == "feats-spatial-avg":
        return VDNAFeatsSpatialAvg()
    elif dist_name == "gaussian":
        return VDNAGauss()
    elif "histogram" in dist_name:
        n_bins = int(dist_name.split("-")[1])
        assert n_bins
        return VDNAHist(hist_nb_bins=n_bins)
    elif dist_name == "feats":
        return VDNAFeats()
    else:
        raise NotImplementedError("VDNA {} not implemented!".format(dist_name))
