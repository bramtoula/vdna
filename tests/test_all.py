import shutil
from pathlib import Path

import numpy as np
import pytest

from vdna import NFD, VDNAProcessor

from utils import get_test_vdnas, check_save_load, compare_vdnas


@pytest.mark.parametrize(
    "distribution_name",
    [
        "gaussian",
        "layer-gaussian",
        "histogram-1000",
        "histogram-50",
        "activation-ranges",
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
class TestVDNAs:
    def test_get_dists(self, distribution_name, feat_extractor):
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

    def test_comp_vdnas(self, distribution_name, feat_extractor, tol=1e-3):
        v1, v2, v3, v4 = get_test_vdnas(distribution_name, feat_extractor)

        assert compare_vdnas(v1, v2) > tol
        assert compare_vdnas(v1, v3) > tol
        assert compare_vdnas(v1, v4) > tol
        assert compare_vdnas(v2, v3) > tol
        assert compare_vdnas(v2, v4) > tol
        assert compare_vdnas(v3, v4) > tol

        assert compare_vdnas(v1, v1) <= tol
        assert compare_vdnas(v2, v2) <= tol
        assert compare_vdnas(v3, v3) <= tol
        assert compare_vdnas(v4, v4) <= tol

    def test_layer_gauss_conversion(self, distribution_name, feat_extractor):
        if distribution_name != "gaussian":
            return
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

        assert NFD(v1, v2) <= 1e-2

    def test_vdnas_sources(self, distribution_name, feat_extractor):
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
        images = [
            "tests/test_data/multiple_images/Lenna_(test_image).png",
            "tests/test_data/multiple_images/4.1.07.tiff",
        ]
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

    def test_make_vdnas_options(self, distribution_name, feat_extractor):
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
    distribution_name = "activation-ranges"
    feat_extractor = "mugs_vit_base"
    vdna_tests = TestVDNAs()
    vdna_tests.test_get_dists(distribution_name, feat_extractor)
    vdna_tests.test_comp_vdnas(distribution_name, feat_extractor)