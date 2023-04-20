from .feature_extraction_model import FeatureExtractionModel

from .clip import CLIPImEncoder
from .dino_resnet50 import resnet50_feat_extractor as DINOResnet50FeatExtractor
from .dino_vit import DINOViTFeatExtractor
from .inception_pytorch import InceptionV3
from .mugs_vit import MugsViTFeatExtractor
from .random_resnet50 import resnet50_feat_extractor as RandomResnet50FeatExtractor
from .vgg16 import VGG16FeatExtractor


def get_feature_extractor(feature_extractor, extraction_settings):
    if feature_extractor == "inception":
        # Input is normalised already
        model = InceptionV3(
            output_blocks=[0, 1, 2, 3],
            resize_input=False,
            normalize_input=False,
            extraction_settings=extraction_settings,
        )
    elif feature_extractor == "dino_resnet50":
        model = DINOResnet50FeatExtractor(extraction_settings=extraction_settings)
    elif feature_extractor == "dino_vit_base":
        model = DINOViTFeatExtractor("base", extraction_settings=extraction_settings)
    elif feature_extractor == "dino_vit_small":
        model = DINOViTFeatExtractor("small", extraction_settings=extraction_settings)
    elif feature_extractor == "rand_resnet50":
        model = RandomResnet50FeatExtractor(extraction_settings=extraction_settings)
    elif feature_extractor == "vgg16":
        model = VGG16FeatExtractor(extraction_settings=extraction_settings)
    elif feature_extractor == "mugs_vit_large":
        model = MugsViTFeatExtractor("large", extraction_settings=extraction_settings)
    elif feature_extractor == "mugs_vit_base":
        model = MugsViTFeatExtractor("base", extraction_settings=extraction_settings)
    elif feature_extractor == "mugs_vit_small":
        model = MugsViTFeatExtractor("small", extraction_settings=extraction_settings)
    elif "clip_im_" in feature_extractor:
        model_version = feature_extractor[8:]
        model = CLIPImEncoder(model_version, extraction_settings=extraction_settings)
    else:
        raise NotImplementedError(f"Feature extractor {feature_extractor} not implemented")
    model.name = feature_extractor
    device = extraction_settings.device
    model = model.to(device)
    model.eval()
    return model
