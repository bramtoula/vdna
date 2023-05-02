# Adding your feature extractor

Here, we detail the process to add new feature extractors to the library.

## **Overview**
You can include a feature extractor by implementing a new class that inherits from `FeatureExtractionModel` defined in [src/vdna/networks/feature_extraction_model.py](../src/vdna/networks/feature_extraction_model.py).

The class should implement the `get_features` method that takes in a batch of images and returns a dictionary mapping layer names to the corresponding activations.

The class should also provide model-specific information during the initialisation, such as image input size, list of neurons per layer, activation ranges, and image normalisation.

## **Before you start**
During this process, we will be adding features to the library. You need to use a version with the features you have added to run some scripts or tests.

One option to do that is to install the library in editable mode. It would allow you to make changes to the library and use the updated version without having to reinstall it:    
```
# From the root of the repository you are updating
pip install -e .
```

## **Detailed steps**

## 1. Create a new class that inherits from `FeatureExtractionModel`
Create a new class that inherits from `FeatureExtractionModel` and put it in a new file in `src/vdna/networks`, such as:
```
src/vdna/networks/my_new_feature_extractor.py
```

In the `__init__` method, you need to provide some data specific to the model by creating a `NetworkSettings` object and passing it to the parent class.
You can also load the model weights in this method if not done in the `__init__` of the model itself.  
More specifically:
- You can load the weights of the model, ideally downloading them from a URL. For some of our supported feature extractors, we use Huggingface Hub to store model weights and load them from the library using the `hf_hub_download` function.
- You need to create a `NetworkSettings` and provide it to the parent class. This is defined in [src/vdna/utils/settings.py](../src/vdna/utils/settings.py). The arguments are:
  - `norm_mean` and `norm_std`, the mean and standard deviation of the image normalisations expected by the network.
  - `min_max_act_per_neuron`, which you can leave empty as in the example below. We will fill it later (see step 4).
  - `expected_size`, a tuple of the expected size of the input images.
  - `name`, the name of the feature extractor.
  
Example:
```
# src/vdna/networks/my_new_feature_extractor.py
import torch
import torch.nn as nn

from ..utils.io import load_dict
from ..utils.settings import ExtractionSettings, NetworkSettings
from .feature_extraction_model import FeatureExtractionModel

# Model you want to add
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.layer_2 = nn.Conv2d(64, 128, 3, 1, 1)

    def forward(self, x):
        all_outputs = {}
        x = self.layer_1(x)
        x = self.layer_2(x)
        return x

# Wrapper you need to create
class MyNewFeatExtractor(FeatureExtractionModel):
    def __init__(self, extraction_settings=ExtractionSettings()):
        min_max_act_per_neuron = {"mins_per_neuron": {}, "maxs_per_neuron": {}}
        network_settings = NetworkSettings(
            [0.485, 0.456, 0.406],      # Image normalisation mean expected for network
            [0.229, 0.224, 0.225],      # Image normalisation std expected for network
            {
                "layer_1": 64,          # Number of neurons in layer 1
                "layer_2": 128,         # Number of neurons in layer 2
            },
            (224, 224),                 # Required size of input images
            "my_new_feat_extractor",    # Name of the feature extractor
        )
        super(MyNewFeatExtractor, self).__init__(network_settings, extraction_settings, min_max_act_per_neuron)
        self.model = MyModel()
        self.model.load_state_dict(torch.load("path/to/model/weights.pth"))
```

## 2. Implement the `get_features` method

The input is a batch of images of shape `(batch_size, 3, H, W)`.  
The expected output is a dictionary mapping layer names to the corresponding activations.  
The layer names should be the same as the ones defined in `__init__` earlier.  
The activations should be of shape `(batch_size, N_l, H_l, W_l)` where `N_l` is the number of neurons in the layer and `H_l` and `W_l` are the height and width of activation maps of the layer.

Example:
```
# src/vdna/networks/my_new_feature_extractor.py
class MyModel(nn.Module):
    . # See above
    .
    .

class MyNewFeatExtractor(FeatureExtractionModel):
    def __init__(self, extraction_settings=ExtractionSettings()):
        . # See above
        .
        .

    def get_features(self, batch):
        # batch is of shape (batch_size, 3, H, W)
        # output is a dictionary mapping layer names to activations
        # You could also build the dictionary in the model's forward method and return it from there 
        output = {}
        output["layer_1"] = self.model.layer_1(batch)             # Expected to have shape (batch_size, 64, H_l1, W_l1)
        output["layer_2"] = self.model.layer_2(output["layer_1"]) # Expected to have shape (batch_size, 128, H_l2, W_l2)
        return output
```

## 3. Add the new feature extractor to the `get_feature_extractor` function of [src/vdna/networks/`__`init`__`.py](../src/vdna/networks/__init__.py)

```
# src/vdna/networks/__init__.py
from .feature_extraction_model import FeatureExtractionModel

from .clip import CLIPImEncoder
.
.
.
from .my_new_feature_extractor import MyNewFeatExtractor


def get_feature_extractor(feature_extractor, extraction_settings):
    if feature_extractor == "inception":
        # Input is normalised already
        model = InceptionV3(
            output_blocks=[0, 1, 2, 3],
            resize_input=False,
            normalize_input=False,
            extraction_settings=extraction_settings,
        )
    .
    .
    .
    elif feature_extractor == "my_new_feat_extractor":
        model = MyNewFeatExtractor(extraction_settings=extraction_settings)
    .
    .
    .
    model.name = feature_extractor
    device = extraction_settings.device
    model = model.to(device)
    model.eval()
    return model
```


## 4. Get activation ranges (if you want to use histograms)
We store the activation ranges in a json file which we have precomputed for supported networks. They are currently stored in json files on the [Huggingface Hub](https://huggingface.co/bramtoula/visual-dna-models).


The goal here is to estimate the minimum and maximum activation values that can be observed for each neuron in the network.
For the provided networks in the library, we have used images from many datasets listed in the paper to cover diverse image domains.

You can generate a similar json file with your images using the script in [scripts/save_activation_ranges.py](../scripts/save_activation_ranges.py):
```
python scripts/save_activation_ranges.py path/to/images/index/file.txt my_new_feat_extractor save/dir/for/activation_ranges.json
```

Once generated, you can update the `min_max_act_per_neuron` dictionary in the `__init__` method of your feature extractor to include the activation ranges:

```
# src/vdna/networks/my_new_feature_extractor.py
.
.
class MyNewFeatExtractor(FeatureExtractionModel):
    def __init__(self, extraction_settings=ExtractionSettings()):
        min_max_act_per_neuron = load_dict("path/to/activation_ranges.json")
        .
        .
        super(MyNewFeatExtractor, self).__init__(network_settings, extraction_settings, min_max_act_per_neuron)
        .
        .
.
.
```

## 5. Run tests to make sure everything works

You can now add the name of your feature extractor to the tests in [tests/test_all.py](../tests/test_all.py) and run the tests to check that everything works. You can comment out the other feature extractors to only test yours:
```
# tests/test_all.py
    .
    .
    .

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
        # "mugs_vit_base",
        # "mugs_vit_large",
        # "vgg16",
        # "inception",
        # "dino_resnet50",
        # "dino_vit_base",
        # "rand_resnet50",
        # "clip_im_vit_b16",
        "my_new_feat_extractor",
    ],
)
class TestVDNAs:
    .
    .
    .
```

Then make sure `pytest` is installed and run the tests:

```
# If pytest is not installed yet: pip install pytest
# Run from the vdna root directory
pytest
```

## 6. Use the feature extractor 
Your feature extractor should be good to go!

If you believe your feature extractor could be useful to others, please consider submitting a pull request to this repository.