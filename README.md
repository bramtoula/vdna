# Visual DNA 🧬
## Introduction
This repository provides a library to compute and compare Visual Distribution of Neuron Activations (VDNAs).
Visuals DNAs allow comparing datasets or images using distributions of neuron activations throughout layers of a feature extractor.
They can be used as an alternative to FID and provide more holistic and granular comparisons.

<p align="center">
  <img src=docs/vis/demo.gif />
</p>

VDNAs are generated by passing images to represent through a frozen pre-trained feature extractor network and monitoring activation values throughout the network.
We can represent the images efficiently by fitting distributions such as histograms or Gaussians to the activations at each neuron.
Because VDNAs keep neuron activations independent, we can compare VDNAs while focusing on neurons that are more relevant to the tasks at hand.

<p align="center">
  <img src=docs/vis/dna-overview.svg  width="700"/>
</p>

For example, we can compare VDNAs to find similar datasets based on attributes of interest selected through specific combinations of neurons.

<p align="center">
  <img src=docs/vis/usecase_overview_a.svg  width="400" />  

We can also represent individual images and check their similarity to datasets or other images.

<p align="center">
  <img src=docs/vis/usecase_overview_b.svg  width="400"/>
</p>

## Resources
This work was introduced in our paper:  
[Visual DNA: Representing and Comparing Images using Distributions of Neuron Activations](https://arxiv.org/abs/2304.10036)  
[Benjamin Ramtoula](https://bramtoula.github.io), [Matthew Gadd](https://mttgdd.github.io/), [Paul Newman](https://www.ori.ox.ac.uk/people/paul-newman/), [Daniele de Martini](https://danieledema.github.io/)  
CVPR 2023  

Please consider citing our paper if you make use of this project in your research.
```
@article{ramtoula2023vdna,
  author    = {Ramtoula, Benjamin and Gadd, Matthew and Newman, Paul and De Martini, Daniele},
  title     = {Visual DNA: Representing and Comparing Images using Distributions of Neuron Activations},
  journal   = {CVPR},
  year      = {2023},
}
```

You can also visit the [project website](https://bramtoula.github.io/vdna/).


# Installation
Visual DNA can be installed using pip:
```
pip install vdna
```

# Quick start
Example of comparing two datasets using VDNAs using all default settings (histograms with 1000 bins, Mugs ViT-B/16 feature extractor, all neurons) from two dataset folders:

```
from vdna import VDNAProcessor, EMD

vdna_proc = VDNAProcessor()

vdna1 = vdna_proc.make_vdna(source="/path/to/dataset1")
vdna2 = vdna_proc.make_vdna(source="/path/to/dataset2")

emd = EMD(vdna1, vdna2)
```

# Detailed usage

This library provides tools to create VDNAs, save and load them, and compare them.

The general expected approach is to use the `VDNAProcessor` to create `VDNA` objects. These can be inspected and the distance between two `VDNAs` can be computed using provided functions.

## Making `VDNAs` with the `VDNAProcessor`


See the [documentation in the code](./src/vdna/vdna_processor.py) for options such as the number of workers, batch size, or device.

Some of the settings allow selecting which feature extractor and distribution to use.
See [below](#supported-vdnas) for a list of supported options.

Example usage:
```
from vdna import VDNAProcessor

vdna_proc = VDNAProcessor()
vdna = vdna_proc.make_vdna(source="/path/to/dataset1", distribution_name="histogram-500", num_workers=16, batch_size=128)
```

We also support other input formats:
```
# From a list of NumPy arrays
vdna_from_numpy = vdna_proc.make_vdna(source=list_of_np_arrays)

# From a .txt file containing the paths to all images, one per line
vdna_from_txt = vdna_proc.make_vdna(source="path/to/index.txt")

# From a list of strings containing the paths to all images
vdna_from_txt = vdna_proc.make_vdna(source=["path/to/im1.png","path/to/im2.jpeg"])
```

## Saving and loading VDNAs
You can save and load VDNAs using `save` and `load_vdna_from_files`. They both expect a path without any extension. 

`save` will save two files in the given path and name, one `.json` with metadata and one `.npz` with distribution content. 

`load_vdna_from_files` will load the two files and return a `VDNA` object.

Example:
```
# This will save files /path/to/save/vdna.json and /path/to/save/vdna.npz
vdna.save("/path/to/save/vdna")

# This will load the two files and return a VDNA object
from vdna import load_vdna_from_files
vdna = load_vdna_from_files("/path/to/save/vdna")
```

We also provide `load_vdna_from_hub` to load VDNAs directly from a HuggingFace Hub repository.


## Inspecting VDNAs
Once you have generated the VDNAs, you can access their distributions.
```
from vdna import load_vdna_from_files

# ----- Loading VDNAs -----
vdna1 = load_vdna_from_files("/path/to/save/vdna1")
vdna2 = load_vdna_from_files("/path/to/save/vdna2")

# ----- Checking feature extractor and distribution -----
print(f"vdna1 uses {vdna1.feature_extractor_name} as feature extractor and {vdna1.name} as distribution.")

# ----- Checking neurons used -----
print("List of layers and neurons in the VDNA:")
for layer_name in vdna1.neurons_list:
	print(f"Layer {layer_name} has {vdna1.neurons_list[layer_name]} neurons")

# ----- Checking distribution values -----
all_neurons_hists = vdna1.get_all_neurons_dists()
print(f"We have {all_neurons_hists.shape[0]} neurons in the VDNA, with {all_neurons_hists.shape[1]} bins each.")
print(f"The highest value in a bin is {all_neurons_hists.max()}")

block_0_neurons_hists = vdna1.get_all_neurons_in_layer_dist(layer_name="block_0")
print(f"We have {block_0_neurons_hists.shape[0]} neurons in the VDNA using block_0 neurons, with {block_0_neurons_hists.shape[1]} bins each.")
print(f"The highest value in a bin is {block_0_neurons_hists.max()}")

specific_neuron_hist = vdna1.get_neuron_dist(layer_name="block_0", neuron_idx=42)
print(f"We have {specific_neuron_hist.shape[1]} bins in the histogram for neuron 42 in block_0.")
print(f"The highest value in a bin is {specific_neuron_hist.max()}")
```

## Comparing VDNAs
We also provide distance functions that take in VDNAs to compare, for example with histogram-based VDNAs:  
```
# ----- Comparing VDNAs -----
# Earth Mover's Distance used to compare histogram-based VDNAs
from vdna import load_vdna_from_files, EMD

vdna1 = load_vdna_from_files("/path/to/save/vdna1)
vdna2 = load_vdna_from_files("/path/to/save/vdna2)

print("EMD averaged over all neuron comparisons:")
print(EMD(vdna1, vdna2))

print("EMD averaged over all neurons of block_0:")
print(EMD(vdna1, vdna2, use_neurons_from_layer="block_0"))

print("EMD comparing neuron 42 of layer block_0")
print(EMD(vdna1, vdna2, use_neurons_from_layer="block_0", use_neuron_index=42))

print("Neuron-wise EMD comparisons as a dict:")
emd_neuron_wise = EMD(vdna1, vdna2, return_neuron_wise=True)
for layer in emd_neuron_wise:
	print(f"EMD using neuron 42 of layer {layer} is {emd_neuron_wise[layer][42]}")
```

# Supported VDNAs
Visual DNAs can be constructed with different feature extractors and distributions.
Here we detail supported options.
## Feature extractors
- `mugs_vit_base` (default)
- `mugs_vit_large`
- `vgg16`
- `inception`
- `dino_resnet50`
- `dino_vit_base`
- `rand_resnet50`
- `clip_im_vit_b16`

## Distributions
- `histogram-{number of bins}` (default is `histogram-1000`): 
  - This uses histograms for each neuron, as in the paper.
  - In the paper, results are based on 1000 bins, although you might be fine using fewer.
  - Can be compared with the Earth Mover's Distance: `from vdna import EMD`.
- `gaussian`: 
  - This uses a Gaussian for each neuron, as in the paper.
  - Can be compared with the neuron-wise Fréchet Distance: `from vdna import NFD`.
- `layer-gaussian`:
  - This fits a multivariate Gaussian for each selected layer of the feature extractor.
  - Can be compared with the neuron-wise Fréchet Distance: `from vdna import NFD`.
  - Can be compared with the layer-wise Fréchet Distance (to reproduce FID for example): `from vdna import FD`.
- `activation-ranges`:
  - This keeps track of minimum and maximum activation values for each neuron.
  - Can be useful to get neuron activation ranges used for normalisation for histograms (see [documentation about adding your feature extractor](docs/add_your_feature_extractor.md))


## FID computation
The commonly used Fréchet Inception Distance (FID) can be computed by creating VDNAs with the `inception` feature extractor and the `layer-gaussian` distribution.
It can then be computed using the `FD` function on layer `block_3`.

For example:  
```
from vdna import VDNAProcessor, FD

vdna_proc = VDNAProcessor()

vdna1 = vdna_proc.make_vdna(source=dataset1_path, distribution_name="layer-gaussian", feat_extractor_name="inception")
vdna2 = vdna_proc.make_vdna(source=dataset2_path, distribution_name="layer-gaussian", feat_extractor_name="inception")

print("FID:")
print(FD(vdna1, vdna2, use_neurons_from_layer="block_3"))
```

# Pre-computed VDNAs of common datasets
Coming soon...  
We are planning to pre-compute VDNAs for common datasets and store them online.

# Acknowledgements
This repository was inspired and adapted using parts of the great [`clean-fid`](https://github.com/GaParmar/clean-fid) library.
Our model implementations and weights are adapted from the following repositories:
- [https://github.com/sail-sg/mugs](https://github.com/sail-sg/mugs)
- [https://github.com/facebookresearch/swav](https://github.com/facebookresearch/swav)
- [https://github.com/facebookresearch/dino](https://github.com/facebookresearch/dino)
- [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- [https://github.com/isl-org/PhotorealismEnhancement](https://github.com/isl-org/PhotorealismEnhancement)
- [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR)
- [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
- [https://github.com/mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid)
  
We are very thankful to the contributors of all these repositories for sharing their work.

# License
The code in this repository is released under the MIT License.
