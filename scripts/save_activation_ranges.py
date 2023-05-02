import argparse
import json
from pathlib import Path
from vdna import VDNAProcessor

def save_activation_ranges(args):
    vdna_proc = VDNAProcessor()

    vdna = vdna_proc.make_vdna(
        source=args.source,
        feat_extractor_name=args.feat_extractor_name,
        distribution_name="activation-ranges",
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
        verbose=args.verbose,
    )

    neurons_list = vdna.neurons_list
    mins_per_neuron = {}
    maxs_per_neuron = {}
    for layer in neurons_list:
        mins_per_neuron[layer] = {}
        maxs_per_neuron[layer] = {}
        for neuron in range(neurons_list[layer]):
            mins_per_neuron[layer][neuron] = vdna.get_neuron_dist(layer, neuron)["min"].item()
            maxs_per_neuron[layer][neuron] = vdna.get_neuron_dist(layer, neuron)["max"].item()

    out = {"mins_per_neuron": mins_per_neuron, "maxs_per_neuron": maxs_per_neuron}
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(out, indent=4)
    with open(save_dir / "activation_ranges.json", "w") as f:
        f.write(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save activation ranges of a model to a JSON file.")
    parser.add_argument("source", type=str, help="source to give to VDNAProcessor.make_vdna")
    parser.add_argument("feat_extractor_name", type=str, help="name of the feature extractor to give to VDNAProcessor.make_vdna")
    parser.add_argument("save_dir", type=str, help="directory to save the activation ranges to")
    parser.add_argument("--seed", type=int, default=0, help="random seed to use (default: 0)")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size to use (default: 64)")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use (default: 'cuda:0')")
    parser.add_argument("--verbose", action="store_true", help="whether to print verbose output (default: False)")

    args = parser.parse_args()
    save_activation_ranges(args)
