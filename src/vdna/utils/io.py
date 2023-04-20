import datetime
import json
import pathlib
import pickle

from torchvision.utils import save_image

from ..version import __version__


def save_images(dir, images):
    save_dir = pathlib.Path(dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        image_path = save_dir / ("{}.png".format(i))
        save_image(image, image_path)


def get_saving_metadata() -> dict:
    data = {"time_of_compute": datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")}
    data["vdna_version"] = __version__
    return data


def load_dict(path):
    path = str(path)
    if path.split(".")[-1] == "json":
        with open(path) as json_file:
            data = json.load(json_file)
    elif path.split(".")[-1] == "pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
    return data
