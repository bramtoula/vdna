from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor


def build_resizer(mode, size):
    if mode == "clean":
        return make_resizer("PIL", False, "bicubic", size)
    # if using legacy tensorflow, do not manually resize outside the network
    elif mode == "legacy_tensorflow":
        return lambda x: x
    elif mode == "legacy_pytorch":
        return make_resizer("PyTorch", False, "bilinear", size)
    else:
        raise ValueError(f"Invalid mode {mode} specified")


"""
Construct a function that resizes a numpy image based on the
flags passed in.
"""


def make_resizer(library, quantize_after, filter, output_size):
    if library == "PIL" and quantize_after:
        name_to_filter = {
            "bicubic": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "lanczos": Image.LANCZOS,
            "box": Image.BOX,
        }

        def func(x):
            x = Image.fromarray(x)
            x = x.resize(output_size, resample=name_to_filter[filter])
            x = np.asarray(x).clip(0, 255).astype(np.uint8)
            return x

    elif library == "PIL" and not quantize_after:
        name_to_filter = {
            "bicubic": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "lanczos": Image.LANCZOS,
            "box": Image.BOX,
        }
        s1, s2 = output_size

        def resize_single_channel(x_np):
            img = Image.fromarray(x_np.astype(np.float32), mode="F")
            img = img.resize(output_size, resample=name_to_filter[filter])
            return np.asarray(img).clip(0, 255).reshape(s1, s2, 1)

        def func(x):
            x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
            x = np.concatenate(x, axis=2).astype(np.float32)
            return x

    elif library == "PyTorch":
        import warnings

        # ignore the numpy warnings
        warnings.filterwarnings("ignore")

        def func(x):
            x = torch.Tensor(x.transpose((2, 0, 1)))[None, ...]
            x = F.interpolate(x, size=output_size, mode=filter, align_corners=False)
            x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)
            if quantize_after:
                x = x.astype(np.uint8)
            return x

    elif library == "TensorFlow":
        import warnings

        # ignore the numpy warnings
        warnings.filterwarnings("ignore")
        import tensorflow as tf

        def func(x):
            x = tf.constant(x)[tf.newaxis, ...]
            x = tf.image.resize(x, output_size, method=filter)
            x = x[0, ...].numpy().clip(0, 255)
            if quantize_after:
                x = x.astype(np.uint8)
            return x

    elif library == "OpenCV":
        import cv2

        name_to_filter = {
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
            "nearest": cv2.INTER_NEAREST,
            "area": cv2.INTER_AREA,
        }

        def func(x):
            x = cv2.resize(x, output_size, interpolation=name_to_filter[filter])
            x = x.clip(0, 255)
            if quantize_after:
                x = x.astype(np.uint8)
            return x

    else:
        raise NotImplementedError("library [%s] is not include" % library)
    return func


def _make_np_img_square(img_np, crop_to_square_pre_resize):
    h, w = img_np.shape[:2]
    if crop_to_square_pre_resize == "none" or h == w:
        return img_np
    elif crop_to_square_pre_resize == "center":
        size = min(h, w)
        return img_np[(h - size) // 2 : (h + size) // 2, (w - size) // 2 : (w + size) // 2]
    elif crop_to_square_pre_resize == "random":
        size = min(h, w)
        if h > w:
            offset_h = np.random.randint(0, h - size)
            offset_w = 0
        else:
            offset_h = 0
            offset_w = np.random.randint(0, w - size)
        return img_np[offset_h : offset_h + size, offset_w : offset_w + size]

    else:
        raise ValueError("invalid crop_to_square_pre_resize")


# Adapted from clean-fid https://github.com/GaParmar/clean-fid
class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    file_paths: List of all file paths in the folder
    images: List of all images
    fn_resize: function that takes an np_array as input [0,255]
    crop_to_square_pre_resize: if "center" use center crop to make square first.
                 If "random" use random crop to make square first.
                 If "none", do not make square before resizing.
    size: size to resize to
    norm_mean: mean to normalize to
    norm_std: std to normalize to
    """

    def __init__(
        self,
        file_paths: Union[None, List[str], List[Path]] = None,
        images: Union[None, List[np.ndarray]] = None,
        resize_mode: str = "clean",
        crop_to_square_pre_resize: str = "none",
        size: Tuple[int, int] = (299, 299),
        norm_mean: List[float] = [0.0, 0.0, 0.0],
        norm_std: List[float] = [1.0, 1.0, 1.0],
    ):

        if file_paths is None and images is not None:
            self.data_mode = "images"
            self.file_paths = []
            self.images = images
        elif file_paths is not None and images is None:
            self.data_mode = "file_paths"
            self.file_paths = file_paths
            self.images = []
        else:
            raise ValueError("ResizeDataset needs either file paths or images")

        self.crop_to_square_pre_resize = crop_to_square_pre_resize
        self.tf_to_tensor = ToTensor()
        self.tf_norm = Normalize(mean=norm_mean, std=norm_std)
        self.size = size
        self.fn_resize = build_resizer(resize_mode, size=size)
        self.custom_np_image_tranform = lambda x: x
        self.custom_pil_image_tranform = lambda x: x
        self.threw_warning_about_resizing = False

    def __len__(self):
        if self.data_mode == "file_paths":
            return len(self.file_paths)
        return len(self.images)

    def _get_image(self, idx):
        if self.data_mode == "file_paths":
            path = str(self.file_paths[idx])
            return Image.open(path).convert("RGB")
        return Image.fromarray(np.uint8(self.images[idx]))

    def _check_no_cropping(self, img_np):
        if self.crop_to_square_pre_resize == "none" and not self.threw_warning_about_resizing:
            h, w = img_np.shape[:2]
            if h / w > 1.5 or w / h > 1.5:
                print(
                    "WARNING: Aspect ratio of image is far from square. Image will still be resized to a square. "
                    "Other options include setting crop_to_square_pre_resize to center or random."
                )
                self.threw_warning_about_resizing = True

    def __getitem__(self, i):
        img_pil = self._get_image(i)

        # apply a custom PIL image transform before resizing the image
        img_pil = self.custom_pil_image_tranform(img_pil)

        img_np = np.array(img_pil)

        # apply a custom np image transform before resizing the image
        img_np = self.custom_np_image_tranform(img_np)

        # If aspect ratio is far from square and no crop is specified, warn user
        self._check_no_cropping(img_np)
        img_np = _make_np_img_square(img_np, self.crop_to_square_pre_resize)

        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        # ToTensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.tf_to_tensor(np.array(img_resized))
        elif img_resized.dtype == "float32":
            img_t = self.tf_to_tensor(img_resized) / 255

        img_t = self.tf_norm(img_t)
        return img_t


def denormalise_tensors(tensors, mean, std):
    tf_denormalise = Compose(
        [
            Normalize(mean=[0.0, 0.0, 0.0], std=[1 / s for s in std]),
            Normalize(mean=[-m for m in mean], std=[1.0, 1.0, 1.0]),
        ]
    )
    return [tf_denormalise(t) for t in tensors]


IM_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp", "npy", "JPEG"}
