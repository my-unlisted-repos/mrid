import os
import shutil
import tempfile
from collections import UserDict
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import SimpleITK as sitk

from .loader import tositk
from .preprocessing import (crop_bg_D, downsample, n4_bias_field_correction,
                            register, register_D, resample_to, resize,
                            skullstrip_D)

if TYPE_CHECKING:
    import torch

def identity(x):return x

class Study(UserDict[str, sitk.Image]):
    """Dictionary of modalities and optionally the segmentation."""
    def __init__(
        self,
        /,
        info: Any = None,
        **data: "np.ndarray | sitk.Image | torch.Tensor | str",
    ):
        """Create a new study.

        :param info: Any additional information that will be retained when you modify this study, for example can be used to store the target. Defaults to None

        **data: Paths to image file or image data, for each modality. If a key starts with `seg`, it is considered segmentataion.

        Example:
        ```py
        # create a study with T1c and T2 modalities, segmentation, and specify benign target.
        # all keys that start from `seg` are considered segmentation.
        study = Study(
            t1c = 'path/t1c.nii.gz',
            t2 = 'path/t1w.nii.gz',
            seg = 'path/seg.nii.gz',
            info = 'benign'
        ).cast_float64()

        # register T1c to SRI24 using SimpleITK-Elastix, and use the same transformation to register T2 and segmentation
        # (this assumes T1c and T2 are in the same space!)
        sri24_registered = study.register('path/sri24.nii.gz', key = 't1c')

        # predict brain mask on T1c using HD-BET model and apply it to all modalities,
        # and z-normalize each modality to 0 mean 1 variance
        skullstripped = sri24_registered.skullstrip('t1c').normalize()

        # create inputs to your model
        inputs = skullstripped.as_tensor_stack() # (2, D, H, W) tensor, where 2 is [T1c, T2]
        targets = skullstripped.seg_tensor() # (D, H, W) tensor
        label = skullstripped.info # 'benign'
        ```
        """
        super().__init__({k: (tositk(v) if not k.startswith('info') else v) for k,v in data.items()})
        self.info = info

    def copy(self):
        d = self.data.copy()
        return Study(self.info, **d)

    def add_(self, key: str, value: "np.ndarray | sitk.Image | torch.Tensor | str | Any", copy_info: str | None = None):
        """Add a new image to this study or replace existing image if key already exists, optionally copy info from one of the existing images

        Args:
            key (str): name of the key for the new image.
            value (np.ndarray | sitk.Image | torch.Tensor | str): the image itself
            copy_info (str | None, optional): optionally copies sitk info from image under this key. Defaults to None.
        """

        if key.startswith('info'):
            self[key] = value
            return self

        sitk_value = tositk(value)
        if copy_info is not None:

            sitk_value.CopyInformation(self[copy_info])

        self[key] = sitk_value
        return self


    def apply(self, fn:Callable[[sitk.Image], sitk.Image] | None, seg_fn: Callable[[sitk.Image], sitk.Image] | None,) -> "Study":
        """Returns a new study with a function applied to all images. If `apply_to_seg` is True, also applies it to segmentation. Function must take and return a SimpleITK image."""
        if fn is None: fn = identity
        if seg_fn is None: seg_fn = identity
        return Study(self.info, **{k: seg_fn(v) if k.startswith('seg') else (v if k.startswith('info') else fn(v)) for k, v in self.items()})

    def cast(self, dtype) -> "Study":
        """Return a new study with all images except segmentation casted to the specified SimpleITK dtype. Segmentation is not changed."""
        return self.apply(partial(sitk.Cast, pixelID = dtype), seg_fn=None)

    def cast_float64(self) -> "Study":
        """Return a new study with all images except segmentation casted to float64. Segmentation is not changed."""
        return self.cast(sitk.sitkFloat64)

    def cast_float32(self) -> "Study":
        """Return a new study with all images except segmentation casted to float32. Segmentation is not changed."""
        return self.cast(sitk.sitkFloat32)

    def normalize(self) -> "Study":
        """Return a new study where all images except segmentation are separately z-normalized to 0 mean and 1 variance."""
        return self.apply(sitk.Normalize, seg_fn=None)

    def rescale_intensity(self, min: float, max: float) -> "Study":
        """Return a new study where all images except segmentation are separately rescaled to the specified intensity range.
        Segmentation is not changed."""
        return self.apply(partial(sitk.RescaleIntensity, outputMinimum = min, outputMaximum = max), seg_fn=None) # type:ignore

    def crop_bg(self, key) -> "Study":
        """Return a new study with cropped black background. Finds the foreground bounding box of image under `key`,
        and uses that bounding box crop crop all other images including segmentation."""
        d = crop_bg_D(self, key)
        return Study(self.info, **d)

    def skullstrip(
        self,
        key,
        mode: Literal["fast", "accurate"] = "accurate",
        do_tta=True,
        device: 'torch.types.Device | Literal["cuda_if_available"]' = "cuda_if_available",
        dilate: int = 0,
    ) -> "Study":
        """Return a new study with all images skullstripped using HD-BET model. This predicts the brain mask of image under `key`
        (please use T1 registered to MNI152 for best results),
        and uses it to remove skull from all images.
        Only use this if all your images are in the same space,
        if they are not, coregister them first. Doesn't affect segmentation."""
        d = skullstrip_D(self, key, mode=mode, do_tta = do_tta, device = device, dilate=dilate)
        return Study(self.info, **d)

    def resample_to(self, to: "np.ndarray | sitk.Image | torch.Tensor | str", interpolation=sitk.sitkLinear) -> "Study":
        """Returns a new study, resamples all images including segmentation to `to`.
        Segmentation always uses nearest interpolation"""
        to = tositk(to)

        return self.apply(
            partial(resample_to, reference = to, interpolation=interpolation),
            partial(resample_to, reference = to, interpolation=sitk.sitkNearestNeighbor),
        )

    def resize(self, size, interpolator=sitk.sitkLinear):
        return self.apply(
            partial(resize, new_size = size, interpolator=interpolator,),
            partial(resize, new_size = size, interpolator=sitk.sitkNearestNeighbor,),
        )

    def downsample(self, factor, dims = None, interpolator=sitk.sitkLinear):
        """factor = 2 for 2x downsampling. Dims None for all dims."""
        return self.apply(
            partial(downsample, factor = factor, dims = dims, interpolator=interpolator,),
            partial(downsample, factor = factor, dims = dims, interpolator=sitk.sitkNearestNeighbor,),
        )

    def register(self, to: "np.ndarray | sitk.Image | torch.Tensor | str", key: str, log_to_console=False) -> "Study":
        """Returns a new study, registers `key` image to `to`, and use transformation parameters to register all other images including segmentation.
        This assumes that all images are in the same space, if they are not, use `register_each` method."""
        to = tositk(to)

        d = register_D(self, reference = to, key = key,log_to_console=log_to_console)
        return Study(self.info, **d)

    def register_each(self, key: str, to: "np.ndarray | sitk.Image | torch.Tensor | str | None",  seg_space_key = None, log_to_console=False) -> "Study":
        """Returns a new study. Registers all other images to `key`. If `to` is specified, register `key` to `to` beforehand.
        Segmentation is registered using affine transforms of `key`, but you can override that by specifying `seg_space_key`."""
        d = self.copy()
        if to is not None:
            to = tositk(to)
            d[key] = register(d[key], reference = to, log_to_console=log_to_console)

        return Study(self.info, **d)


    def n4_bias_field_correction(self, key: str, shrink: int) -> "Study":
        """Returns a new study with corrected bias field of the image under `key`. Doesn't affect other images.

        Shrink is by how many times to shrink the size of input image for calculating the bias field.
        The bias field is then applied to original size (unshrinked) image.
        Setting shrink to 1 disables it, but n4 algorithm may take several minutes.
        Setting it to ~4 is good enough in most cases and will be significantly faster (usually few seconds).
        """
        new = self.copy()
        new[key] = n4_bias_field_correction(new[key], shrink = shrink)
        return new

    def sorted_keys(self) -> list[str]:
        """Returns a list of keys sorted alphabetically."""
        return sorted(self.keys())

    def _stack_items(self, with_images, with_seg, order = None):
        if not (with_images or with_seg): raise ValueError("At least one of `images` or `seg` must be True")
        items = []
        if order is not None:
            items = [(k, self[k]) for k in order]
        else:
            # make sure items are always sorted in the same order
            if with_images: items = sorted([(k,v) for k,v in self.items() if not k.startswith(('seg', 'info'))], key = lambda x: x[0])
            if with_seg: items.extend(sorted([(k,v) for k,v in self.items() if k.startswith('seg')], key = lambda x: x[0]))
        return items

    def as_numpy_stack(self, with_images = True, with_seg = False, dtype=None, order = None) -> np.ndarray:
        """Returns all volumes as a numpy array stacked along a new first axis.
        If `with_images` is True, images are included.
        If `with_seg` is True, segmentations are included.
        Images are sorted by key name, if both `with_images` and `with_seg` are True, segmentations are always last.

        :param with_seg: Whether to include images. Defaults to True.
        :param with_seg: Whether to include segmentations. Will raise KeyError if `seg` doesn't exist. Defaults to False.
        :raises KeyError: Raised when `with_seg` is True but no segmentation exists in this study.
        :return: A numpy.ndarray of shape [image, D, H, W]
        """
        items = self._stack_items(with_images, with_seg, order = order)

        stacked = np.array([sitk.GetArrayFromImage(v) for _,v in items])
        if dtype is not None: stacked = stacked.astype(dtype, copy=False)
        return stacked

    def as_tensor_stack(self, with_images = True, with_seg = False, dtype=None, order = None) -> "torch.Tensor":
        """Returns all images as a torch tensor stacked along a new first axis.
        If `with_seg` is True, segmentation is also included.
        Images are sorted by key name, segmentation is always last.

        :param with_seg: Whether to include segmentation as the last channel. Will raise KeyError if `seg` doesn't exist. Defaults to False
        :raises KeyError: Raised when `with_seg` is True but no segmentation exists in this study.
        :return: A torch.Tensor of shape [image, D, H, W]
        """
        import torch
        items = self._stack_items(with_images, with_seg, order = order)

        stacked = torch.stack([torch.from_numpy(sitk.GetArrayFromImage(v)) for _,v in items])
        if dtype is not None: stacked = stacked.to(dtype, copy=False)
        return stacked

    def as_numpy_dict(self):
        """Returns a dictionary with numpy arrays of all images"""
        return {k: (sitk.GetArrayFromImage(v) if isinstance(v, sitk.Image) else v) for k,v in self.items()}

    def as_tensor_dict(self):
        """Returns a dictionary with torch tensors of all images"""
        import torch
        return {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k,v in self.as_numpy_dict()}

    def _with_no_seg(self):
        return Study(self.info, **{k:v for k,v in self.items() if not k.startswith('seg')})

    def _with_no_info(self):
        return Study(self.info, **{k:v for k,v in self.items() if not k.startswith('info')})

    def key_to_numpy(self, key:str) -> np.ndarray:
        """Returns image under `key` as a numpy array"""
        return sitk.GetArrayFromImage(self[key])

    def key_to_tensor(self, key:str) -> "torch.Tensor":
        """Returns image under `key` as a torch tensor"""
        import torch
        return torch.from_numpy(self.key_to_numpy(key))

    def write_to_dir(self, path: str, prefix:str = '', suffix: str = '', ext:str = 'nii.gz', mkdir = False, use_compression=True):
        """Writes all images to a directory, with filenames being `{path}/{prefix}{key}{suffix}.{ext}`"""
        if ext.startswith('.'): ext = ext[1:]

        if not os.path.exists(path):
            if mkdir: os.mkdir(path)
            else: raise FileNotFoundError(f"Directory {path} doesn't exist and {mkdir = }")

        for k,v in self.items():
            if k.startswith('info'):
                try:
                    import joblib
                    joblib.dump(v, os.path.join(path, f"{prefix}{k}{suffix}.joblib"))
                except Exception as e:
                    print(f"Couldn't save {k}:\n{e!r}")

            else:

                # this handles non ascii chars
                with tempfile.TemporaryDirectory() as temp_path:
                    sitk.WriteImage(v, os.path.join(temp_path, f"{prefix}{k}{suffix}.{ext}"), useCompression=use_compression)
                    shutil.move(os.path.join(temp_path, f"{prefix}{k}{suffix}.{ext}"), path)


    @classmethod
    def from_dir(cls, path, prefix: str = '', suffix: str = '', ext: str = 'nii.gz'):
        files = os.listdir(path)

        study = {}
        for f in files:
            full = os.path.join(path, f)
            name:str = f
            if prefix == '' or name.startswith(prefix):
                name = name[len(prefix):]

                if name.lower().endswith(f'{suffix.lower()}.{ext.lower()}'):
                    name = name[:-len(f'{suffix.lower()}.{ext.lower()}')]

                    study[name] = full

                elif name.lower().endswith(f'{suffix.lower()}.joblib'):
                    name = name[:-len(f'{suffix.lower()}.joblib')]


                    try:
                        import joblib
                        study[name] = joblib.load(full)
                    except Exception as e:
                        print(f"Couldn't load {full}:\n{e!r}")

        return cls(**study)