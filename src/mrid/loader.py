from typing import TYPE_CHECKING, TypeAlias
import os
import SimpleITK as sitk
import numpy as np


from ._utils import TORCH_INSTALLED
MRILike: TypeAlias = "np.ndarray | sitk.Image | torch.Tensor | str"

if TYPE_CHECKING:
    import torch

def imread_pil(path:str) -> np.ndarray:
    import PIL.Image
    return np.array(PIL.Image.open(path))


def read_sitk(path: str) -> sitk.Image:
    if os.path.isfile(path): return sitk.ReadImage(path)
    if not os.path.exists(path): raise FileNotFoundError(path)
    # temporary
    first_file = os.listdir(path)[0]
    import pydicom
    try:
        ext = first_file.split('.')[-1].strip().lower()
        if ext in ('jpg', 'jpeg', 'png'): raise ValueError

        _test_if_opening_dicom_raises = pydicom.dcmread(os.path.join(path, first_file))
        from .preprocessing.dicom_to_nifti import dicom2sitk
        return dicom2sitk(path)
    except Exception as e1:
        try:
            files = [os.path.join(path, f) for f in sorted(os.listdir(path), key = lambda s: int(''.join(c for c in s if c.isnumeric())))]
            arr = np.stack([imread_pil(f) for f in files])
            if arr.ndim == 4: arr = arr.mean(-1)
            return sitk.GetImageFromArray(np.stack([imread_pil(f) for f in files]), isVector=False)
        except Exception as e2:
            print('Unable to load image')
            print(f'PIL: {e2!r}')
            raise e1 from e2

def tositk(x: MRILike) -> sitk.Image:
    """Load an image into an itk.Image object.
    `x` can be a numpy array, a sitk.Image, a torch.Tensor or a string (path to an image file)."""
    if isinstance(x ,np.ndarray): return sitk.GetImageFromArray(x)
    if isinstance(x, sitk.Image): return x
    if isinstance(x, str): return read_sitk(x)
    if TORCH_INSTALLED:
        import torch
        if isinstance(x, torch.Tensor): return sitk.GetImageFromArray(x.numpy())
    raise TypeError(f"Unsupported type {type(x)}")

def tonumpy(x: MRILike) -> np.ndarray:
    """Load an image into a numpy.ndarray.
    `x` can be a numpy array, a sitk.Image, a torch.Tensor or a string (path to an image file)."""
    if isinstance(x ,np.ndarray): return x
    if isinstance(x, sitk.Image): return sitk.GetArrayFromImage(x)
    if isinstance(x, str): return sitk.GetArrayFromImage(read_sitk(x))
    if TORCH_INSTALLED:
        import torch
        if isinstance(x, torch.Tensor): return x.numpy()
    raise TypeError(f"Unsupported type {type(x)}")

def totensor(x: MRILike) -> "torch.Tensor":
    """Load an image into a torch.Tensor.
    `x` can be a numpy array, a sitk.Image, a torch.Tensor or a string (path to an image file)."""
    import torch
    if isinstance(x ,np.ndarray): return torch.from_numpy(x)
    if isinstance(x, sitk.Image): return torch.from_numpy(sitk.GetArrayFromImage(x))
    if isinstance(x, str): return  torch.from_numpy(sitk.GetArrayFromImage(read_sitk(x)))
    if isinstance(x, torch.Tensor): return x
    raise TypeError(f"Unsupported type {type(x)}")