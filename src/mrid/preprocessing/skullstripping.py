import os

import tempfile
from collections.abc import Mapping
from typing import Any, Literal

import SimpleITK as sitk


def predict_brain_mask(input:sitk.Image, mode:Literal['fast', 'accurate'] = 'accurate', do_tta=True, device = 'cuda_if_available') -> sitk.Image:
    """Runs skullstripping using HD BET (https://github.com/MIC-DKFZ/HD-BET).
    Requires it to be installed.
    Returns brain mask as sitk.Image, where background is 0, and brain is 1.

    :param input: _description_
    :param mode: _description_, defaults to 'accurate'
    :param do_tta: _description_, defaults to True
    :return: _description_
    """
    from HD_BET_v2023.run import run_hd_bet # TODO fix this
    if device == 'cuda_if_available':
        import torch
        if torch.cuda.is_available(): device = 'cuda'
        else: device = 'cpu'

    with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as temp2:

        sitk.WriteImage(input, os.path.join(temp, 't1.nii.gz'))

        # run skullstripping
        run_hd_bet(os.path.join(temp, 't1.nii.gz'), os.path.join(temp2, 't1.nii.gz'), mode=mode, do_tta=do_tta, device = device) # type:ignore

        # return sitk image
        #print(os.listdir(temp2))
        return sitk.ReadImage(os.path.join(temp2, 't1_mask.nii.gz'))

def apply_brain_mask(input:sitk.Image, mask:sitk.Image, dilate:int | None=0) -> sitk.Image:
    """Applies brain mask to input image.

    Args:
        input (str | sitk.Image): Path to a nifti file or a sitk.Image of the image to generate brain mask from, all inputs must be in MNI152 space.
        mask (str | sitk.Image): Path to a nifti file or a sitk.Image of the brain mask to apply to the input image.
    """
    if dilate is not None and dilate != 0: mask = sitk.BinaryDilate(mask, (dilate, dilate, dilate))
    mask = sitk.Cast(mask, input.GetPixelID())
    return sitk.Multiply(input, mask)


def skullstrip(input:sitk.Image, dilate=0) -> sitk.Image:
    """Skullstrips an image using HD BET (https://github.com/MIC-DKFZ/HD-BET).
    Requires it to be installed. Returns skullstripped image as sitk.Image."""
    mask = predict_brain_mask(input)
    return apply_brain_mask(input, mask, dilate)


def skullstrip_D(
    images: Mapping[str, sitk.Image],
    key: str,
    mode: Literal["fast", "accurate"] = "accurate",
    do_tta=True,
    device="cuda_if_available",
    dilate=0,
    add_mask=True,
) -> dict[str, sitk.Image]:
    template = images[key]
    mask = predict_brain_mask(template, mode = mode, do_tta = do_tta, device=device)

    info = {k:v for k,v in images.items() if k.startswith('info')}
    images = {k:v for k,v in images.items() if not k.startswith('info')}

    ret = {key: (apply_brain_mask(image, mask, dilate) if not key.startswith('seg') else image) for key, image in images.items()}
    if add_mask: ret['seg_brain_mask'] = mask
    ret.update(info)
    return ret