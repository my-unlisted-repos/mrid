from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any
import numpy as np
import SimpleITK as sitk
from ..loader import tositk

if TYPE_CHECKING:
    import torch

def resample_to(input:sitk.Image, reference: sitk.Image, interpolation=sitk.sitkNearestNeighbor) -> sitk.Image:
    """Resample `input` to `reference`, both can be either a `sitk.Image` or a path to a nifti file that will be loaded.

    Resampling uses spatial information embedded in nifti file / sitk.Image - size, origin, spacing and direction.

    `input` is transformed in such a way that those attributes will match `reference`.
    That doesn't guarantee perfect allginment.
    """
    return sitk.Resample(
            input,
            reference,
            sitk.Transform(),
            interpolation
        )


def default_pmap():
    """Default parameter maps for registration"""
    euler = sitk.GetDefaultParameterMap('translation')
    euler['Transform'] = ['EulerTransform']
    pmap = sitk.VectorOfParameterMap()
    pmap.append(sitk.GetDefaultParameterMap("translation"))
    pmap.append(euler)
    pmap.append(sitk.GetDefaultParameterMap("rigid"))
    pmap.append(sitk.GetDefaultParameterMap("affine"))
    return pmap


def register(input:sitk.Image, reference: sitk.Image, pmap: Any = None, log_to_console=False) -> sitk.Image:
    """Register `input` to `reference`, both can be either a `sitk.Image` or a path to a nifti file that will be loaded. Returns `input` registered to `reference`.

    Registering means input image is transformed using affine transforms to match the reference,
    where affine matrix is found using adaptive gradient descent (by elastix default)
    with a loss function that somehow measures how well `input` matches reference.
    it will have the same size, orientation, etc, and the should be perfectly alligned."""
    if pmap is None: pmap = default_pmap()

    # create elastix filter
    elastix = sitk.ElastixImageFilter()
    if log_to_console: elastix.LogToConsoleOn()
    else: elastix.LogToConsoleOff()
    elastix.SetFixedImage(reference)
    elastix.SetMovingImage(input)

    # set it to elastix filter and execute
    elastix.SetParameterMap(pmap)
    elastix.Execute()
    return elastix.GetResultImage()


def register_D(
    images: Mapping[str, sitk.Image],
    reference: "np.ndarray | sitk.Image | torch.Tensor | str",
    key: str,
    pmap: Any = None,
    log_to_console=False,
) -> dict[str, sitk.Image]:
    """Register `input` to reference, then use that transformation to also register `other`, which is usually segmentation."""
    reference = tositk(reference)

    # create elastix filter
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(reference)
    elastix.SetMovingImage(images[key])

    # set pmap to elastix filter and execute
    if pmap is None: pmap = default_pmap()
    elastix.SetParameterMap(pmap)

    if log_to_console: elastix.LogToConsoleOn()
    else: elastix.LogToConsoleOff()
    #elastix.LogToFileOn() # TEMPORARY

    registered_key = elastix.Execute()

    # apply transformation map to all images
    res = {key: registered_key}

    # process segs last because it sets resample interpolator to nearest
    for k,v in sorted(list(images.items()), key = lambda x: 1 if x[0].startswith('seg') else 0):
        if k.startswith('info'):
            res[k] = v
            continue

        if k != key:
            # create filter that will apply the trained elastix parameters to the segmentation
            transform = sitk.TransformixImageFilter()
            tmap = elastix.GetTransformParameterMap()

            # set nearest neighbour interpolation when registering segmentation
            if k.startswith('seg'):
                tmap[0]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
                tmap[1]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
                tmap[2]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
                tmap[3]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]

            transform.SetTransformParameterMap(tmap)
            transform.SetMovingImage(v)
            transform.LogToConsoleOff()

            res[k] = transform.Execute()

    return res


def resize(img: sitk.Image, new_size, interpolator=sitk.sitkLinear):
    """source: https://gist.github.com/lixinqi98/1bbd3596492f20b776fed2778f7cd48c"""
    new_size = list(reversed(new_size))

    # img = sitk.ReadImage(img)
    dimension = img.GetDimension()

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)

    reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                  zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

    # Create the reference image with a zero origin, identity direction cosine matrix and dimension
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()
    reference_size = new_size
    reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]

    reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as
    # this takes into account size, spacing and direction cosines. For the vast majority of images the direction
    # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the
    # spacing will not yield the correct coordinates resulting in a long debugging session.
    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    # Transform which maps from the reference_image to the current img with the translation mapping the image
    # origins to each other.
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))

    # centered_transform = sitk.Transform(transform)
    # centered_transform.AddTransform(centering_transform)

    centered_transform = sitk.CompositeTransform([transform, centering_transform])

    # Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth
    # segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that
    # no new labels are introduced.

    return sitk.Resample(img, reference_image, centered_transform, interpolator, 0.0)

def downsample(image:sitk.Image, factor:float, dims: Sequence[int] | None, interpolator=sitk.sitkLinear):
    """factor = 2 for 2x downsampling"""
    size = sitk.GetArrayFromImage(image).shape
    size = [round(s/factor) if (dims is None or i in dims) else s for i,s in enumerate(size)]
    return resize(image, size, interpolator=interpolator)