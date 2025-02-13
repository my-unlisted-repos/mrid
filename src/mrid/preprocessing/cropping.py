from collections.abc import Mapping
import SimpleITK as sitk


def crop_bg(image: sitk.Image) -> sitk.Image:
    rescaled = sitk.RescaleIntensity(image, 0, 255)
    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.Execute(sitk.OtsuThreshold(rescaled, 0, 255))
    bbox = filt.GetBoundingBox(255)
    return sitk.RegionOfInterest( image, bbox[int(len(bbox) / 2) :],  bbox[0 : int(len(bbox) / 2)],)


def crop_bg_D(images: Mapping[str, sitk.Image], key: str) -> dict[str, sitk.Image]:
    """Finds the bounding box of `images[key]` and crops all images in `images` to that bounding box."""
    image = images[key]
    rescaled = sitk.RescaleIntensity(image, 0, 255)
    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.Execute(sitk.OtsuThreshold(rescaled, 0, 255))
    bbox = filt.GetBoundingBox(255)

    info = {k:v for k,v in images.items() if k.startswith('info')}
    images = {k:v for k,v in images.items() if not k.startswith('info')}
    ret = {k: sitk.RegionOfInterest(v, bbox[int(len(bbox) / 2) :],  bbox[0 : int(len(bbox) / 2)]) for k,v in images.items()}
    ret.update(info)
    return ret

