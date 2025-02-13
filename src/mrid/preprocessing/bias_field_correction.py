import SimpleITK as sitk

def n4_bias_field_correction(image: sitk.Image, shrink: int = 1,):
    norm_image = sitk.RescaleIntensity(image, 0, 255)
    mask = sitk.OtsuThreshold(norm_image, 0, 1)

    if shrink > 1:
        reduced = sitk.Shrink(image, [shrink] * image.GetDimension())
        mask = sitk.Shrink(mask, [shrink] * mask.GetDimension())

    else: reduced = image

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(reduced, mask)
    log_bias_field = corrector.GetLogBiasFieldAsImage(image)

    return image / sitk.Cast(sitk.Exp(log_bias_field), image.GetPixelID())
