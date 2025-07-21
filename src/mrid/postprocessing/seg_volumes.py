import SimpleITK as sitk
import numpy as np

def calculate_segmentation_volumes(
    segmentation_image: sitk.Image,
    class_labels: dict[str, int] | list[tuple[str, int]] | list[int]
) -> dict[str, float]:
    """
    Calculates the physical volume of each specified segmentation class
    in a SimpleITK image.

    Args:
        segmentation_image: A SimpleITK Image containing integer labels for
                           semantic segmentation. The image MUST have correct
                           spacing information set for the volume calculation
                           to be in physical units (e.g., mm³).
        class_labels: Defines the classes and their corresponding integer labels
                      in the segmentation image. Can be:
                      - A dictionary: {'ClassName1': label1, 'ClassName2': label2, ...}
                      - A list of tuples: [('ClassName1', label1), ('ClassName2', label2), ...]
                      - A list of integers: [label1, label2, ...] (Class names
                        will be automatically generated as 'Label_labelX').

    Returns:
        A dictionary where keys are the class names and values are their
        calculated volumes in physical units (derived from image spacing,
        e.g., mm³ if spacing is in mm). Returns 0.0 for labels specified
        in class_labels but not present in the image.

    Raises:
        TypeError: If segmentation_image is not a sitk.Image.
        TypeError: If class_labels is not a dict, list of tuples, or list of ints.
        ValueError: If image spacing is not set or invalid.
        RuntimeError: If SimpleITK filter execution fails.
    """
    if not isinstance(segmentation_image, sitk.Image):
        raise TypeError("Input 'segmentation_image' must be a SimpleITK.Image.")

    # Check image spacing
    spacing = segmentation_image.GetSpacing()
    if not spacing or len(spacing) != segmentation_image.GetDimension():
        raise ValueError("Segmentation image must have valid spacing information set.")
    if any(s <= 0 for s in spacing):
        raise ValueError(f"Image spacing components must be positive: {spacing}")

    # --- Prepare label dictionary ---
    label_dict: dict = {}
    if isinstance(class_labels, dict):
        label_dict = class_labels
    elif isinstance(class_labels, list):
        if not class_labels:
            print("Warning: 'class_labels' list is empty. No volumes will be calculated.")
            return {}
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], int) for item in class_labels):
            label_dict = dict(class_labels) # type: ignore
        elif all(isinstance(item, int) for item in class_labels):
            label_dict = {f"Label_{label}": label for label in class_labels}
        else:
            raise TypeError("If 'class_labels' is a list, it must contain either (str, int) tuples or only ints.")
    else:
        raise TypeError("Input 'class_labels' must be a dictionary, a list of (str, int) tuples, or a list of ints.")

    if not label_dict:
        print("Warning: No valid class labels provided after processing input. No volumes calculated.")
        return {}

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    label_image_type = segmentation_image.GetPixelID()
    if label_image_type not in [sitk.sitkUInt8, sitk.sitkUInt16, sitk.sitkUInt32, sitk.sitkUInt64,
                                 sitk.sitkInt8, sitk.sitkInt16, sitk.sitkInt32, sitk.sitkInt64]:
        print(f"Warning: Segmentation image pixel type ({segmentation_image.GetPixelIDValue()}) "
               f"is not a typical integer type. Casting to sitk.sitkUInt32 for statistics calculation.")
        segmentation_image = sitk.Cast(segmentation_image, sitk.sitkUInt32)

    shape_stats.Execute(segmentation_image)

    volumes: dict[str, float] = {}

    # print(f"Labels found in image by LabelShapeStatisticsImageFilter: {present_labels}")
    # print(f"Calculating volumes for requested labels: {list(label_dict.values())}")

    for class_name, label_value in label_dict.items():
        if not isinstance(label_value, int) or label_value < 0:
            print(f"Warning: Invalid label value '{label_value}' for class '{class_name}'. Skipping.")
            continue

        if shape_stats.HasLabel(label_value):
            volume = shape_stats.GetPhysicalSize(label_value)
            volumes[class_name] = volume
        else:
            volumes[class_name] = 0.0

    return volumes