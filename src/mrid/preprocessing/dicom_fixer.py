import datetime
import os

import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.uid import generate_uid


def fix_dicom_series_uids(input_folder, output_folder):
    """
    Reads DICOM files from input_folder, assigns a new common SeriesInstanceUID,
    generates new unique SOPInstanceUIDs and sequential InstanceNumbers,
    and saves the modified files to output_folder.

    Args:
        input_folder (str): Path to the folder containing the problematic DICOM files.
        output_folder (str): Path to the folder where fixed DICOM files will be saved.
                             It will be created if it doesn't exist.
    """
    print("Starting DICOM UID fixing process...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    dicom_files_info = []

    # --- Step 1: Read all DICOM files and gather basic info ---
    print("\n--- Reading DICOM files ---")
    for filename in os.listdir(input_folder):
        input_filepath = os.path.join(input_folder, filename)
        if os.path.isfile(input_filepath):
            try:
                # Read the file; defer_size helps with large files initially
                ds = pydicom.dcmread(input_filepath, defer_size="512 KB", stop_before_pixels=False)

                # Basic check if it's a valid DICOM file with image data
                if 'PixelData' in ds:
                    # Store dataset and original filename
                    dicom_files_info.append({'dataset': ds, 'original_filename': filename})
                    print(f"Read: {filename}")
                else:
                    print(f"Skipping (no PixelData): {filename}")

            except InvalidDicomError:
                print(f"Skipping (not a valid DICOM file or issue reading): {filename}")
            except Exception as e:
                print(f"Skipping (Error reading {filename}): {e}")

    if not dicom_files_info:
        print("\nNo valid DICOM files with PixelData found in the input folder. Exiting.")
        return

    # --- Step 2: Sort files (important for assigning Instance Numbers correctly) ---
    print("\n--- Sorting files ---")
    try:
        # Attempt to sort by existing InstanceNumber if present and valid
        dicom_files_info.sort(key=lambda info: int(info['dataset'].get('InstanceNumber', 0)))
        print("Sorted based on existing InstanceNumber.")
    except (ValueError, TypeError, AttributeError) as e:
        # Fallback to sorting by original filename if InstanceNumber is missing/invalid
        print(f"Warning: Could not sort reliably by InstanceNumber ({e}). Falling back to filename sorting.")
        dicom_files_info.sort(key=lambda info: info['original_filename'])


    # --- Step 3: Generate new UIDs and modify datasets ---
    print("\n--- Generating new UIDs and modifying datasets ---")

    # Generate ONE new Series Instance UID for ALL files in this batch
    new_series_uid = generate_uid()
    print(f"Generated new Series Instance UID for this batch: {new_series_uid}")

    # Process each file
    for i, file_info in enumerate(dicom_files_info):
        ds = file_info['dataset']
        original_filename = file_info['original_filename']
        instance_number = i + 1 # Generate sequential instance number (1-based)

        # Generate a NEW, UNIQUE SOP Instance UID for this specific file
        new_sop_instance_uid = generate_uid()

        # --- Update Key DICOM Tags ---
        ds.SeriesInstanceUID = new_series_uid
        ds.SOPInstanceUID = new_sop_instance_uid
        ds.InstanceNumber = str(instance_number) # VR 'IS' (Integer String)

        # Update File Meta Information (Group 0002)
        # Check if file_meta exists (it should for standard DICOM files)
        if hasattr(ds, 'file_meta') and ds.file_meta:
            ds.file_meta.MediaStorageSOPInstanceUID = new_sop_instance_uid
            # Optional: You might want to update Implementation Class UID and Version Name
            # ds.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
            # ds.file_meta.ImplementationVersionName = f"PYDICOM {pydicom.__version__}"
        else:
            print(f"Warning: File Meta Information (Group 0002) missing or empty in {original_filename}. Cannot update MediaStorageSOPInstanceUID.")
            # Consider adding default File Meta if crucial, but usually not needed just for fixing UIDs.

        # Optional: Add/update other potentially helpful tags (but avoid geometry if unsure)
        # ds.add_new(0x00080013, 'TM', datetime.datetime.now().strftime('%H%M%S.%f')[:16]) # Instance Creation Time (dummy)
        # ds.add_new(0x00080033, 'TM', datetime.datetime.now().strftime('%H%M%S.%f')[:16]) # Content Time (dummy)

        print(f"  Processing {original_filename}:")
        print(f"    Set InstanceNumber: {instance_number}")
        print(f"    Set SeriesInstanceUID: {new_series_uid}")
        print(f"    Set SOPInstanceUID: {new_sop_instance_uid}")
        if hasattr(ds, 'file_meta') and ds.file_meta:
            print(f"    Set MediaStorageSOPInstanceUID: {new_sop_instance_uid}")


    # --- Step 4: Save modified files ---
    print("\n--- Saving modified DICOM files ---")
    saved_count = 0
    error_count = 0
    for file_info in dicom_files_info:
        ds = file_info['dataset']
        # Use original filename for output, or generate a new one if desired
        output_filepath = os.path.join(output_folder, file_info['original_filename'])
        # Example of generating a new filename:
        # output_filename = f"image_{ds.InstanceNumber.zfill(4)}.dcm"
        # output_filepath = os.path.join(output_folder, output_filename)

        try:
            # Ensure directory exists (though created earlier, good practice)
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            # Save the modified dataset. write_like_original=False ensures standard encoding.
            # Set write_like_original=True if you need to preserve specific transfer syntax etc.
            ds.save_as(output_filepath, write_like_original=True)
            print(f"Saved: {output_filepath}")
            saved_count += 1
        except Exception as e:
            print(f"Error saving file {output_filepath}: {e}")
            error_count += 1

    print("\n--- Process Summary ---")
    print(f"Total files read: {len(dicom_files_info)}")
    print(f"Files successfully saved: {saved_count}")
    print(f"Files failed to save: {error_count}")
    print(f"Fixed files are located in: {output_folder}")
    print("--- Finished ---")


# --- Example Usage ---
if __name__ == "__main__":
    # --- !!! IMPORTANT: SET YOUR FOLDER PATHS HERE !!! ---
    input_directory = r"123" # Use raw string (r"...") or double backslashes (\\) on Windows
    output_directory = r"456"

    # Check if input directory exists
    if not os.path.isdir(input_directory):
        print(f"Error: Input directory not found: {input_directory}")
    else:
        fix_dicom_series_uids(input_directory, output_directory)