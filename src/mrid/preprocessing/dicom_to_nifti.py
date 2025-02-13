"""PostGlioSeg, Author - SponegBob SquarePants"""
import os, subprocess, shutil
import tempfile
import SimpleITK as sitk

def dicom2nifti(inpath:str, outfolder:str, outfname:str, mkdirs=True, save_BIDS=False) -> str:
    """Convert dicom folder to nii.gz and return path to the output file, uses dcm2niix (https://github.com/rordenlab/dcm2niix) which needs to be installed.

    Args:
        inpath (str):
            Path to the dicom files of a single study and single modality.
            Software like weasis has dicom export functionality that can be used to organize DICOM files into folders by patients studies and modalities.

        outfolder (str): Path to the output folder (e.g. `D:/MRI/patient001/0`).

        outname (str):
            Output filename, excluding `.nii.gz` because it will be added by `dcm2niix`.
            Can use modifiers (e.g. `%d` will be replaced with series description string from DICOM metadata), as explained here https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage#General_Usage

        mkdirs (bool, optional): Whether to create `outfolder` if it doesn't exist, otherwise throws an error. Defaults to True.

        save_BIDS (bool, optional): Whether to save extra BIDS sidecar - extra info in JSON format that can't be saved into nifti. Defaults to False.
    """
    # dicom2niix doesnt support non-ascii paths, so convert to temporary directory
    if not inpath.isascii():
        temp = tempfile.TemporaryDirectory()
        shutil.copytree(inpath, os.path.join(temp.name, 'stuff'))
        inpath = os.path.join(temp.name, 'stuff')
    else:
        temp = None

    # create output dir if not exists
    if outfolder != '':
        if not os.path.exists(outfolder):
            if mkdirs: os.makedirs(outfolder)
            else: raise NotADirectoryError(f"Output path {outfolder} doesn't exist")

    # create a list of files in the folder
    files_before = os.listdir(outfolder)

    # run dcm2niix
    BIDS = 'y' if save_BIDS else 'n'
    subprocess.run(["dcm2niix",
                    "-z", "y", # compression
                    "-m", "n", # disable stacking images from different studies
                    "-b", BIDS, # save additional JSON info that can't be saved into nifti (https://bids.neuroimaging.io/ BIDS sidecar format)
                    "-o", os.path.normpath(outfolder), # output folder
                    "-f", outfname, # output filename
                    os.path.normpath(inpath)], # input folder
                   check=True)

    # create a list of files in the folder after running dcm2niix
    files_after = os.listdir(outfolder)
    # find what new nifti files were created
    new_files = list(set(files_after) - set(files_before))
    new_nii_files = [i for i in new_files if '.nii.gz' in i]
    if len(new_nii_files) > 1: print(f"More than one nifti file was created in {outfolder}, path to the first one will be returned. Something may be wrong.")
    if len(new_nii_files) == 0: print(f"No nifti files were created in {outfolder}")

    if temp is not None: temp.cleanup() # type:ignore

    # return path to the created nifti file
    return os.path.join(outfolder, new_nii_files[0])


def dicom2sitk(inpath:str) -> sitk.Image:
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = dicom2nifti(inpath=inpath, outfolder=tmpdir, outfname='temp', mkdirs=False, save_BIDS=False)
        return sitk.ReadImage(nifti_path)