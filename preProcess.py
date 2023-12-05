
import matplotlib
from utils import GetSlicesArray, Unpack2DNpz, UnpackNpz, UnpackSlices, Unpack2DCT, ReadNifti, GetFiles, ReadNiftiFolder, SaveData
import nibabel as nib
import numpy as np
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti
import SimpleITK as sitk
import matplotlib.pyplot as plt
from utils import Save2DData
import os
from pathlib import Path
# file to convert 2D arrays into pseudo-3D nifti images for use with nnunet 
# must be 2D slices and not 3D arrays
def convert_2d_image_to_nifti(img, output_filename_truncated: str, spacing=(999, 1, 1),
                              transform=None, is_seg: bool = False) -> None:
    """
    Reads an image (must be a format that it recognized by skimage.io.imread) and converts it into a series of niftis.
    The image can have an arbitrary number of input channels which will be exported separately (_0000.nii.gz,
    _0001.nii.gz, etc for images and only .nii.gz for seg).
    Spacing can be ignored most of the time.
    !!!2D images are often natural images which do not have a voxel spacing that could be used for resampling. These images
    must be resampled by you prior to converting them to nifti!!!
    Datasets converted with this utility can only be used with the 2d U-Net configuration of nnU-Net
    If Transform is not None it will be applied to the image after loading.
    Segmentations will be converted to np.uint32!
    :param is_seg:
    :param transform:
    :param input_filename:
    :param output_filename_truncated: do not use a file ending for this one! Example: output_name='./converted/image1'. This
    function will add the suffix (_0000) and file ending (.nii.gz) for you.
    :param spacing:
    :return:
    """
    #img = io.imread(input_filename)

    if transform is not None:
        img = transform(img)

    if len(img.shape) == 2:  # 2d image with no color channels
        img = img[None, None]  # add dimensions
    else:
        assert len(img.shape) == 3, "image should be 3d with color channel last but has shape %s" % str(img.shape)
        # we assume that the color channel is the last dimension. Transpose it to be in first
        img = img.transpose((2, 0, 1))
        # add third dimension
        img = img[:, None]

    # image is now (c, x, x, z) where x=1 since it's 2d
    if is_seg:
        assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'

    for j, i in enumerate(img):

        if is_seg:
            i = i.astype(np.uint32)

        itk_img = sitk.GetImageFromArray(i)
        itk_img.SetSpacing(list(spacing)[::-1])
        if not is_seg:
            sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")
def ExtractPng(filedir):
    
    entries = Path(filedir)
    fname, id, pathlist = [],[],[]
    idTick = 0
    images = []
    ids = []
    for entry in entries.iterdir():
        print(entry)
        segg = sitk.ReadImage(str(entry), imageIO= "PNGImageIO")
        seg = sitk.GetArrayFromImage(segg)
        print(seg.shape)
        images.append(seg)
        name =  entry.name[4:12]
        ids.append([name])
    return images, ids

def main():
    # Reading dicoms to nifti, comment out
    '''
    mask_dicom_path = "/Users/oliviamurray/Documents/PhD/MISTIE/my segmentations /further testing/testing masks"
    ct_dicom_path = "/Users/oliviamurray/Documents/PhD/MISTIE/my segmentations /further testing/testing scans"
    haem_dicom_path = "/Users/oliviamurray/Documents/PhD/MISTIE/my segmentations /haematomas"
    mask_array, mask_ids = ReadNiftiFolder(mask_dicom_path)
    CT_array, CT_ids = ReadNiftiFolder(ct_dicom_path)
    haematoma_array, haematoma_ids = ReadNiftiFolder(haem_dicom_path)
    SaveData(mask_array, mask_ids,"/Users/oliviamurray/Documents/PhD/MISTIE/my segmentations /further testing/internal_capsule_masks.npz" )
    SaveData(CT_array, CT_ids,"/Users/oliviamurray/Documents/PhD/MISTIE/my segmentations /further testing/ct_scans.npz" )
    #SaveData(haematoma_array, haematoma_ids,"/Users/oliviamurray/Documents/PhD/MISTIE/my segmentations /haematoma_masks.npz" )
    # end of section

    # converting and saving 2D arrays
    ic_array, ic_ids = UnpackNpz("/Users/oliviamurray/Documents/PhD/MISTIE/my segmentations /further testing/internal_capsule_masks.npz")

    haem_array, haem_ids = UnpackNpz("/Users/oliviamurray/Documents/PhD/MISTIE/my segmentations /haematoma_masks.npz")
    ct_array, ct_ids = UnpackNpz("/Users/oliviamurray/Documents/PhD/MISTIE/my segmentations /further testing/ct_scans.npz" )
    # order arrays
    ic_ordered =[]
    for id in ct_ids:
        idic = ic_ids.tolist().index(id)
        ic_ordered.append(ic_array[idic])
    ic_slices = GetSlicesArray(ic_ordered)
    haem_slices = GetSlicesArray(haem_array)
    Save2DData(ct_array, ct_ids, ic_ordered, ct_ids, ic_slices, ct_ids, path = "/Users/oliviamurray/Documents/PhD/MISTIE/my segmentations /further testing/ic_ct_2d.npz" )
    # end of section

   '''
    img2, ids2 = ExtractPng('/Users/oliviamurray/Downloads/Fetoscopy Placenta Dataset/Vessel_segmentation_annotations/video02/images')
    img3,ids3 = ExtractPng('/Users/oliviamurray/Downloads/Fetoscopy Placenta Dataset/Vessel_segmentation_annotations/video03/images')
    img4,ids4 = ExtractPng('/Users/oliviamurray/Downloads/Fetoscopy Placenta Dataset/Vessel_segmentation_annotations/video04/images')
    img5,ids5 = ExtractPng('/Users/oliviamurray/Downloads/Fetoscopy Placenta Dataset/Vessel_segmentation_annotations/video05/images')
    img6,ids6 = ExtractPng('/Users/oliviamurray/Downloads/Fetoscopy Placenta Dataset/Vessel_segmentation_annotations/video06/images')
    img = img2 +img3 +img4 +img5 +img6
    id = ids2 + ids3 + ids4 + ids5 +ids6
    img = np.array(img)
    id = np.array(id)
    # converting 2d to pseudo 3d nifti
    #path_2d = "/Users/oliviamurray/Documents/PhD/MISTIE/my segmentations /further testing/ic_ct_2d.npz"
    #cts, masks, ids, slices= Unpack2DNpz(path_2d)
    
    for i in range(0, len(img)):
        title = "/Users/oliviamurray/Documents/PhD/MISTIE/MedICSS/FetRegNifti/Train" + str(id[i]) 
        convert_2d_image_to_nifti(img[i], title, is_seg=False)
    for i in range(0, len(id)):
        title = "/Users/oliviamurray/Documents/PhD/MISTIE/MedICSS/FetRegNifti/Train" + str(id[i]) + "_mask"
        convert_2d_image_to_nifti(id[i], title, is_seg=True)


    '''    
    test_ids, test_cts, test_masks, test_slices, test_slice_ids = [],[],[],[],[]
    for id in ids3d:
        if id not in ids:
            if id in mask_ids3d:
                test_ids.append(id)
                ct_place = ids3d.tolist().index(id)
                mask_place = mask_ids3d.tolist().index(id)
                slice_place = slices_ids3d.tolist().index(id)
                test_cts.append(cts3d[ct_place])
                test_masks.append(masks3d[mask_place])
                test_slices.append(slices3d[slice_place])
    print(len(test_ids), len(test_cts), len(test_slices), len(test_masks))
    print(len(cts3d), len(test_cts), len(cts))
    save_path = '/Users/oliviamurray/Documents/PhD/MISTIE/mask_data/adrian_anti_matched_2d_training.npz'
    #d = Save2DData(test_cts, test_ids,test_masks, test_ids, test_slices, test_ids, save_path )
    '''
    """
    test_load = nib.load("/Users/oliviamurray/Documents/PhD/MISTIE/my segmentations /training/2D masks/2027_0000.nii.gz").get_fdata()
    test_load_m = nib.load('/Users/oliviamurray/Documents/PhD/MISTIE/training_data/maskmatchedNifti/2027.nii.gz').get_fdata()
    print(test_load.shape)
    print(test_load_m.shape)
    test = test_load[:,:,0]
    test_m = test_load_m[:,:,0]
    plt.imshow(test, cmap= 'gray')
    plt.imshow(test_m, alpha = 0.5)
    plt.show()
    """
if __name__ == '__main__':
 main()