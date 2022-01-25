import os
from SimpleITK.SimpleITK import And
import numpy as np
import scipy
import SimpleITK as sitk
def GetSlices(seg_arr):
    values= []
    for i in range(0,len(seg_arr)):
        val = np.sum(seg_arr[i,:,:])
        if val !=0:
            values.append(i)
    return values

def GetSlicesArray(masks):
    slices_array =[]
    for image in masks:
        slices = GetSlices(image)
        slices_array.append((slices))
    return np.asarray(slices_array)



def StapleGun(masks_a, masks_b, masks_c):

    slices_array_a = GetSlicesArray(masks_a)
    slices_array_b = GetSlicesArray(masks_b)
    slices_array_c = GetSlicesArray(masks_c)
    staple_array = []
    id = []
    id_ticker = 0
    for i in range (0, 84):
        # for each patient, generate overlap array
        # get rid of j loop  and use list comprehension for the overlap array to select 'j' value instead

        for j in slices_array_a[i]:
            staple_slices_array = []
            #if slices_array_a[i][j] == slices_array_b[i][j] == slices_array_c[i][j]:
            #if (slices_array_a[i][j] == np.asarray(slices_array_b[i]).any()) and (slices_array_a[i][j] == np.asarray(slices_array_c[i]).any()) :
            if j in slices_array_b[i] and j in slices_array_c[i]:
                id_ticker += 1
                mask_a = sitk.GetImageFromArray(masks_a[i][j])
                mask_b = sitk.GetImageFromArray(masks_b[i][j])
                mask_c = sitk.GetImageFromArray(masks_c[i][j])


                staple_filter = sitk.STAPLEImageFilter()
                staple_filter.SetForegroundValue(1)

                staple_image = staple_filter.Execute(mask_a,mask_b, mask_c)
                staple_image = staple_image > 0.5                                
                staple_slice_array = sitk.GetArrayFromImage(staple_image)
                iter = staple_filter.GetElapsedIterations()
                print("Patient: ", i, ", Slice: ", j)
                print("iter: ", iter)
                staple_slices_array.append(staple_slice_array)
                id.append(id_ticker)

            staple_array.append(staple_slices_array)
    return staple_array, id
def UnpackNpz(path):
    data = np.load(path, allow_pickle=True)
    masks = data['masks']
    ids = data['ids']
    return masks, ids
def main():
 adrian_path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/adrian_masks.npz'
 sacha_path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/sacha_masks.npz'
 paul_path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/paul_masks.npz'

 masks_paul, ids_paul = UnpackNpz(paul_path)
 masks_adrian, ids_adrian = UnpackNpz(adrian_path)
 masks_sacha, ids_sacha = UnpackNpz(sacha_path)

 staple_array, staple_ids = StapleGun(masks_a= masks_adrian, masks_b= masks_paul, masks_c= masks_sacha)
 print("staple_array shape: ", staple_array.shape, "staple_ids = ", staple_ids)
  #print(masks[i].shape)
  #print(slices)
'''
 data_s = np.load(sacha_path, allow_pickle=True)
 masks_s = data['masks']
 ids_s = data['ids']
 print("sacha: ")
 print(masks_s.shape)
 for i in range(0,84):
  slices = GetSlices(masks_s[i])
  print(masks[i].shape)
  print(slices)
  
 data_p = np.load(sacha_path, allow_pickle=True)
 masks_p = data['masks']
 ids_p = data['ids']
 print(masks_s.shape)
 for i in range(0,84):
  slices = GetSlices(masks_p[i])
  print("paul: ")
  print(masks[i].shape)
  print(slices)
  
  '''


if __name__ == '__main__':
 main()