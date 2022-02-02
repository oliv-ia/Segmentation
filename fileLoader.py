import os
from SimpleITK.SimpleITK import And
import numpy as np
import scipy
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.segmentation import join_segmentations
import pandas as pd
def GetSlices(seg_arr):
    values= []
    for i in range(0,len(seg_arr)):
        val = np.sum(seg_arr[i,:,:])
        if val >=  20000 :
            values.append(i)
    return values

def GetSlicesArray(masks):
    slices_array =[]
    for image in masks:
        slices = GetSlices(image)
        slices_array.append((slices))
    return np.asarray(slices_array)


def StapleGun(masks_a, masks_b, masks_c, ids_a, ids_b, ids_c):

    slices_array_a = GetSlicesArray(masks_a)
    slices_array_b = GetSlicesArray(masks_b)
    slices_array_c = GetSlicesArray(masks_c)
    """
    #removed CR slices, for when I have all of the data sheets 
    path_a = "path"
    path_b = "path"
    path_c = "path"
    slices_array_a = RemoveCoronaRadiataSlices(masks_a, ids_a, path_a)
    slices_array_b = RemoveCoronaRadiataSlices(masks_b, ids_b, path_b)
    slices_array_c = RemoveCoronaRadiataSlices(masks_c, ids_c, path_c) 
    """

    staple_array = []
    slices_array = []
    id = []
    id_ticker = 0
    

    for i in range (0, 77) :
        id_ticker += 1
        staple_slices_array = []  
        slices_id =[]
        patient_id = ids_a[i]

        for j in slices_array_a[i]:
            if j in slices_array_b[i] and j in slices_array_c[i]:

                masks_a[i][j][masks_a[i][j]==255] = 1
                masks_b[i][j][masks_b[i][j]==255] = 1
                masks_c[i][j][masks_c[i][j]==255] = 1

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
                slices_id.append(j)
                
        id.append((patient_id))
        staple_array.append(staple_slices_array)
        slices_array.append(slices_id)
        
    return staple_array, id, slices_array

def UnpackNpz(path):
    data = np.load(path, allow_pickle=True)
    masks = data['masks']
    ids = data['ids']
    return masks, ids

def OnesNZeros(mask_array):
    mask_array_copy = mask_array
    slices_array = GetSlicesArray(mask_array_copy)
    for i in range(0, len(mask_array_copy)):
        for slic in slices_array[i]:
            mask_array_copy[i][slic][mask_array_copy[i][slic]==255] = 1
    return mask_array_copy

def HasCoronaRadiata(csv_path):
    data_frame = pd.read_excel(csv_path, sheet_name='Sheet1', usecols = ['PRIME_ID ', 'CR_distance_to_hematoma_(mm)'])
    
    CR =data_frame['CR_distance_to_hematoma_(mm)'].tolist()
    ids = data_frame['PRIME_ID '].tolist()
    ind = []
    for i in range(0,len(CR)):
        if CR[i] !=0:
            ind.append(i)
    patients=[]
    for i in ind:
        patient = ids[i][0:4]
        patients.append(patient)

    return np.asarray(patients, dtype = int)

def RemoveCoronaRadiataSlices(mask_array, mask_ids, csv_path):
    mask_ids_dummy = mask_ids.astype(int).tolist()
    patients = HasCoronaRadiata(csv_path)
    slices_array = GetSlicesArray(mask_array)
    for id in patients:
        if id in mask_ids_dummy:
            
            ind = mask_ids_dummy.index(id)
            length= len(slices_array[ind])
            if id != 0:
             slices_array[ind].pop(length-1)
         
    return slices_array

  


def IdSkip(ids_a,ids_b,ids_c, masks_a, masks_b, masks_c):
    mask_a = masks_a.tolist()
    mask_b = masks_b.tolist()
    mask_c = masks_c.tolist()
    id_a = ids_a.tolist()
    id_b = ids_b.tolist()
    id_c = ids_c.tolist()
    ids_skip =[]
    for id in ids_a:
        if id not in ids_b or id not in ids_c:
            ids_skip.append(id)
    for id in ids_skip:
        if id in ids_a:
            i = id_a.index(id)
            mask_a.pop(i)
            id_a.pop(i)

        if id in ids_b:
            i = id_b.index(id)
            mask_b.pop(i)
            id_b.pop(i)

        if id in ids_c:
            i = id_c.index(id)
            mask_c.pop(i)
            id_c.pop(i)

    return np.asarray(mask_a), np.asarray(id_a), np.asarray(mask_b), np.asarray(id_b), np.asarray(mask_c), np.asarray(id_c)

def dice_loss(input, target):
    
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    print("iflat: ", iflat, "tflat: ", tflat)
    intersection = (iflat * tflat).sum(-1)
    print(intersection)
    print(iflat.sum(-1) + tflat.sum(-1))
    return ((2. * intersection + smooth) /
              (iflat.sum(-1) + tflat.sum(-1) + smooth))


def compute_dice_coefficient(mask_pred, mask_gt):

  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum 

def DiceCaster(staple_array, staple_ids, staple_slices, masks_a, ids_a):
    
    id_a = ids_a.tolist()
    dice_array =[]
    for i in range(0, len(staple_array)):
        dices_array =[]
        for j in range(0, len(staple_array[i])):
            patient_id = staple_ids[i]
            slices_arr = staple_slices[i]
            place_holder = id_a.index(patient_id)
            ind = slices_arr[j]
            
            #print("place holder: ", place_holder, "ind: ", ind, "i: ", i, "j: ", j )
            dice_coeff = compute_dice_coefficient(masks_a[place_holder][ind,:,:], staple_array[i][j])
            dices_array.append(dice_coeff)
        dice_array.append(dices_array)   
    return dice_array

def AverageDice(dice_array):
    average = []
    for patient in dice_array:
        cleanedList = [x for x in patient if str(x) != 'nan']
        sum= np.sum(cleanedList)
        length = len(cleanedList)
        avg = sum/length
        average.append(avg)

    cleanedAverage = [x for x in average if str(x) != 'nan']
    length = len(cleanedAverage)
    sum = np.sum(cleanedAverage)
    total_avg = sum/length
    return total_avg, average

def GetSliceOverlap(slices_a, ids_a, slices_b, ids_b, slices_c, ids_c):
    percent = []
    
    total_intersection = 0
    total_union = 0
    for i in ids_a:
        ind_a = ids_a.tolist().index(i)
        ind_p= ids_b.tolist().index(i)
        ind_s = ids_c.tolist().index(i)
        print(ind_a, ind_p, ind_s)
        intersection = set(slices_a[ind_a]) & set(slices_b[ind_p]) & set(slices_c[ind_s])
        union = set(slices_a[ind_a])|set(slices_b[ind_p])|set(slices_c[ind_s])
        smooth = 0.001
        percentage = len(intersection)/(len(union)+smooth)
        percent.append(percentage)
        total_intersection += len(intersection)
        total_union += len(union)

    total_percentage = total_intersection/total_union
    return total_percentage, percent

    
    

def main():
    adrian_path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/adrian_masks.npz'
    sacha_path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/sacha_masks.npz'
    paul_path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/paul_masks.npz'

    masks_paul, ids_paul = UnpackNpz(paul_path)
    masks_adrian, ids_adrian = UnpackNpz(adrian_path)
    masks_sacha, ids_sacha = UnpackNpz(sacha_path)
    print("unpacked")


    new_mask_paul,new_id_paul,new_mask_adrian,new_id_adrian,new_mask_sacha,new_id_sacha = IdSkip(ids_paul, ids_adrian, ids_sacha, masks_paul, masks_adrian, masks_sacha)
    staple_array, staple_ids, staple_slices = StapleGun(masks_a= new_mask_adrian, masks_b= new_mask_paul, masks_c= new_mask_sacha, ids_a= new_id_adrian, ids_b = new_id_paul, ids_c= new_id_sacha)

    dice_array_adrian = DiceCaster(staple_array, staple_ids, staple_slices, new_mask_adrian, new_id_adrian)
    adrian_total_avg, adrian_patient_avg = AverageDice(dice_array_adrian)

    dice_array_paul = DiceCaster(staple_array, staple_ids, staple_slices, new_mask_paul, new_id_paul)
    paul_total_avg, paul_patient_avg = AverageDice(dice_array_paul)

    dice_array_sacha = DiceCaster(staple_array, staple_ids, staple_slices, new_mask_sacha, new_id_sacha)
    sacha_total_avg, sacha_patient_avg = AverageDice(dice_array_sacha)

    print("avg adrian: ", adrian_total_avg, "avg paul: ", paul_total_avg, "avg sacha: ", sacha_total_avg)





if __name__ == '__main__':
 main()