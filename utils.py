import os
from pathlib import Path
from SimpleITK.SimpleITK import And
import numpy as np
import scipy
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.segmentation import join_segmentations
import pandas as pd
from scipy.ndimage import label
def GetSlices(seg_arr):
    values= []
    for i in range(0,len(seg_arr)):
        val = np.sum(seg_arr[i,:,:])
        if val >= 50750:
            values.append(i)
    return values

def GetSlicesArray(masks):
    slices_array =[]
    for image in masks:
        slices = GetSlices(image)
        slices_array.append((slices))
    return np.asarray(slices_array)
    
def GetArea(seg_arr, staple_slices, patient):
    values = []
    for i in staple_slices[patient]:
        
        val = np.sum(seg_arr[i,:,:])
        
        val/255
        values.append(val.astype(int))
    return values

def GetAreaArray(masks, masks_ids, staple_id, staple_slices):
    area_array =[]
    masks_ids_l = masks_ids.tolist()
    for id in staple_id:
       
        place_holder = masks_ids_l.index(id)
        
        areas = GetArea(masks[place_holder], staple_slices, place_holder)
        area_array.append(areas)
    return np.asarray(area_array)

def StapleGun(masks_a, masks_b, masks_c, ids_a, ids_b, ids_c):
    """
    slices_array_a = GetSlicesArray(masks_a)
    slices_array_b = GetSlicesArray(masks_b)
    slices_array_c = GetSlicesArray(masks_c)
    """
    #removed CR slices, for when I have all of the data sheets 
    path_a = "/Users/olivia/Documents/PhD/MISTIE/mask_data/CRdata.xlsx"
 
    slices_array_a = NewRemoveCoronaRadiataSlices(masks_a, ids_a, path_a, sheetname='adrian')
    slices_array_b = NewRemoveCoronaRadiataSlices(masks_b, ids_b, path_a, sheetname = 'paul')
    slices_array_c = NewRemoveCoronaRadiataSlices(masks_c, ids_c, path_a, sheetname = 'sacha') 
    

    staple_array = []
    slices_array = []
    id = []
    id_ticker = 0
    

    for i in range (0, len(ids_c)) :
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

def StapleGun4(masks_a, masks_b, masks_c, masks_d, ids_a, ids_b, ids_c, ids_d):
    """
    slices_array_a = GetSlicesArray(masks_a)
    slices_array_b = GetSlicesArray(masks_b)
    slices_array_c = GetSlicesArray(masks_c)
    """
    #removed CR slices, for when I have all of the data sheets 
    path_a = "/Users/olivia/Documents/PhD/MISTIE/mask_data/CRdata.xlsx"
 
    slices_array_a = NewRemoveCoronaRadiataSlices(masks_a, ids_a, path_a, sheetname='adrian')
    slices_array_b = NewRemoveCoronaRadiataSlices(masks_b, ids_b, path_a, sheetname = 'paul')
    slices_array_c = NewRemoveCoronaRadiataSlices(masks_c, ids_c, path_a, sheetname = 'sacha') 
    slices_array_d = NewRemoveCoronaRadiataSlices(masks_d, ids_d, path_a, sheetname = 'h')

    staple_array = []
    slices_array = []
    id = []
    id_ticker = 0
    

    for i in range (0, len(ids_d)) :
        id_ticker += 1
        staple_slices_array = []  
        slices_id =[]
        patient_id = ids_a[i]

        for j in slices_array_a[i]:
            if j in slices_array_b[i] and j in slices_array_c[i] and j in slices_array_d[i]:

                masks_a[i][j][masks_a[i][j]==255] = 1
                masks_b[i][j][masks_b[i][j]==255] = 1
                masks_c[i][j][masks_c[i][j]==255] = 1
                masks_d[i][j][masks_d[i][j]==255] = 1

                mask_a = sitk.GetImageFromArray(masks_a[i][j])
                mask_b = sitk.GetImageFromArray(masks_b[i][j])
                mask_c = sitk.GetImageFromArray(masks_c[i][j])
                mask_d = sitk.GetImageFromArray(masks_d[i][j])

                staple_filter = sitk.STAPLEImageFilter()
                staple_filter.SetForegroundValue(1)
                staple_image = staple_filter.Execute(mask_d ,mask_b, mask_c, mask_a)
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

def StapleGunPLIC(slices__a, slices__b, slices__c, slices__d, p_masks_a, p_masks_b, p_masks_c, p_masks_d, p_ids_a, p_ids_b, p_ids_c, p_ids_d ):
    """
    slices_array_a = GetSlicesArray(masks_a)
    slices_array_b = GetSlicesArray(masks_b)
    slices_array_c = GetSlicesArray(masks_c)
    """
    #removed CR slices, for when I have all of the data sheets 
    path_a = "/Users/olivia/Documents/PhD/MISTIE/mask_data/CRdata.xlsx"
 

    
    slices_array_a = remove_extra_ALIC(p_ids_a, slices__a, sheetname = "adrian")
    slices_array_b = remove_extra_ALIC(p_ids_b, slices__b, sheetname = "paul")
    slices_array_c = remove_extra_ALIC(p_ids_c, slices__c, sheetname = "sacha")
    slices_array_d = remove_extra_ALIC(p_ids_d, slices__d, sheetname = "h")
    
    """
    slices_array_a = slices__a
    slices_array_b = slices__b
    slices_array_c = slices__c
    slices_array_d = slices__d
    """
    staple_array = []
    slices_array = []
    id = []
    id_ticker = 0
    

    for i in range (0, len(p_ids_d)) :
        id_ticker += 1
        staple_slices_array = []  
        slices_id =[]
        patient_id = p_ids_a[i]

        for j in slices_array_a[i]:
            if j in slices_array_b[i] and j in slices_array_c[i] and j in slices_array_d[i]:


                p_masks_a[i][j][p_masks_a[i][j]!=0] = 1
                p_masks_b[i][j][p_masks_b[i][j]!=0] = 1
                p_masks_c[i][j][p_masks_c[i][j]!=0] = 1
                p_masks_d[i][j][p_masks_d[i][j]!=0] = 1

                print(" unique: ",np.unique(p_masks_a[i][j]))
                print(" unique: ",np.unique(p_masks_b[i][j]))
                print(" unique: ",np.unique(p_masks_c[i][j]))
                print(" unique: ",np.unique(p_masks_d[i][j]))

                mask_a = sitk.GetImageFromArray(p_masks_a[i][j])
                mask_b = sitk.GetImageFromArray(p_masks_b[i][j])
                mask_c = sitk.GetImageFromArray(p_masks_c[i][j])
                mask_d = sitk.GetImageFromArray(p_masks_d[i][j])

                staple_filter = sitk.STAPLEImageFilter()
                staple_filter.SetForegroundValue(1)
                staple_image = staple_filter.Execute(mask_d ,mask_b, mask_c, mask_a)
                staple_image = staple_image > 0.5                                
                staple_slice_array = sitk.GetArrayFromImage(staple_image)
                iter = staple_filter.GetElapsedIterations()
                print("Patient: ", p_ids_d[i], ", Slice: ", j)
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
def UnpackSlices(path):
    data = np.load(path, allow_pickle=True)
    slices = data['slices']
    ids = data['ids']
    return slices, ids
def OnesNZeros(mask_array):
    mask_array_copy = mask_array
    slices_array = GetSlicesArray(mask_array_copy)
    for i in range(0, len(mask_array_copy)):
        for slic in slices_array[i]:
            mask_array_copy[i][slic][mask_array_copy[i][slic]==255] = 1
    return mask_array_copy

def HasCoronaRadiata(csv_path, sheetname):
    data_frame = pd.read_excel(csv_path, sheet_name= sheetname, usecols = ['PRIME ID ', 'CR distance to hematoma (mm)'])
    
    CR =data_frame['CR distance to hematoma (mm)'].tolist()
    ids = data_frame['PRIME ID '].tolist()
    ids = [str(x) for x in ids]
    print("ids: ",ids)
    ind = []
    for i in range(0,len(CR)):
        if CR[i] !=0:
            ind.append(i)

    patients=[]
    for i in ind:
        print("id data: ", ids[i])
        patient = ids[i][0:4]
        patients.append(patient)
    print("patients: ", patients)
    return np.asarray(patients, dtype = int)

def RemoveCoronaRadiataSlices(mask_array, mask_ids, csv_path, sheetname):
    mask_ids_dummy = mask_ids.astype(int).tolist()
    patients = HasCoronaRadiata(csv_path, sheetname)
    slices_array = GetSlicesArray(mask_array)
    for id in patients:
        if id in mask_ids_dummy:
            
            ind = mask_ids_dummy.index(id)
            length= len(slices_array[ind])
            if id != 0:
               if len(slices_array[ind]) != 0:
                  slices_array[ind].pop(length-1)
         
    return slices_array

def NewRemoveCoronaRadiataSlices(mask_array, mask_ids, csv_path, sheetname):

    removed_slices = []

    slices_array = GetSlicesArray(mask_array)
    mask_ids_dummy = mask_ids.astype(int).tolist()
    data_frame = pd.read_excel(csv_path, sheet_name= sheetname, usecols = ['cr missing','cr upside down' ])
    CR_upsidedown = data_frame['cr upside down'].astype(int).tolist()
    CR_missing = data_frame['cr missing'].astype(int).tolist()
    

    for id in mask_ids_dummy:
        
        if id in CR_upsidedown:
            place_holder = mask_ids_dummy.index(id)
            if len(slices_array[place_holder]) != 0:
                slices_array[place_holder].pop(0)
        elif id in CR_missing:
            slices_array = slices_array
        else:
            place_holder = mask_ids_dummy.index(id)
            if len(slices_array[place_holder]) != 0:
                slices_array[place_holder].pop(len(slices_array[place_holder])-1)
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
def IdSkip2(ids_a,ids_b, masks_a, masks_b):
    mask_a = masks_a.tolist()
    mask_b = masks_b.tolist()
    
    id_a = ids_a.tolist()
    id_b = ids_b.tolist()
    print("before: ", len(id_a), len(id_b), len(set(id_a)), len(set(id_b)))
    ids_skip =[]
    for id in ids_b:
        if id not in ids_a:
            ids_skip.append(id)
    for idb in ids_a:
        if idb not in ids_b:
            ids_skip.append(idb)
    #ids_skip_set = set(id_a)- set(id_b)
    print("idskip: ", ids_skip)
    for idsk in ids_skip:
        if idsk in ids_a:
            i = id_a.index(idsk)
            mask_a.pop(i)
            id_a.pop(i)
        elif idsk in ids_b:
            i = id_b.index(idsk)
            mask_b.pop(i)
            id_b.pop(i)
    print("after: ", len(id_a), len(id_b))
    return np.asarray(mask_a), np.asarray(id_a), np.asarray(mask_b), np.asarray(id_b)
def IdSkip4(ids_a,ids_b,ids_c, ids_d, masks_a, masks_b, masks_c, masks_d):
    mask_a = masks_a.tolist()
    mask_b = masks_b.tolist()
    mask_c = masks_c.tolist()
    mask_d = masks_d.tolist()
    id_a = ids_a.tolist()
    id_b = ids_b.tolist()
    id_c = ids_c.tolist()
    id_d = ids_d.tolist()
    ids_skip =[]
    for id in ids_a:
        if id not in ids_b or id not in ids_c or id not in ids_d:
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
        if id in ids_d:
            i = id_d.index(id)
            mask_d.pop(i)
            id_d.pop(i)


    return np.asarray(mask_a), np.asarray(id_a), np.asarray(mask_b), np.asarray(id_b), np.asarray(mask_c), np.asarray(id_c), np.asarray(mask_d), np.asarray(id_d)

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
        #for j in range(0, len(staple_slices[i])):
        for j in range(0, len(staple_array[i])):
            patient_id = staple_ids[i]
            slices_arr = staple_slices[i]
            place_holder = id_a.index(patient_id)
            ind = slices_arr[j]
            
            #print("place holder: ", place_holder, "ind: ", ind, "i: ", i, "j: ", j )
            dice_coeff = compute_dice_coefficient(masks_a[place_holder][ind,:,:], staple_array[i][j])
            dice_coeff = dice_coeff * 100
            dices_array.append(dice_coeff)
        dice_array.append(dices_array)   
    return dice_array

    
def DiceCaster2(masks_b_, ids_b_, slices_b, slices_b_ids, masks_a_, ids_a_, slices_a, slices_a_ids):
    print("befored: ",len(ids_a_), len(ids_b_))
    masks_a, ids_a, masks_b, ids_b = IdSkip2(ids_a_, ids_b_, masks_a_, masks_b_)
    print(ids_a, ids_b)
    print("afterd: ",len(ids_a) ,len(ids_b))
    id_a = ids_a.tolist()
    dice_array =[]
    for i in range(0, len(masks_b)):
        dices_array =[]
        patient_id = ids_b[i]
        print(patient_id)
        place_holder = id_a.index(patient_id)
        if patient_id in slices_a_ids and patient_id in slices_b_ids:
            slices_place_a = slices_a_ids.tolist().index(patient_id)
            slices_place_b = slices_b_ids.tolist().index(patient_id)

            print(slices_b[i])
            for j in slices_b[slices_place_b]:
                
                
                print(" j: ", j, slices_b[slices_place_b], slices_a[slices_place_a])
                if j in slices_a[slices_place_a]:
                    print("true")
                    
    
            

                    
        
                    masks_a[i][j][masks_a[i][j] != 0] =1
                    masks_b[i][j][masks_b[i][j] !=0 ] =1

                    dice_coeff = compute_dice_coefficient(masks_a[i][j], masks_b[i][j])
                    print("dice: ", dice_coeff)
                    dice_coeff = dice_coeff * 100
                    dices_array.append(dice_coeff)
                else:
                    print("false")
            dice_array.append(dices_array)   
    return dice_array, ids_a

def DiceCaster3(masks_b, ids_b, slices_b, masks_a, ids_a, slices_a):
    
    id_a = ids_a.tolist()
    dice_array =[]
    for i in range(0, len(masks_b)):
        dices_array =[]
        patient_id = ids_b[i]
        print(patient_id)
        patient_id = ids_b[i]
        place_holder = id_a.index(patient_id)
        #if len(slices_b)!=  0 and len(slices_a[place_holder]) != 0:
        for j in slices_b[i]:

            print(" j: ", j, slices_b[i], slices_a[i])
            if j in slices_a[place_holder]:
                print("true")
                
                #for j in range(0, len(slices_b[i])):
                #for j in range(0, len(staple_array[i])):
        
                
                #slices_arr = slices_b[i]
                
                #ind = slices_arr[j]
                #if ind in slices_a[i]:

                #print("place holder: ", place_holder, "ind: ", ind, "i: ", i, "j: ", j )
                dice_coeff = compute_dice_coefficient(masks_a[place_holder][j,:,:], masks_b[i][j,:,:])
                dice_coeff = dice_coeff * 100
                dices_array.append(dice_coeff)
            else:
                print("false")
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

def GetSliceOverlap(masks_a, ids_a, masks_b, ids_b, masks_c, ids_c):
    percent = []
    path = "/Users/olivia/Documents/PhD/MISTIE/mask_data/CRdata.xlsx"
    '''
    slices_a = GetSlicesArray(masks_a)
    slices_b = GetSlicesArray(masks_b)
    slices_c = GetSlicesArray(masks_c)
    '''
    slices_a = NewRemoveCoronaRadiataSlices(masks_a, ids_a, path, sheetname = 'adrian')
    slices_b = NewRemoveCoronaRadiataSlices(masks_b, ids_b, path, sheetname = 'paul')
    slices_c = NewRemoveCoronaRadiataSlices(masks_c, ids_c, path, sheetname = 'sacha')

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

def GetSliceOverlap4(masks_a, ids_a, masks_b, ids_b, masks_c, ids_c, masks_d, ids_d):
    percent = []
    path = "/Users/olivia/Documents/PhD/MISTIE/mask_data/CRdata.xlsx"
    '''
    slices_a = GetSlicesArray(masks_a)
    slices_b = GetSlicesArray(masks_b)
    slices_c = GetSlicesArray(masks_c)
    '''
    slices_a = NewRemoveCoronaRadiataSlices(masks_a, ids_a, path, sheetname = 'adrian')
    slices_b = NewRemoveCoronaRadiataSlices(masks_b, ids_b, path, sheetname = 'paul')
    slices_c = NewRemoveCoronaRadiataSlices(masks_c, ids_c, path, sheetname = 'sacha')
    slices_d = NewRemoveCoronaRadiataSlices(masks_d, ids_d, path, sheetname = 'h')

    total_intersection = 0
    total_union = 0
    for i in ids_a:
        ind_a = ids_a.tolist().index(i)
        ind_p = ids_b.tolist().index(i)
        ind_s = ids_c.tolist().index(i)
        ind_h = ids_d.tolist().index(i)
        print(ind_a, ind_p, ind_s, ind_h)
        intersection = set(slices_a[ind_a]) & set(slices_b[ind_p]) & set(slices_c[ind_s]) & set(slices_d[ind_h])
        union = set(slices_a[ind_a])|set(slices_b[ind_p])|set(slices_c[ind_s])|set(slices_d[ind_h])
        smooth = 0.001
        percentage = len(intersection)/(len(union)+smooth)
        percent.append(percentage)
        total_intersection += len(intersection)
        total_union += len(union)

    total_percentage = total_intersection/total_union
    return total_percentage, percent
def SaveStapleData(inputs, ids, slices):
    path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/staple_masks.npz'
    ids = np.array(ids)
    #print("final shape: ", inputs.shape,  ids.shape)
    np.savez(path, masks = inputs,  ids = ids, slices = slices) 
    print("Saved data") 

def ImageFlipper(image):
    image = np.flip(image, axis=0)        # flip CC
    #image = np.flip(image, axis=2)        # flip LR
    return image
def ImageFlipperUD(image):
    image = np.flip(image, axis=1) 
    return image

def SelectCT(dir, mask, mask2, mask3, mask4):
    # dir is path to diagnostic CT
    
    scans = Path(dir)
    # scan is CT scan file eg BONE
    for scan in scans.iterdir():
        if scan.name != '.DS_Store':
            print(scan.name)
            
            mask_len = len(mask)
            mask2_len = len(mask2)
            mask3_len = len(mask3)
            mask4_len = len(mask4)
            entries = Path(scan)
            # entry is 
            
            for entry in entries.iterdir():
                #print(entry.name)
                keep_list =[]
                if entry.name != '.DS_Store':
                    size_list = []
                    images = Path(entry)
                    for image in images.iterdir():
                        if image.name != '.DS_Store':
                            name = image.stem
                            
                            size = len(name)
                            z = name[size -2: size ]
                            
                            if z.startswith('.'):
                                z = z[1:]
                                
                                size_list.append(int(z))
                            else:
                                size_list.append(int(z))
                    #print("size list: ", size_list)
                    maximum = max(size_list)
                    print("maximum: ", maximum, "len: ", mask_len)
                    if mask_len == maximum or mask2_len == maximum or mask3_len == maximum or mask4_len == maximum:
                        keep_list.append(image.parent)
                        print("match!")
                    elif mask_len == maximum +1 or mask_len == maximum -1 or mask2_len == maximum +1 or mask2_len == maximum -1 or mask3_len == maximum +1 or mask3_len == maximum -1 or mask4_len == maximum +1 or mask4_len == maximum -1:
                        keep_list.append(image.parent)
                        print("match +- 1!")
    return keep_list
from scipy.ndimage import label, generate_binary_structure, maximum
def ALICRemover(mask_array_ALIC ,CR_slice_array):
    mask_array = mask_array_ALIC.copy()
    s = generate_binary_structure(2,2)
    s2 = [[1,1],
          [1,1]]
    for i in range(0, len(mask_array)):
        
        slices = CR_slice_array[i]
        print(len(mask_array[i]), len(slices))
        for slic in slices:
            print("slic: ", slic)
            
            labeled_mask, num_labels = label(mask_array[i][slic], structure = s)
            labs = []
            sums = []
            for j in range (1, num_labels+1):
                
                sum = np.sum(labeled_mask[labeled_mask == j])
                labs.append(j)
                sums.append(sum)
            print(sums)
            print(labs)
            if len(labs) != 0:

                max = np.max(sums)
                ind = sums.index(max)
                max_label = labs[ind]

              
                
                labeled_mask[labeled_mask != max_label] =0 
                labeled_mask[labeled_mask != 0] =1
                print("match: ", len(mask_array[i]))
                mask_array[i][slic] = labeled_mask
    return mask_array

def remove_extra_ALIC( mask_ids, CR_slices, sheetname ):
    print("save check")
    tidy_slices = CR_slices.copy()
    alic_path = "/Users/oliviamurray/Documents/PhD/MISTIE/mask_data/ALIC_data.xlsx"
    alic_frame = pd.read_excel(alic_path, sheet_name= sheetname, usecols = ['id','slice_index' ])
    ids = alic_frame['id'].astype(str).tolist()
    slice_index = alic_frame['slice_index'].astype(str).tolist()
    for d in range(0, len(ids)):
        id = ids[d]
        if id in mask_ids:
                
            print(id)
            placehold = mask_ids.tolist().index(id)
            
            slices = CR_slices[placehold].copy()
            if len(slices) != 0:

                print("slices before: ", slices)
                index_array = []
                print("test: ", slice_index[d])
                
                for i in range(1, len(slice_index[d])-1):
                    
                    index_array.append(int(slice_index[d][i]))
                print("to del: ", index_array)
                for index in sorted(index_array, reverse=True):
                    del slices[index]
                print("slices after: ", slices)
                print("tidy before: ", tidy_slices[placehold])
                tidy_slices[placehold] = slices
                print("tidy after: ", tidy_slices[placehold])

    return tidy_slices
    
def SlicesSkip(ids, slices_a, ids_a, slices_b, ids_b, slices_c,ids_c, slices_d,ids_d):
    skipped_a, skipped_b, skipped_c, skipped_d = [],[],[],[]
    for id in ids:
        pl1 = ids_a.tolist().index(id)
        pl2 = ids_b.tolist().index(id)
        pl3 = ids_c.tolist().index(id)
        pl4 = ids_d.tolist().index(id)
        skipped_a.append(slices_a[pl1])
        skipped_b.append(slices_b[pl2])
        skipped_c.append(slices_c[pl3])
        skipped_d.append(slices_d[pl4])
    return skipped_a, skipped_b, skipped_c, skipped_d

def Save2DData(cts, ct_ids, masks, mask_ids, slices_arr, slices_ids, path):
    ct_2d_arr , mask_2d_arr, id_2d_arr, slice_2d_arr = [], [] ,[], []
    for i in range(len(ct_ids)):
        patient = ct_ids[i]
        print(patient)
        skip = [ '2338', '2054', '2053' , '2212', '2318', '2066', '2451']
        if patient not in skip:

            place_slice = slices_ids.index(patient)
            place_mask = mask_ids.index(patient)
            slices = slices_arr[place_slice]
            masks[place_mask] = masks[place_mask].astype(float)
            for slic in slices:
                window = 80
                level = 40
                print("slice: ", slic)
                vmin = (level/2) - window
                vmax = (level/2) + window
                cts[i][slic][cts[i][slic]>vmax] = vmax
                cts[i][slic][cts[i][slic]<vmin] = vmin
                ct_2d = cts[i][slic]

                masks[place_mask][slic][masks[place_mask][slic] != 0] = 1
                masks[place_mask][slic][masks[place_mask][slic] == 0] = np.nan
                mask_2d = masks[place_mask][slic]
                id_2d = ct_ids[i]
                ct_2d_arr.append(ct_2d)
                mask_2d_arr.append(mask_2d)
                id_2d_arr.append(id_2d)
                slice_2d_arr.append(slic)

    np.savez(path, cts = ct_2d_arr, masks = mask_2d_arr,  ids = id_2d_arr, slices = slice_2d_arr) 
def Save2DCT(cts, ct_ids, slices_arr, slices_ids, path):
    ct_2d_arr , id_2d_arr, slice_2d_arr = [], [] ,[], []
    for i in range(len(ct_ids)):
        patient = ct_ids[i]
        print(patient)
        skip = [ '2338', '2054', '2053' , '2212', '2318', '2066', '2451']
        if patient not in skip:

            place_slice = slices_ids.index(patient)
            
            slices = slices_arr[place_slice]
           
            for slic in slices:
                window = 80
                level = 40
                print("slice: ", slic)
                vmin = (level/2) - window
                vmax = (level/2) + window
                cts[i][slic][cts[i][slic]>vmax] = vmax
                cts[i][slic][cts[i][slic]<vmin] = vmin
                ct_2d = cts[i][slic]

                
                id_2d = ct_ids[i]
                ct_2d_arr.append(ct_2d)
                
                id_2d_arr.append(id_2d)
                slice_2d_arr.append(slic)

    np.savez(path, cts = ct_2d_arr,  ids = id_2d_arr, slices = slice_2d_arr) 
def Unpack2DNpz(path):
    
    data = np.load(path, allow_pickle=True)
    cts = data['cts']
    masks = data['masks']
    ids = data['ids']
    slices = data['slices']
    return cts, masks, ids, slices

def GetInvolved(ids_a, ids_b,masks_a, masks_b ):
    # according to adrian
    a_info_path = "/Users/oliviamurray/Documents/PhD/MISTIE/PRIME_ICH_Data.xlsx"
    mask_sk_a, ids_sk_a, mask_sk_b, ids_sk_b = IdSkip2(ids_a, ids_b, masks_a, masks_b)
    a_involve =[]
    data_frame = pd.read_excel(a_info_path, sheet_name= "APJ", usecols = ['PRIME_ID', 'PLIC involvement '])
    ID = data_frame['PRIME_ID'].tolist()
    print("id len: ", len(ID))
    involve = data_frame['PLIC involvement '].tolist()
    for i in range(0, len(ID)):
        
        if ID[i][0:4] in ids_sk_a:
            print(ID[i][0:4])
            a_involve.append(involve[i])
    return a_involve
