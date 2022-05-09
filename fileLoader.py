import os

from SimpleITK.SimpleITK import And
import numpy as np
import scipy
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.segmentation import join_segmentations
import pandas as pd
from utils import GetSlicesArray, StapleGun, UnpackNpz, NewRemoveCoronaRadiataSlices, IdSkip, IdSkip4, DiceCaster, DiceCaster3, AverageDice, GetSliceOverlap


def main():
    adrian_path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/adrian_masks.npz'
    sacha_path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/sacha_masks.npz'
    paul_path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/paul_masks.npz'
    h_path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/h_masks.npz'

    masks_paul, ids_paul = UnpackNpz(paul_path)
    masks_adrian, ids_adrian = UnpackNpz(adrian_path)
    masks_sacha, ids_sacha = UnpackNpz(sacha_path)
    masks_h, ids_h = UnpackNpz(h_path)
    print("unpacked")

    # staple analysis
    new_mask_paul,new_id_paul,new_mask_adrian,new_id_adrian,new_mask_sacha,new_id_sacha, new_mask_h, new_id_h = IdSkip4(ids_paul, ids_adrian, ids_sacha, ids_h, masks_paul, masks_adrian, masks_sacha, masks_h)
    total_percentage, percentage = GetSliceOverlap(new_mask_paul,new_id_paul,new_mask_adrian,new_id_adrian,new_mask_sacha,new_id_sacha)
    slices_df = pd.DataFrame(percentage, columns = ["Slice IOU"], index = new_id_adrian)
    slices_df["Total IOU"] = total_percentage
    slices_df.index.name = "Patient ID"
    staple_array, staple_ids, staple_slices = StapleGun(masks_a= new_mask_adrian, masks_b= new_mask_paul, masks_c= new_mask_h, ids_a= new_id_adrian, ids_b = new_id_paul, ids_c= new_id_h)

 

    dice_array_adrian = DiceCaster(staple_array, staple_ids, staple_slices, new_mask_adrian, new_id_adrian)
    adrian_total_avg, adrian_patient_avg = AverageDice(dice_array_adrian)
    adrian_df = pd.DataFrame(dice_array_adrian, columns = ["Dice 1", "Dice 2", "Dice 3"  ] , index = staple_ids)
    #adrian_df.insert(0, "Patient_ID", new_id_adrian, True)
    adrian_df["Average Dice"] = adrian_patient_avg
    adrian_df.index.name = "Patient ID"


    dice_array_paul = DiceCaster(staple_array, staple_ids, staple_slices, new_mask_paul, new_id_paul)
    paul_total_avg, paul_patient_avg = AverageDice(dice_array_paul)

    paul_df = pd.DataFrame(dice_array_paul, columns = ["Dice 1", "Dice 2", "Dice 3"], index = staple_ids)
    #paul_df.insert(0, "Patient_ID", new_id_paul, True)
    paul_df["Average Dice"] = paul_patient_avg
    paul_df.index.name = "Patient ID"

    dice_array_sacha = DiceCaster(staple_array, staple_ids, staple_slices, new_mask_sacha, new_id_sacha)
    sacha_total_avg, sacha_patient_avg = AverageDice(dice_array_sacha)

    sacha_df = pd.DataFrame(dice_array_sacha, columns = ["Dice 1", "Dice 2", "Dice 3", ], index = staple_ids)
    #sacha_df.insert(0, "Patient_ID", new_id_sacha, True)
    sacha_df["Average Dice"] = sacha_patient_avg
    sacha_df.index.name = "Patient ID"

    # dice analysis 
    cr_path = "/Users/olivia/Documents/PhD/MISTIE/mask_data/CRdata.xlsx"
    cr_slices_adrian = NewRemoveCoronaRadiataSlices(new_mask_adrian, new_id_adrian, cr_path, sheetname= 'adrian')
    cr_slices_paul = NewRemoveCoronaRadiataSlices(new_mask_paul, new_id_paul, cr_path, sheetname= 'paul')
    cr_slices_sacha = NewRemoveCoronaRadiataSlices(new_mask_sacha, new_id_sacha, cr_path, sheetname= 'sacha')
    print("adrian paul")
    dice_array_AP = DiceCaster3(new_mask_adrian, new_id_adrian, cr_slices_adrian, new_mask_paul, new_id_paul, cr_slices_paul)
    print("adrian sacha")
    dice_array_AS = DiceCaster3(new_mask_adrian, new_id_adrian, cr_slices_adrian, new_mask_sacha, new_id_sacha, cr_slices_sacha)
    print("paul sacha")
    dice_array_PS = DiceCaster3(new_mask_paul, new_id_paul, cr_slices_paul,new_mask_sacha, new_id_sacha, cr_slices_sacha)

    AP_df = pd.DataFrame(dice_array_AP, columns = ["Dice 1", "Dice 2", "Dice 3", "Dice 4", "Dice 5"], index = new_id_adrian)
    AS_df = pd.DataFrame(dice_array_AS, columns = ["Dice 1", "Dice 2", "Dice 3"], index = new_id_adrian)
    PS_df = pd.DataFrame(dice_array_PS, columns = ["Dice 1", "Dice 2", "Dice 3", "Dice 4"], index = new_id_paul)
    AP_total_avg, AP_patient_avg = AverageDice(dice_array_AP)
    AS_total_avg, AS_patient_avg = AverageDice(dice_array_AS)
    PS_total_avg, PS_patient_avg = AverageDice(dice_array_PS)
    AP_df["Average Dice"] = AP_patient_avg
    AS_df["Average Dice"] = AS_patient_avg
    PS_df["Average Dice"] = PS_patient_avg

    total_df = pd.DataFrame([adrian_total_avg, paul_total_avg, sacha_total_avg, AP_total_avg, AS_total_avg, PS_total_avg], index =["Adrian STAPLE average Dice","Paul STAPLE average Dice", "Sacha STAPLE average Dice", "Adrian Paul average Dice", "Adrian Sacha average Dice", "Paul Sacha average Dice"] )
   
    print("avg adrian: ", adrian_total_avg, "avg paul: ", paul_total_avg, "avg sacha: ", sacha_total_avg, "avg AP: ", AP_total_avg, "avg AS: ", AS_total_avg, "avg PS: ", PS_total_avg)

    with pd.ExcelWriter('/Users/olivia/Documents/PhD/MISTIE/mask_data/output.xlsx') as writer:  
        adrian_df.to_excel(writer, sheet_name='Adrian')
        paul_df.to_excel(writer, sheet_name='Paul')
        sacha_df.to_excel(writer, sheet_name='Sacha')
        total_df.to_excel(writer, sheet_name='Total Dice')
        slices_df.to_excel(writer,sheet_name ='Slices' )
        AP_df.to_excel(writer, sheet_name='Adrian vs Paul')
        AS_df.to_excel(writer, sheet_name='Adrian vs Sacha')
        PS_df.to_excel(writer, sheet_name='Paul vs Sacha')





if __name__ == '__main__':
 main()