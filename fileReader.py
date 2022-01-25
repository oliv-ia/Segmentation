#15/11/2021
# Olivia Murray
# Script to convert extracted masks to dataset array
import os
from pathlib import Path
import numpy as np
import shutil
import SimpleITK as sitk
import scipy
from scipy.ndimage.measurements import center_of_mass
def GetFiles(filedir):
    entries = Path(filedir)
    fname, id, pathlist = [],[],[]
    idTick = 0
 
    for entry in entries.iterdir():
     idTick += 1
     skip = np.array([2006, 2031, 2170])
     if entry.name[0:4] != "2006" and entry.name[0:4] != ".DS_" and entry.name[0:4] != "2031" and entry.name[0:4] != "2170" and entry.name[0:4] != "2035":
        fname.append(entry.name)
        id.append(idTick)
        pathlist.append(str(filedir) + "/" + str(entry.name))

    return np.array(fname), np.array(id), np.array(pathlist)
def ReadMask(filedir):
    seg = sitk.ReadImage(filedir, imageIO= "NiftiImageIO")  #="NiftiImageIO")
    seg_arr = sitk.GetArrayFromImage(seg)
    seg_arr = seg_arr.astype(np.int64)
    return seg_arr
def GetSlices(seg_arr):
    values= []
    for i in range(0,len(seg_arr, axis = 0)):
        val = np.sum(seg_arr[i,:,:])
        if val !=0:
            values.append(i)
    return values

def SaveData(inputs, ids):
    path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/paul_masks.npz'
    ids = np.array(ids)
    #print("final shape: ", inputs.shape,  ids.shape)
    np.savez(path, masks = inputs,  ids = ids) 
    print("Saved data") 

def Cropping(input):
    coords = center_of_mass(input)
    

adrian_path = "/Users/olivia/Documents/PhD/MISTIE/adrianexp"
scans_path = "/Users/olivia/Documents/PhD/MISTIE/adrianexp"
sacha_path = "/Users/olivia/Documents/PhD/MISTIE/sachaexp"
paul_path = "/Users/olivia/Documents/PhD/MISTIE/paulexp"
fname, ids, pathlist = GetFiles(paul_path)

#print(fname, ids.shape)
#print(pathlist)
mask_pathlist = []
for dir in pathlist:   
    directory = "scan.hdr"
    mask_entry = os.path.join(dir , directory)
    mask_pathlist.append(mask_entry)


mask_array = []
for dir in mask_pathlist:
    mask = ReadMask(dir)
    mask_array.append(mask)
    
    print ("mask shape: ", mask.shape)
    print(" Read for ", dir)

print(len(mask_array[0]), len(mask_array[1]), len(mask_array[2]))

print(fname)
masks = SaveData(mask_array, fname)

"""

ct_pathlist = []
fname_scan, id_scan, pathlist_scan = GetFiles(scans_path)
for dir in pathlist_scan:
    directory = str(fname) + "/Diagnostic CT/"
    ct_entry = os.path.join(dir, directory)

ct_array = []
for dir in pathlist:
    ct = ReadMask(dir)
    ct_array.append(ct)
mask_array = np.array(mask_array)
ct_array = np.array(ct_array)
print(mask_array.shape, mask_array.shape)
"""