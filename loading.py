import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage
from pathlib import Path

def GetFiles(filedir):
    entries = Path(filedir)
    fname, id, pathlist = [],[],[]
    idTick = 0
 
    for entry in entries.iterdir():
     idTick += 1
     fname.append(entry.name)
     id.append(idTick)
     pathlist.append(str(filedir) + "\\" + str(entry.name))

    return fname, id, pathlist

def DicomToArray(path):
  x = sitk.ReadImage(path, imageIO="NiftiImageIO")
  y = sitk.GetArrayFromImage(x).astype(float)
  return y

def GetSliceNumber(mask):
  slice_number = 0
  max_range = mask.shape[0]
  for x in range(0,max_range):
    slice = mask[x,:,:]
    val = np.sum(slice)
    if val != 0:
      slice_number = x
  return slice_number

def main():

    cts, ids, pathlist_ct  = GetFiles("C:\\Users\\Olivia\\Documents\\PhD\\Imaging\\Segmentations\\images")#

    print("path: ", cts, "ids: ", ids, "pathlist :", pathlist_ct)
    x = DicomToArray(pathlist_ct[1])
    

    masks, ids, pathlist_mask = GetFiles("C:\\Users\\Olivia\\Documents\\PhD\\Imaging\\Segmentations\\masks")
    y = DicomToArray(pathlist_mask[1])
    slice_number = GetSliceNumber(y)
    print("slice_no: ", slice_number)

    #im = plt.imshow(x[slice_number], cmap = "gray")
    #plt.show()
    im = plt.imshow(y[slice_number], cmap= "gray")
    plt.show()
    
    return 

if __name__ == '__main__':
    main()