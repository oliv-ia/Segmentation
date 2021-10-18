import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage
from pathlib import Path
from mpl_toolkits.axes_grid1 import ImageGrid

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
    # extracting CT scans, creating CT array
    cts_fname, ids, pathlist_ct  = GetFiles("C:\\Users\\Olivia\\Documents\\PhD\\Imaging\\Segmentations\\images")#
    cts = []
    for i in range(0, len(pathlist_ct)):
      x = DicomToArray(pathlist_ct[i])
      cts.append(x)
    
    # Extracting masks, creating mask array
    masks_fname, ids, pathlist_mask = GetFiles("C:\\Users\\Olivia\\Documents\\PhD\\Imaging\\Segmentations\\masks")
    masks, slice_numbers  = [],[]
    for i in range(0, len(pathlist_mask) ):
      y = DicomToArray(pathlist_mask[i])
      masks.append(y)
      slice_number = GetSliceNumber(y)
      slice_numbers.append(slice_number)
    
    # dispalying images
    fig=plt.figure(figsize=(10, 120))
    ax = []
    columns = 4
    rows = int(len(cts)/columns + 0.5)
 
    print("len cts: ",len(cts))
    for i in range(0, len(cts)):
      ax.append(fig.add_subplot(rows,columns, i+1))
      masks[i][masks[i]==0] = np.nan
      plt.imshow(cts[i][slice_numbers[i],:,:], cmap="gray")
      plt.imshow(masks[i][slice_numbers[i],:,:], cmap = "autumn", alpha = 0.7)

      ax[-1].set_title("Patient:" + str(ids[i]))
      plt.axis("off")
    plt.tight_layout(True)
    plt.show()

    return 

if __name__ == '__main__':
    main()