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
from utils import SelectCT, UnpackNpz
import shutil

def main():
    adrian_path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/adrian_masks.npz'
    paul_path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/paul_masks.npz'
    sacha_path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/sacha_masks.npz'
    h_path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/h_masks.npz'
    masks_adrian, ids_adrian = UnpackNpz(adrian_path)
    masks_paul, ids_paul = UnpackNpz(paul_path)
    masks_sacha, ids_sacha = UnpackNpz(sacha_path)
    masks_h, ids_h = UnpackNpz(h_path)
    path_scan = "/Users/olivia/Documents/PhD/MISTIE/scans_original" 

    patients = Path(path_scan)
    file_list = []
    for patient in patients.iterdir():
        print("dir: ", patient)
        if patient.name[0:4] in ids_adrian and patient.name[0:4] in ids_paul and patient.name[0:4] in ids_sacha and patient.name[0:4] in ids_h :
            placeholder = ids_adrian.tolist().index(patient.name[0:4])
            placeholder2 = ids_paul.tolist().index(patient.name[0:4])
            placeholder3 = ids_sacha.tolist().index(patient.name[0:4])
            placeholder4 = ids_h.tolist().index(patient.name[0:4])
            keep_files = SelectCT(patient,masks_adrian[placeholder], masks_paul[placeholder2], masks_sacha[placeholder3],masks_h[placeholder4] )
            if len(keep_files) != 0:
                # Create new patient directory 
                directory = str(patient.name[0:4])
                # Parent Directories 
                parent_dir = "/Users/olivia/Documents/PhD/MISTIE/matched_scan"
                path = os.path.join(parent_dir, directory)    
                # Create the directory  
                os.makedirs(path) 
                print("Directory '% s' created" % directory)

                # copying keepfile
                for file in keep_files:

                    fromDirectory = file
                    toDirectory = "/Users/olivia/Documents/PhD/MISTIE/matched_scan/" + str(patient.name[0:4]) + "/Diagnostic CT"

                    shutil.copytree(fromDirectory, toDirectory)


            file_list.append(keep_files)
    print(file_list)

if __name__ == '__main__':
 main()