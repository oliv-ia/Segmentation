# 11/11/2021
# Olivia Murray
# File to sort database into correct format for osirix plugin 
import os
from pathlib import Path
import numpy as np
import shutil
def GetFiles(filedir):
    entries = Path(filedir)
    fname, id, pathlist = [],[],[]
    idTick = 0
 
    for entry in entries.iterdir():
     idTick += 1
     fname.append(entry.name)
     id.append(idTick)
     pathlist.append(str(filedir) + "/" + str(entry.name))

    return np.array(fname), np.array(id), np.array(pathlist)


"""
fname, id, pathlist = GetFiles("/Users/olivia/Documents/PhD/MISTIE/scans_adrian")
fname_roi, id_roi, pathlist_roi = GetFiles("/Users/olivia/Documents/PhD/MISTIE/SachaROI")
#print(fname_roi)
y = [z[:4] for z in fname]
tick, tickf = 0,0
badFile = []
goodFile = []
roi = [z[0:4] for z in fname_roi]
        
for i in range(len(roi)):
    if roi[i] == "PRIM" or roi[i] == ".DS_":
        roi[i]=0
print(roi)
for file in fname:
    if file[0:4] == "PRIM" or file[0:4] == ".DS_" :
        file =0
    else:
        for name in roi:
        
            if file[0:4] == name:
             print("yep", file[0:4])
             tick += 1
        if np.array(file[0:4]).astype(np.int) != np.array(roi).astype(np.int).any():
            tickf += 1
        
            badFile.append(file[0:4])

for name in roi:
    for file in fname:
        if name == file[0:4]:
            goodFile.append(file)
            


print(tick)       
print(tickf)
print(len(badFile), badFile)
print(len(roi), len(fname)
)
print("goodFile: ", len(goodFile), goodFile)



from shutil import copytree






for name in goodFile:
    # Leaf directory 
    directory = str(name[0:4])
        
    # Parent Directories 
    parent_dir = "/Users/olivia/Documents/PhD/MISTIE/sachaexp"
        
    # Path 
    path = os.path.join(parent_dir, directory) 
        
    # Create the directory 
    # 'Nikhil' 
    os.makedirs(path) 
    print("Directory '% s' created" % directory)

for file in goodFile:
   
    newPath = shutil.copy("/Users/olivia/Documents/PhD/MISTIE/SachaROI/" + str(file[0:4]) + "_SC.rois_series", '/Users/olivia/Documents/PhD/MISTIE/sachaexp/' + str(file[0:4]))


"""

"""

# copy subdirectory example
for file in goodFile:
 
        fromDirectory = "/Users/olivia/Documents/PhD/MISTIE/scans_adrian/" +str(file) 
        toDirectory = "/Users/olivia/Documents/PhD/MISTIE/sachaexp/" + str(file[0:4]) + "/scan"

        shutil.copytree(fromDirectory, toDirectory)

"""