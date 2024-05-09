"""
    ConvertPicklesToVTPs.py
    =======================
    
    Exports CM4 .pickle configuration files as .vtp meshes for visualisation in Paraview.
    Directories to be converted are specified as arguments.
    Original .pickle files are retained.
    Any .vtp files in the target directory(s) sharing names with pickle files in the same location will be overwritten.

    Created 16. 03. 15 
    W. P. J. Smith
    
    Updated Jan 2022 for use with python3
    A. Yip
"""

import os
import sys
import pickle
from CellModeller.VTPWriter import *

# mesh resolution - higher values take longer to write but look better
num_points = 8

# specify target directory(s) as argument
if len(sys.argv) < 2:
    print('Error: Need to specify at least 1 target directory.')
    print('Usage is $ python ConvertPicklesToVTUs.py <dir1> <dir2> ... <dirN>.')
    exit()
Dirs = sys.argv[1:]

# go through all the directories specified
for targetDir in Dirs:
    print('Converting pickles in '+targetDir+':')

    # Go through all the pickle files in this dir and convert them
    for file in os.listdir(targetDir):
        if file.endswith(".pickle"):
            fileName, fileExtension = os.path.splitext(file)
            fileOut = fileName+'.vtp'
            print ('Converting '+file+' to '+fileOut+'...')
            data = pickle.load(open(targetDir+file,'rb'))
            cs = data['cellStates']     
            print ('Number of cells:', len(cs))
            myWriter = CapsuleWriter(num_points, targetDir+fileOut)
            myWriter.writeConfiguration(cs)

print('All done.')

