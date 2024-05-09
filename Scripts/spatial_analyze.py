import sys
import os
import math
import numpy as np
sys.path.append('.')
import pickle
import CellModeller
import matplotlib.pyplot as plt

def velocity_vectors(cells):
    """
    1. Get cell centers for start of arrows
    2. Get x and y velocity component for each cell
    3. Get color for each cell
    3. Convert cell centers to a meshgrid
    4. Velocity components need to line up with meshgrid
    5. Plot using quiver: https://pythonforundergradengineers.com/quiver-plot-with-matplotlib-and-jupyter-notebooks.html
    """
    x = np.zeros(len(cells))
    y = np.zeros(len(cells))
    v_x = np.zeros(len(cells))
    v_y = np.zeros(len(cells))
    color = [0]*len(cells)
    for i, cell in enumerate(cells.values()):
        x[i] = cell.pos[0]
        y[i] = cell.pos[1]
        v_x[i] = cell.vel[0]
        v_y[i] = cell.vel[1]
        color[i] = tuple(cell.color)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.quiver(x, y, v_x, v_y, color=color)
    
    return fig, ax
    

pname= sys.argv[1]
print(('opening '+ pname))
fout=open('spatial.txt', "w")
fout.write("r spec_level \n")

bin_num=20
rad_max=95

narray=np.zeros(bin_num)
specArray=np.zeros(bin_num)

data = pickle.load(open(pname,'r'))
cs = data['cellStates']
it = iter(cs)
n = len(cs)

rArray = np.multiply(list(range(0,bin_num)),rad_max/bin_num)

for it in cs:
    r = np.sqrt(cs[it].pos[0]*cs[it].pos[0]+cs[it].pos[1]*cs[it].pos[1])
    for x in range(0,bin_num):
        if((r>=x*rad_max/bin_num)and(r<(x+1)*rad_max/bin_num)):
            specArray[x]=specArray[x]*narray[x]+cs[it].species[0]
            narray[x]+=1
            specArray[x]=specArray[x]/narray[x]

for x in range(0,bin_num):
    fout.write(str((x+1)*rad_max/bin_num/2)+" "+str(narray[x]))
    if(narray[x]==0): fout.write(" 0.0\n")
    else: fout.write(" "+str(specArray[x])+"\n")

fout.close()

plt.plot(rArray,specArray)
plt.show()
