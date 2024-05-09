'''
This is a demo simulation for growth in a microfluidic device with two open boundaries. This tutorial shows how to remove cells from the simulation.

The geometry is as follows:

|--------------open-----------------|
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|___________________________________|

Domain size is 20 X 20 um. 

For more information about how this functionality was implemented, see Simulatory.py, ModuleRegulator.py, CLBacterium.py, and CLBacterium.cl

Most of the code for cell removal is from William P.J. Smith:
https://github.com/WilliamPJSmith/CM4-A
'''

import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.GUI import Renderers
import numpy
import math

cell_cols = {0:[0,1.0,0], 1:[1.0,0,0], 2:[0,0,1.0]} #RGB cell colours
cell_lens = {0:3.5, 1:3.5, 2:3.5} #target cell lengths
cell_growr = {0:1, 1:1, 2:1} #growth rates

#Geometry
min_x = 0.0
max_x = 20.0
min_y = -0.1
max_y = 20.0

def setup(sim):
    # Set biophysics, signalling, and regulation models
    biophys = CLBacterium(sim, jitter_z=False, max_planes=2, max_cells=5000)

    # use this file for reg too
    regul = ModuleRegulator(sim, sim.moduleName)
    # Only biophys and regulation
    sim.init(biophys, regul, None, None)
 
    planeWeight = 1.0
    biophys.addPlane((min_x,0,0), (1,0,0), planeWeight)             # left side
    biophys.addPlane((max_x,0,0), (-1,0,0), planeWeight)         # right side
    #biophys.addPlane((0,0,0), (0,0,+1), planeWeight)             # base
 
    # Specify the initial cell and its location in the simulation
    sim.addCell(cellType=0, pos=(max_x/2,max_y/2,0), dir=(0,1,0))
    sim.addCell(cellType=1, pos=(6,2,0), dir=(0,1,0))
    sim.addCell(cellType=2, pos=(15,2,0), dir=(0,1,0))


    # Add some objects to draw the models
    therenderer = Renderers.GLBacteriumRenderer(sim)
    sim.addRenderer(therenderer)
    sim.pickleSteps = 20

def init(cell):
    # Specify mean and distribution of initial cell size
    cell.targetVol = cell_lens[cell.cellType] + random.uniform(0.0,0.5)
    # Specify growth rate of cells
    cell.growthRate = cell_growr[cell.cellType]
    cell.color = cell_cols[cell.cellType]

def update(cells):
    #Iterate through each cell and flag cells that reach target size for division
    for (id, cell) in cells.items():
        if cell.pos[1] > max_y or cell.pos[1] < min_y:
            cell.deathFlag = True
        if cell.volume > cell.targetVol and cell.deathFlag == False:
            cell.divideFlag = True

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    d1.targetVol = cell_lens[parent.cellType] + random.uniform(0.0,0.5)
    d2.targetVol = cell_lens[parent.cellType] + random.uniform(0.0,0.5)
    
def kill(cell):
    """
    This function specifies how to change a cell's state when it dies.
    """ 
    # define what happens to a cell's state when it dies
    cell.growthRate = 0.0       # dead cells can't grow any more
    cell.divideFlag = False        # dead cells can't divide
