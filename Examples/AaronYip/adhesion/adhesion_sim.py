"""
This case simulates a microcolony growing from a single cell.
The cell is adhesive to other cells and walls. The strength of adhesion is set by "adhesion".

The model was adapted from:
Kan, A., Del Valle, I., Rudge, T., Federici, F., & Haseloff, J. (2018). 
Intercellular adhesion promotes clonal mixing in growing bacterial populations. 
Journal of The Royal Society Interface, 15(146), 20180406.

The model was updated to align with the 2020 version of CellModeller.
Adhesion to walls was also added and verified (see wall_adhesion_test.py), but has yet to be
validated experimentally.
"""

import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacteriumAdhesion import CLBacteriumAdhesion
from CellModeller.GUI import Renderers
import numpy
import math

gamma = 200 
adhesion = 10000
dt = 0.01
sim_time = 8.0

def setup(sim):
    # Set biophysics module
    biophys = CLBacteriumAdhesion(sim, jitter_z=False, max_cells=5000, gamma=gamma, cgs_tol=1e-4)

    # Set up regulation module
    regul = ModuleRegulator(sim, sim.moduleName)	
    # Only biophys and regulation
    sim.init(biophys, regul, None, None)
 
    # Specify the initial cell and its location in the simulation
    sim.addCell(cellType=0, pos=(0,0,0), dir=(1,0,0))

    # Add some objects to draw the models
    therenderer = Renderers.GLBacteriumRenderer(sim)
    sim.addRenderer(therenderer)
    
    # Specify how often data is saved
    sim.pickleSteps = 10
    sim.dt = dt #h
    
def adhLogicCL():
    return """return min(adh_str1, adh_str2);"""

def init(cell):
    # Specify mean and distribution of initial cell size
    cell.targetVol = 5
    # Specify growth rate of cells
    cell.growthRate = 1.0
    cell.color = (0, 1, 0) 
    cell.cellAdh = adhesion

def update(cells):
    #Iterate through each cell and flag cells that reach target size for division
    for (id, cell) in cells.items():
        if cell.volume > cell.targetVol:
            cell.divideFlag = True

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    d1.targetVol = 5
    d2.targetVol = 5
