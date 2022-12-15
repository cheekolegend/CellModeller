"""
This is just a standard simulation without adhesion for comparison to
the adhesion simulation.
"""

import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.GUI import Renderers
import numpy
import math

gamma = 200 
dt = 0.01
sim_time = 8.0

def setup(sim):
    # Set biophysics module
    biophys = CLBacterium(sim, jitter_z=False, max_cells=5000, gamma=gamma, cgs_tol=1e-4)

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

def init(cell):
    # Specify mean and distribution of initial cell size
    cell.targetVol = 5
    # Specify growth rate of cells
    cell.growthRate = 1.0
    cell.color = (0, 1, 0) 

def update(cells):
    #Iterate through each cell and flag cells that reach target size for division
    for (id, cell) in cells.items():
        if cell.volume > cell.targetVol:
            cell.divideFlag = True

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    d1.targetVol = 5
    d2.targetVol = 5
