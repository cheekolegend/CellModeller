import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium_reg_param import CLBacterium_reg_param
from CellModeller.GUI import Renderers
import numpy
import math

gamma = 10
reg_param = 0.1 
dt = 0.02
sim_time = 16.0

def setup(sim):
    # Set biophysics module
    biophys = CLBacterium_reg_param(sim, jitter_z=False, max_cells=2000, reg_param=reg_param, gamma=gamma)

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
    sim.pickleSteps = 100
    sim.dt = dt #h

def init(cell):
    # Specify mean and distribution of initial cell size
    cell.targetVol = 3.5
    
    # Specify growth rate of cells
    cell.growthRate = 1.0
    
    # Extra, non relevant stuff
    cell.strainRate_rolling = 0
    cell.color = (0, 1, 0) 

def update(cells):
    #Iterate through each cell and flag cells that reach target size for division
    for (id, cell) in cells.items():
        if cell.volume > cell.targetVol:
            cell.divideFlag = True
        
        if cell.cellAge != 1:    
            cell.color = (0, cell.strainRate_rolling/dt/cell.growthRate, 0)

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    d1.targetVol = parent.targetVol
    d2.targetVol = parent.targetVol
        
def setparams(param_dict):
    global gamma, reg_param
    gamma = param_dict['gamma']
    reg_param = param_dict['reg_param']

