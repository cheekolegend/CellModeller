"""
This is a reproduction of the simulation(s) from Fig. 6 in:

Kan, A., Del Valle, I., Rudge, T., Federici, F., & Haseloff, J. (2018). 
Intercellular adhesion promotes clonal mixing in growing bacterial populations. 
Journal of The Royal Society Interface, 15(146), 20180406.
"""

import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacteriumAdhesion import CLBacteriumAdhesion
from CellModeller.GUI import Renderers
import numpy
import math

# Simulation parameters
max_cells = 20
dt = 0.025 #h
sim_time = 20.0

cell_col = [0.0,1.0,0.0]
cell_radius = 0.5
cell_adh = 0.1 # 0.1 gives a fun simulation :) 0-1 gives expected results from Kan's 2018 paper
force = 0.005 # 0.005 gives a fun simulation :), 0.1 gives expected results from Kan's 2018 paper

def setup(sim):
    # Set biophysics, signalling, and regulation models
    biophys = CLBacteriumAdhesion(sim, dt=0.025, max_planes=2, jitter_z=False, cgs_tol=1e-5, compNeighbours=False, max_contacts=32, gamma=100.0, max_cells=max_cells, externalForces=True)

    # use this file for reg too
    regul = ModuleRegulator(sim, sim.moduleName)
    # Only biophys and regulation
    sim.init(biophys, regul, None, None)

    # Left cell
    sim.addCell(pos=(0.0,0,0), dir=(0,1,0), rad=cell_radius, cellAdh=cell_adh, cellForce=(0,force,0,0))
    # Middle cells
    for i in range(1,9):
        sim.addCell(pos=(i*2*cell_radius,0,0), dir=(0,1,0), rad=cell_radius, cellAdh=cell_adh, cellForce=(0,0,0,0))
    # Right cell
    sim.addCell(pos=(9*2*cell_radius,0,0), dir=(0,1,0), rad=cell_radius, cellAdh=cell_adh, cellForce=(0,-force,0,0))

    # Add some objects to draw the models
    therenderer = Renderers.GLBacteriumRenderer(sim)
    sim.addRenderer(therenderer)

    sim.pickleSteps = 10

# Adhesion logic - here the adhesion strength is the weakest cellAdh of the two contacting cells
# For logic on wall adhesion, see build_tangent_matrix function in AdhKernel.cl
def adhLogicCL():
    return """return min(adh_str1, adh_str2);"""
        
def init(cell):
    cell.targetVol = 4
    cell.growthRate = 0
    cell.color = cell_col

def update(cells):
    #Iterate through each cell and flag cells that reach target size for division
    pass

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    pass
