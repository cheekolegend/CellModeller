import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.GUI import Renderers
import numpy
import math
from scipy.stats import norm, lognorm

import conjugation_modules

# Cell initialization
donor_frac = 0.25
n_start = 60

# Time settings
dt = 1/60 #h
sim_time = 24.0 #h

# Morphology. Numbers taken from MG1655 sept 2022 agarose pad experiment
# Radius (lognorm)
rad_mean = 0.27
rad_std = 0.29
rad_loc = 0.17
# Division length (norm)
div_mean = 5.45
div_std = 1.86
# Growth rate (norm)
gr_mean = 1.60
gr_std = 0.37

# Fluorescence
rfp_intensities = {'donor':1, 'recip':0, 'trans':1} #RGB cell colours
gfp_intensities = {'donor':0, 'recip':1, 'trans':1} #RGB cell colours
colors = {'donor': (1,0,0), 'recip': (0,1,0), 'trans': (1,1,0)}

# Geometry
len_x = 80.0
len_y = 80.0

# Conjugation parameters
conj_freq = 1 # conjugation per contact per hour
conj_prob = conj_freq*dt # probability on a per-time-step basis

def setup(sim):
    sim.dt = dt
    
    # Set biophysics module
    biophys = CLBacterium(sim, jitter_z=False, max_planes=3, gamma=200, max_cells=30000, cgs_tol=1E-5, compNeighbours=True, max_contacts=36)

    # Set up regulation module
    regul = ModuleRegulator(sim, sim.moduleName)	
    # Only biophys and regulation
    sim.init(biophys, regul, None, None)
    
    # Define walls
    planeWeight = 1.0
    biophys.addPlane((0,0,0), (1,0,0), planeWeight) 		    # left side
    biophys.addPlane((len_x,0,0), (-1,0,0), planeWeight) 		# right side
    biophys.addPlane((0,0,0), (0,1,0), planeWeight) 			# base
 
    # Specify the initial cell and its location in the simulation
    add_donors_recips(sim, n_start, donor_frac, len_x, len_y)
    
    # Add some objects to draw the models
    therenderer = Renderers.GLBacteriumRenderer(sim)
    sim.addRenderer(therenderer)
    
    # Specify how often data is saved
    sim.pickleSteps = 5

def init(cell):
    cell.targetVol = min(8, norm.rvs(loc=div_mean, scale=div_std))
    cell.growthRate = min(2.1, norm.rvs(loc=gr_mean, scale=gr_std))
    cell.rfp_intensity = rfp_intensities[cell.cellType]
    cell.gfp_intensity = gfp_intensities[cell.cellType]
    cell.color = colors[cell.cellType]
    cell.transconjugant_flag = 0

#TODO: implement delay in conjugation time for transconjugants?
#TODO: conjugation frequency as a function of time receieved?
def update(sim, cells):
    for (id, cell) in cells.items():
        # Cell removal and division
        if cell.pos[1] > (len_y):
            cell.deathFlag = True
        if cell.volume > cell.targetVol and cell.deathFlag == False:
            cell.divideFlag = True
        
        # Conjugation
        conjugation_modules.conjugation_basic(sim, cells, cell, conj_freq, dt, rfp_intensities, colors)

def divide(parent, d1, d2):
    d1.targetVol = min(8, norm.rvs(loc=div_mean, scale=div_std))
    d2.targetVol = min(8, norm.rvs(loc=div_mean, scale=div_std))            
    d1.growthRate = min(2.1, norm.rvs(loc=gr_mean, scale=gr_std))
    d2.growthRate = min(2.1, norm.rvs(loc=gr_mean, scale=gr_std))
    
def kill(cell):
    """
    This function specifies how to change a cell's state when it dies.
    """ 
    # define what happens to a cell's state when it dies
    cell.growthRate = 0.0       # dead cells can't grow any more
    cell.divideFlag = False     # dead cells can't divide
    
def add_donors_recips(sim, n_start, donor_frac, len_x, len_y):
    """
    Randomly seed donor and recip cells in a square domain
    @param sim          Simulator object
    @param n_start      Number of cells to start with (int)
    @param donor_frac   Fraction of initial donors (float)
    @param len_x        Length of domain (um)
    @param len_y        Height of domain (um)
    """
    # Create cells
    recips = round(n_start*(1-donor_frac))
    donors = round(n_start*donor_frac)
    
    for i in range(recips):
        sim.addCell(cellType = 'recip', \
                    pos = (random.uniform(0, len_x), random.uniform(0, len_y), 0), \
                    dir=(random.uniform(0, 1), random.uniform(0, 1), 0), \
                    rad = lognorm.rvs(
                                      rad_std,  
                                      loc = rad_loc, 
                                      scale = rad_mean
                                     )
                   )
                   
    for i in range(donors):
        sim.addCell(cellType = 'donor', \
                    pos = (random.uniform(0, len_x), random.uniform(0, len_y), 0), \
                    dir=(random.uniform(0, 1), random.uniform(0, 1), 0), \
                    rad = lognorm.rvs(
                                      rad_std,  
                                      loc = rad_loc, 
                                      scale = rad_mean
                                     )
                    )

