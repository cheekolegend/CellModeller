import ast
import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.GUI import Renderers
import numpy as np
import pandas as pd
import math
from scipy.stats import norm, lognorm

import conjugation_modules

def import_initial_properties(csv_file):
    df = pd.read_csv(csv_file)
    df.fillna(0, inplace=True) # In case there is a blank recorded
    property_dict = {}
    
    for index, row in df.iterrows():
    
        targetVol = row['targetVol'] - 2*row['radius']
        if targetVol < 2*row['radius']:
            targetVol = 2*row['radius']
            
        growthRate = row['growthRate']
        if growthRate <= 0:
            growthRate = 0
            
        property_dict[row['id']] = {'targetVol': targetVol, 'growthRate': growthRate}
        
    return property_dict

csv_file = 'initialization/1Trackrefiner.tracking_cells-Linear Regression-analysis.csv'

'''
# Cell initialization
donor_frac = 0.25
n_start = 60
'''

# Time settings
dt = 0.01666666666666666666666666666667 #h
sim_time = 24.0 #h

div_len_stats = {"donor": {"scale": 1.9592209857789205, "shape": 0.47111176511493447, "loc": 0.2470319137391556}, "recip": {"scale": 4.502285805518995, "shape": 0.49847889772092263, "loc": -0.2731092194298479}, "trans": {"scale": 2.7446828551727425, "shape": 0.7393497612917819, "loc": 0.3965872160355173}}
growthRate_stats = {"donor": {"mu": 0.48407932706487505, "std": 0.333882168176535}, "recip": {"mu": 0.7196055272812362, "std": 0.4195266198042525, "n": 119661}, "trans": {"mu": 0.6482650142827283, "std": 0.449057685412902}}

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

# Read in initial physiological parameters
cell_property_dict = import_initial_properties(csv_file)

def setup(sim):
    sim.dt = dt
    sim.hgt_events = {'stepNum': [], 'donor_id': [], 'recip_id': []}
    
    # Set biophysics module
    biophys = CLBacterium(sim, jitter_z=False, max_planes=3, gamma=100, max_cells=30000, cgs_tol=1E-5, compNeighbours=True, max_contacts=36)

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
    #add_donors_recips(sim, n_start, donor_frac, len_x, len_y)
    import_cells_from_csv(sim, csv_file)
    
    # Add some objects to draw the models
    therenderer = Renderers.GLBacteriumRenderer(sim)
    sim.addRenderer(therenderer)
    
    # Specify how often data is saved
    sim.pickleSteps = 5
    

def init(cell):
    cell.targetVol = cell_property_dict[cell.id]['targetVol']
    cell.growthRate = cell_property_dict[cell.id]['growthRate']
    cell.rfp_intensity = rfp_intensities[cell.cellType]
    cell.gfp_intensity = gfp_intensities[cell.cellType]
    cell.color = colors[cell.cellType]
    cell.transconjugant_flag = 0


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
    # Specify target cell size that triggers cell division
    d1.growthRate = np.random.normal(growthRate_stats[d1.cellType]['mu'], growthRate_stats[d1.cellType]['std'])
    d2.growthRate = np.random.normal(growthRate_stats[d2.cellType]['mu'], growthRate_stats[d2.cellType]['std'])
    d1.targetVol = lognorm.rvs(div_len_stats[d1.cellType]['shape'], loc=div_len_stats[d1.cellType]['loc'], scale=div_len_stats[d1.cellType]['scale'])
    d2.targetVol = lognorm.rvs(div_len_stats[d2.cellType]['shape'], loc=div_len_stats[d2.cellType]['loc'], scale=div_len_stats[d2.cellType]['scale'])
    
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

def import_cells_from_csv(sim, csv_file):
    df = pd.read_csv(csv_file, converters={'pos': ast.literal_eval, 'dir': ast.literal_eval})
    for index, row in df.iterrows():
        gfp_intensity = row['gfp_intensity']
        rfp_intensity = row['rfp_intensity']
        pos = row['pos']
        r = row['radius']
        l = row['length'] - 2*r
        angle = row['dir']
        if gfp_intensity > rfp_intensity:
            cellType = 'recip'
        else:
            cellType = 'donor'
            
        if r or l != 0:
            sim.addCell(pos=(pos[0],pos[1],0), 
                        dir=(angle[0],angle[1],0), 
                        length=l, 
                        rad=r, 
                        cellType=cellType)
                        

