import os
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

'''
# Cell initialization
donor_frac = 0.25
n_start = 60
'''
# Time settings
dt = 0.01666666666666666666666666666667 # 1/60 h
sim_time = 3.0 #h

# Calibration parameters
conj_freq = 1 # conjugations per hour
delay = 1/2 # time before a new transconjugant can conjugate plasmid (h)
m_max = 2 # m_max times more likely for a transconjugant to conjugate after receiving plasmid
t_half = 1/2 # time before conjugation boost m_max is halved (h)
death_rate_donor = 0.1 # deaths per donor per hour
death_rate_trans = 0.1 # deaths per donor per hour

# Single-cell properties
div_len_stats = {"donor": {"scale": 2.406459745720652, "shape": 0.42601771988713844, "loc": -0.04172547958959558}, "recip": {"scale": 3.815676440941102, "shape": 0.5041533396839497, "loc": 0.24137414729843287}, "trans": {"scale": 4.653243452922401, "shape": 0.4701909049349248, "loc": -0.22373214746568143}}
growthRate_stats = {"donor": {"mu": 0.49573136108114246, "std": 0.37547052570799827}, "recip": {"mu": 0.5634495694201387, "std": 0.40053660093269366}, "trans": {"mu": 0.580504425262798, "std": 0.41313222050938087}}
min_growthRate = 0
min_div_length = 1.5

# Fluorescence
rfp_intensities = {'donor':1, 'recip':0, 'trans':1} #RGB cell colours
gfp_intensities = {'donor':0, 'recip':1, 'trans':1} #RGB cell colours
colors = {'donor': (255/255,0,0), 'recip': (0,200/255,200/255), 'trans': (200/255,255/255,255/255)}

# Geometry
len_x = 80.0
len_y = 80.0

# Read in initial physiological parameters
global cell_property_dict

def setup(sim):
    global cell_property_dict
    
    sim.dt = dt
    sim.hgt_events = {'stepNum': [], 'donor_id': [], 'recip_id': []}
    
    # Set biophysics module
    #biophys = CLBacterium_reg_param(sim, jitter_z=False, max_planes=3, gamma=200, reg_param=1, max_cells=30000, cgs_tol=1E-5, compNeighbours=True, max_contacts=42)
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
    scene_file = "scene_io/" + os.path.basename(sim.outputDirName) + "/scene.txt"
    #scene_file = "scene.txt" # USE THIS WHEN TESTING
    with open(scene_file, 'r') as file:
        # Read the entire content of the file into a string
        scene = file.read()
        scene = scene.rstrip('\n')
    csv_file = f'initialization/{scene}Trackrefiner.tracking_cells_Linear Regression_analysis.csv'
    cell_property_dict = import_initial_properties(csv_file)
    import_cells_from_csv(sim, csv_file)
    #add_donors_recips(sim, n_start, donor_frac, len_x, len_y)
        
    # Add some objects to draw the models
    therenderer = Renderers.GLBacteriumRenderer(sim)
    sim.addRenderer(therenderer)
    
    # Specify how often data is saved
    sim.pickleSteps = 5
    

def init(cell):
    global cell_property_dict
    cell.targetVol = cell_property_dict[cell.id]['targetVol']
    cell.growthRate = cell_property_dict[cell.id]['growthRate']
    cell.rfp_intensity = rfp_intensities[cell.cellType]
    cell.gfp_intensity = gfp_intensities[cell.cellType]
    cell.color = colors[cell.cellType]
    cell.transconjugant_flag = 0
    cell.time = 0
    cell.conj_time = -10


def update(sim, cells):

    for (id, cell) in cells.items():
        # Cell removal and division
        if cell.pos[1] > (len_y):
            cell.deathFlag = True
        if cell.volume > cell.targetVol and cell.deathFlag == False:
            cell.divideFlag = True
            
        # Cell randomly stops growing
        if cell.cellType == 'donor':
            if random.random() < death_rate_donor*dt:
                cell.deathFlag = True
        elif cell.cellType == 'trans':
            if random.random() < death_rate_trans*dt:
                cell.deathFlag = True
        
        # Conjugation
        conjugation_modules.conjugation_full(sim, cells, cell, conj_freq, dt, rfp_intensities, colors, delay, m_max, t_half)
          

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    d1.growthRate = max(min_growthRate, np.random.normal(growthRate_stats[d1.cellType]['mu'], growthRate_stats[d1.cellType]['std']))
    d2.growthRate = max(min_growthRate, np.random.normal(growthRate_stats[d2.cellType]['mu'], growthRate_stats[d2.cellType]['std']))
    d1.targetVol = max(min_div_length, lognorm.rvs(div_len_stats[d1.cellType]['shape'], loc=div_len_stats[d1.cellType]['loc'], scale=div_len_stats[d1.cellType]['scale']))
    d2.targetVol = max(min_div_length, lognorm.rvs(div_len_stats[d2.cellType]['shape'], loc=div_len_stats[d2.cellType]['loc'], scale=div_len_stats[d2.cellType]['scale']))
    
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
        r = 0.6 # row['radius']
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
                        
def setparams(param_dict):
    global conj_freq, delay, m_max, t_half, death_rate_donor, death_rate_trans
    conj_freq = param_dict['conj_freq']
    delay = param_dict['delay']
    m_max = param_dict['m_max']
    t_half = param_dict['t_half']
    death_rate_donor = param_dict['death_rate_donor']
    death_rate_trans = param_dict['death_rate_trans']
    
    
    
    
