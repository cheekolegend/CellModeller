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

# Cell initialization
donor_frac = 2/3
n_start = 400

# Time settings
dt = 0.01666666666666666666666666666667 # 1/60 h
sim_time = 16.0 #h

# Calibration parameters
conj_freq = 0.5142786401713151 # conjugations per hour
delay = 0.6288688280654475 # time before a new transconjugant can conjugate plasmid (h)
m_max = 3.0371970337762533 # m_max times more likely for a transconjugant to conjugate after receiving plasmid
t_half = 0.5001704428814664 # time before conjugation boost m_max is halved (h)
death_rate_recip = 0.06508671512237757 
death_rate_donor = 0.31819885071066445 # deaths per donor per hour
death_rate_trans = 0.26802855404172576 # deaths per donor per hour
death_delay = [0.5, 3] # h

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

def setup(sim):
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
 
    add_donors_recips(sim, n_start, donor_frac, len_x, len_y)
        
    # Add some objects to draw the models
    therenderer = Renderers.GLBacteriumRenderer(sim)
    sim.addRenderer(therenderer)
    
    # Specify how often data is saved
    sim.pickleSteps = 5
    

def init(cell):
    cell.targetVol = max(min_div_length, lognorm.rvs(div_len_stats[cell.cellType]['shape'], loc=div_len_stats[cell.cellType]['loc'], scale=div_len_stats[cell.cellType]['scale']))
    cell.growthRate = max(min_growthRate, np.random.normal(growthRate_stats[cell.cellType]['mu'], growthRate_stats[cell.cellType]['std']))
    cell.length = cell.targetVol/2
    cell.rfp_intensity = rfp_intensities[cell.cellType]
    cell.gfp_intensity = gfp_intensities[cell.cellType]
    cell.color = colors[cell.cellType]
    cell.transconjugant_flag = 0
    cell.time = 0
    cell.conj_time = -10
    cell.death_time = 0

def update(sim, cells):
    global death_rate_donor, death_rate_trans, death_rate_recip
    death_rates = {'donor': death_rate_donor, 'recip': death_rate_recip, 'trans': death_rate_trans} # put these two lines here for the parameter sweep
    
    for (id, cell) in cells.items():
        # Cell removal and division
        if cell.pos[1] > (len_y):
            cell.deathFlag = True
        if cell.volume > cell.targetVol and cell.deathFlag == False:
            cell.divideFlag = True
            
        # Trigger cell death countdown
        death_rate = death_rates[cell.cellType]
        if cell.death_time == 0 and random.random() < death_rate*dt:
            cell.death_time = cell.time
            cell.growthRate = 0
            cell.death_delay = np.random.uniform(low=death_delay[0], high=death_delay[1])
            
        # Cell death
        if cell.death_time != 0 and (cell.time - cell.death_time) >= cell.death_delay:
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
    
def add_donors_recips(sim, n_start, donor_frac, len_x, len_y, min_dist=2):
    """
    Randomly seed donor and recip cells in a square domain.
    Ensure that cells do not overlap
    @param sim          Simulator object
    @param n_start      Number of cells to start with (int)
    @param donor_frac   Fraction of initial donors (float)
    @param len_x        Length of domain (um)
    @param len_y        Height of domain (um)
    @param min_dist     Minimum distance between initial cell centroid positions (um)
    """
    
    # Create cells
    recips = round(n_start*(1-donor_frac))
    donors = round(n_start*donor_frac)
    
    recip_counter = 0
    donor_counter = 0
    
    # Ensure that cells don't overlap
    all_centroids = np.zeros((n_start, 3), dtype=int)
       
    while recip_counter < recips:
        pos = (random.uniform(0, len_x), random.uniform(0, len_y), 0)
        distances = np.linalg.norm(all_centroids - pos, axis=1)
        if any(distances < min_dist):
            pass 
        else:
            sim.addCell(cellType = 'recip', \
                        pos = pos, \
                        dir=(random.uniform(0, 1), random.uniform(0, 1), 0), \
                        rad = 0.6
                       )
            all_centroids[recip_counter] = pos
            recip_counter += 1
       
    while donor_counter < donors:
        pos = (random.uniform(0, len_x), random.uniform(0, len_y), 0)
        distances = np.linalg.norm(all_centroids - pos, axis=1)
        if any(distances < min_dist):
            pass 
        else:
            sim.addCell(cellType = 'donor', \
                        pos = pos, \
                        dir=(random.uniform(0, 1), random.uniform(0, 1), 0), \
                        rad = 0.6
                       )
            all_centroids[recip_counter+donor_counter] = pos
            donor_counter += 1
                        
def setparams(param_dict):
    global n_start
    n_start = param_dict['n_start']   
    
