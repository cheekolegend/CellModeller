import random
import pickle
import math
import numpy as np
import scipy
from shapely.geometry import Polygon

'''
Contains various conjugation models to use in CellModeller simulation file.
'''

def conjugation_basic(sim, cells, cell, conj_per_t, dt, rfp_intensities, colors):
    """
    Donors activate conjugation genes a constant frequency. 
    If conjugation is activated, a neighbour is selected at random and converted to a transconjugant.
    
    @param sim          CellModeller sim object
    @param cells        cellstates dict
    @param cell         cellstate object
    @param conj_per_t   conjugation events per donor-recipient contact per time
    """
    conj_prob = conj_per_t*dt
    
    if len(cell.neighbours) > 0 and (cell.cellType == 'donor' or cell.cellType == 'trans') and cell.growthRate > 0: #if a donor
        rand_num = random.random() 
        
        if rand_num < conj_prob:
            neigh = random.choice(cell.neighbours)
            recip = cells[neigh]
            
            if recip.cellType == 'recip' and recip.growthRate > 0: #if a recipient
                convert_to_transconjugant(cell, recip, sim, rfp_intensities, colors)
                        
def convert_to_transconjugant(donor, recip, sim, rfp_intensities, colors):
    recip.cellType = 'trans'
    recip.rfp_intensity = rfp_intensities[recip.cellType]
    recip.transconjugant_flag = 1
    recip.color = colors[recip.cellType]
    sim.hgt_events['stepNum'].append(sim.stepNum) # Need to have this initialized outside of loop in update function: hgt_events[sim.stepNum] = {'donor_id': [], 'recip_id': []}
    sim.hgt_events['donor_id'].append(donor.id)
    sim.hgt_events['recip_id'].append(recip.id)
