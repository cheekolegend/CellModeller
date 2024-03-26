import random
import numpy as np
import scipy

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
    
    if len(cell.neighbours) > 0 and (cell.cellType == 'donor' or cell.cellType == 'trans'): #if a donor
        rand_num = random.random() 
        
        if rand_num < conj_prob:
            neigh = random.choice(cell.neighbours)
            recip = cells[neigh]
            
            if recip.cellType == 'recip': #if a recipient
                convert_to_transconjugant(recip, sim, neigh, rfp_intensities, colors)
                
def conjugation_delay(sim, cells, cell, conj_per_t, dt, rfp_intensities, colors,
                      delay_mean, delay_std):
    """
    Incorporates a delay between plasmid uptake and being able to conjugate in transconjugants.
    Simulation requires initializing cell.conj_time
    
    Donors activate conjugation genes a constant frequency with a delay factor. The delay between uptaking the plasmid and conjugating is normally distributed.
    If conjugation is activated, a neighbour is selected at random and converted to a transconjugant.
    
    @param sim          CellModeller sim object
    @param cells        cellstates dict
    @param cell         cellstate object
    @param conj_per_t   conjugation events per donor-recipient contact per time
    """
   
    conj_prob = conj_per_t*dt
 
    if len(cell.neighbours) > 0 and (cell.cellType == 'donor' or cell.cellType == 'trans'): #if a donor
        time_passed = cell.time - cell.conj_time
        p_delay = scipy.stats.norm.cdf(time_passed, loc=delay_mean, scale=delay_std)
        rand_num = random.random() 
        
        if rand_num < conj_prob*p_delay:
            neigh = random.choice(cell.neighbours)
            recip = cells[neigh]
            
            if recip.cellType == 'recip':
                convert_to_transconjugant(recip, sim, neigh, rfp_intensities, colors)
                recip.conj_time = recip.time
                
def convert_to_transconjugant(recip, sim, neigh, rfp_intensities, colors):
    recip.cellType = 'trans'
    recip.rfp_intensity = rfp_intensities[recip.cellType]
    recip.transconjugant_flag = 1
    recip.color = colors[recip.cellType]
    sim.hgt_events[neigh] = {'stepNum':sim.stepNum, 'recip_id':neigh, 'donor_id':id} #{recip_id: (stepNum, donor_id), ...}    