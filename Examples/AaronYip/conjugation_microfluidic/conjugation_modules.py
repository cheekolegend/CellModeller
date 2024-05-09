import random
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
    
    if len(cell.neighbours) > 0 and (cell.cellType == 'donor' or cell.cellType == 'trans'): #if a donor
        rand_num = random.random() 
        
        if rand_num < conj_prob:
            neigh = random.choice(cell.neighbours)
            recip = cells[neigh]
            
            if recip.cellType == 'recip': #if a recipient
                convert_to_transconjugant(cell, recip, sim, rfp_intensities, colors)
                
def conjugation_contact(sim, cells, cell, conj_per_t, dt, rfp_intensities, colors):
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
            polygon1 = get_bounding_polygon(cell.pos, cell.length, cell.radius, cell.dir, reach=1.0, donor=cell.cellType)
        
            # Calculate probability of conjugation (or miss) based on overlap area
            p = []
            for neigh in cell.neighbours:
                recip = cells[neigh]
                polygon2 = get_bounding_polygon(recip.pos, recip.length, recip.radius, recip.dir, reach=1.0, donor=recip.cellType)
                p.append(polygon1.intersection(polygon2).area/polygon1.area)
            p_miss = 1 - sum(p)
            
            # Select recipient (or miss) with probability proportional to overlap area
            p.append(p_miss)
            selections = cell.neighbours
            selections.append(cell.id) # This will be a miss because cell cannot be a recip.
            recip = cells[random.choices(selections, weights=p)[0]]
            
            # Convert to transconjugant if selected cell is a recipient type
            if recip.cellType == 'recip': #if a recipient
                convert_to_transconjugant(cell, recip, sim, rfp_intensities, colors)
                
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
                convert_to_transconjugant(cell, recip, sim, rfp_intensities, colors)
                recip.conj_time = recip.time
                
#TODO: code for transitory-derepression of new transconjugants
                
def convert_to_transconjugant(donor, recip, sim, rfp_intensities, colors):
    recip.cellType = 'trans'
    recip.rfp_intensity = rfp_intensities[recip.cellType]
    recip.transconjugant_flag = 1
    recip.color = colors[recip.cellType]
    sim.hgt_events['stepNum'].append(sim.stepNum) # Need to have this initialized outside of loop in update function: hgt_events[sim.stepNum] = {'donor_id': [], 'recip_id': []}
    sim.hgt_events['donor_id'].append(donor.id)
    sim.hgt_events['recip_id'].append(recip.id)
    
def convert_cellmodeller_orientation_to_radians(cell_dir):
    """
    Converts cell.dir to an orientation in radians

    @param  cell_dir    list containing cell direction in [x, y, z]
    @return orientation cell orientation in radians (float)

    Note: only valid for 2D case
    """
    # Convert vector into a unit vector
    magnitude = np.linalg.norm(cell_dir)
    cell_dir_unit_vector = cell_dir / magnitude

    # Calculate orientation in radians
    if cell_dir_unit_vector[0] != 0:
        orientation = np.arctan(cell_dir_unit_vector[1] / cell_dir_unit_vector[0])
    else:
        # If x component is zero, cell is straight up
        orientation = math.pi / 2

    return orientation
    
def get_bounding_polygon(pos, length, radius, cell_dir, reach=1.0, donor=False):
    """
    Puts a bounding polygon around the cell. Donors have a hexagonal box, whereas recipients have a rectangular box.
    Here, length refers to the length of the cylinder (i.e. CellModeller length)
    
    Recipient:
    p12--------p11
    |           |
    |           |
    |           |
    |           |
    p22--------p21
    
    Donor:
            end1
        ----    ----
    ----            ----
    p12                 p11
    |                   |
    |                   |
    |                   |
    |                   |
    p22                 p21
    ----            ----
        ----    ----
            end2
    """
    angle = convert_cellmodeller_orientation_to_radians(cell_dir)
    
    # Get poles of cell
    end1_x = pos[0] + (length/2 + radius)*np.cos(angle) 
    end1_y = pos[1] + (length/2 + radius)*np.sin(angle) 
    end2_x = pos[0] - (length/2 + radius)*np.cos(angle) 
    end2_y = pos[1] - (length/2 + radius)*np.sin(angle) 
    
    if donor == False or donor == 'recip': # Get a bounding box
        # end1 +- radius
        x11 = end1_x + radius*np.cos(angle + math.pi/2)
        y11 = end1_y + radius*np.sin(angle + math.pi/2)
        x12 = end1_x - radius*np.cos(angle + math.pi/2)
        y12 = end1_y - radius*np.sin(angle + math.pi/2)
        
        # end2 +- radius
        x21 = end2_x + radius*np.cos(angle + math.pi/2)
        y21 = end2_y + radius*np.sin(angle + math.pi/2)
        x22 = end2_x - radius*np.cos(angle + math.pi/2)
        y22 = end2_y - radius*np.sin(angle + math.pi/2)
        
        bounding_polygon = Polygon([(x11,y11), (x21,y21), (x22,y22), (x12,y12)])
    
    elif donor == True or donor == 'donor' or donor =='trans':
        # end1 +- (radius + reach)
        x11 = pos[0] + length/2*np.cos(angle) + (radius + reach)*np.cos(angle + math.pi/2)
        y11 = pos[1] + length/2*np.sin(angle) + (radius + reach)*np.sin(angle + math.pi/2)
        x12 = pos[0] + length/2*np.cos(angle) - (radius + reach)*np.cos(angle + math.pi/2)
        y12 = pos[1] + length/2*np.sin(angle) - (radius + reach)*np.sin(angle + math.pi/2)
        
        # end2 +- (radius + reach)
        x21 = pos[0] - length/2*np.cos(angle) + (radius + reach)*np.cos(angle + math.pi/2)
        y21 = pos[1] - length/2*np.sin(angle) + (radius + reach)*np.sin(angle + math.pi/2)
        x22 = pos[0] - length/2*np.cos(angle) - (radius + reach)*np.cos(angle + math.pi/2)
        y22 = pos[1] - length/2*np.sin(angle) - (radius + reach)*np.sin(angle + math.pi/2)
        
        bounding_polygon = Polygon([(end1_x,end1_y), (x11,y11), (x21,y21), (end2_x, end2_y), (x22,y22), (x12,y12)])
        
    return bounding_polygon
    
def intersection_area(G, id1, id2):
    cell1 = G.nodes[id1]
    cell2 = G.nodes[id2]
    
    polygon1 = get_bounding_polygon(cell1['centroid'], cell1['length'], cell1['radius'], cell1['dir'], reach=1.0, donor=cell1['cellType'])
    polygon2 = get_bounding_polygon(cell2['centroid'], cell2['length'], cell2['radius'], cell2['dir'], reach=1.0, donor=cell2['cellType'])
    
    return polygon1.intersection(polygon2).area
    
def intersection_area_cs_dict(cell1, cell2):   
    polygon1 = get_bounding_polygon(cell1.pos, cell1.length, cell1.radius, cell1.dir, reach=1.0, donor=cell1.cellType)
    polygon2 = get_bounding_polygon(cell2.pos, cell2.length, cell2.radius, cell2.dir, reach=1.0, donor=cell2.cellType)
    
    return polygon1.intersection(polygon2).area