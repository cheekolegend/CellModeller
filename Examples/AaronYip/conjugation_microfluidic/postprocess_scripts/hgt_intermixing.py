import sys
import os
import pickle
import CellModeller
import networkx as nx
from .helperFunctions import create_pickle_list, get_max_cell_type, load_cellStates, load_data, read_time_step, generate_network
from . import intermixing, hgt
#from helperFunctions import create_pickle_list, get_max_cell_type, load_cellStates, read_time_step, generate_network
#import intermixing, hgt
import csv
import math
import numpy as np
import pandas as pd
from itertools import combinations

def main(file_dir_path='', dt=1/60):
    # Reading files and paths
    if not file_dir_path:
        file_dir_path = sys.argv[1] 
    
    # Process data
    pickle_list = create_pickle_list(file_dir_path)
    
    df = pd.DataFrame()
    
    # Loop through all time steps
    for file in pickle_list:
        data = load_data(file_dir_path, file)
        cells = data['cellStates']
        hgt_events_data = data['hgt_events']
        stepNum = data['stepNum']
        time = stepNum*dt
        
        # Generate network
        G = generate_network(cells)
        #G = adjust_cell_types(G, recip_types=[0], trans_types=[2])
        
        # Extra calculations
        n_cells = nx.number_of_nodes(G)
        
        # HGT measurements
        donor_recip_cts, trans_recip_cts = hgt.count_dr_tr_contacts(G, donor_label='donor', recip_label='recip', trans_label='trans')
        if stepNum in hgt_events_data:
            hgt_events = len(hgt_events_data[stepNum]['donor_id'])
        else:
            hgt_events = 0
        #hgt_efficiency = hgt.calc_hgt_efficiency(populations, recip_types=[1], trans_types=[2])
        
        # Count populations
        count_cellTypes_list = count_cell_types(G, cellTypes=['donor', 'recip', 'trans'])
        
        # Calculate intermixing summary stats
        intermixing_stats = intermixing.get_intermixing_stats(G, cellTypes=['donor', 'recip', 'trans'])   
        
        output_dict = {'Time (h)': time, 
                       'donor_cts': donor_recip_cts, 
                       'trans_cts': trans_recip_cts, 
                       'event_counts': hgt_events, 
                       'donors': count_cellTypes_list['donor'], 
                       'recips': count_cellTypes_list['recip'],
                       'trans': count_cellTypes_list['trans'],
                       'entropy': intermixing_stats['entropy'][0],
                       'contagion': intermixing_stats['contagion'][0],
                       'avg_neighbours_donor': intermixing_stats['avg_neighbours_donor'][0],
                       'avg_neighbours_recip': intermixing_stats['avg_neighbours_recip'][0],
                       'avg_neighbours_trans': intermixing_stats['avg_neighbours_trans'][0],
                       }
        df_single = pd.DataFrame(output_dict, index=[0])
        df = pd.concat([df, df_single])
        
    # Calculated values
    df['cells'] = df['donors'] + df['recips'] + df['trans']
    df['recip_frac'] = df['recips']/df['cells']
    df['donor_frac'] = df['donors']/df['cells']
    df['trans_frac'] = df['trans']/df['cells']
    df['donor_trans_cts'] = df['donor_cts'] + df['trans_cts']
    df['conj_per_contact_per_time'] = df['event_counts']/df['donor_trans_cts']/dt
    df['conj_efficiency'] = df['trans']/df['donor_trans_cts']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df
                 
def adjust_cell_types(G, recip_types=[1], trans_types=[2]):
    """
    Converts transconjugant cellTypes into their original recipient types.
    recip_types must align with trans_types (i.e.cellType = 1 becomes cellType = 2 after being infected)
    
    @param  G           Graph of the cellStates
    @param  recip_types recipient cellTypes
    @param  trans_types transconjugant cellTypes
    @return G           Adjusted graph of cellStates, where transconjugants have been converted to original recip type
    """
    
    for n in G:
        for index, trans_type in enumerate(trans_types):
            if G.nodes[n]['cellType'] == trans_type:
                G.nodes[n]['cellType'] = recip_types[index]
                
    return G
        
def count_cell_types(G, cellTypes=['donor', 'recip', 'trans']):
    """
    Count the number of each type of cell in a subgraph representing a single time point.
    """
    # Initialize storage variable
    cellType_counts = {}
    for cellType in cellTypes:
        cellType_counts[cellType] = 0
    
    # Loop through subgraph and count nodes
    for node, cellType in G.nodes(data='cellType'):
        cellType_counts[cellType] += 1
        
    return cellType_counts
    
       
if __name__ == "__main__":
    dt = 1/60 #h
    df = main(dt=dt)
    print(df)
