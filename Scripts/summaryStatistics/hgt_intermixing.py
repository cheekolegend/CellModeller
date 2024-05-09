import sys
import os
import pickle
import CellModeller
import networkx as nx
from helperFunctions import create_pickle_list, get_max_cell_type, load_cellStates, read_time_step, generate_network
import intermixing
import hgt
from countPopulations import count_cell_types
import csv
import math
import numpy as np
from itertools import combinations

def main(file_dir_path=''):
    # Reading files and paths
    if not file_dir_path:
        file_dir_path = sys.argv[1] 
    output_file = os.path.join(file_dir_path, 'hgt_intermixing.csv')
    
    # Process data
    pickle_list = create_pickle_list(file_dir_path)
    
    # Spreadsheet columns
    header = ['Time step', 
              'Cells',
              'cellType0', 'cellType1','cellType2',
              'Entropy',
              'Max Entropy',
              'Contagion Index',
              'Average Neighbours',
              'HGT Events',
              'Conj. Eff.',
              'Entropy Score'
              ]
              
    with open (output_file, 'w', newline = '') as csvfile:
        # Initialize csv file
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        # Get max_cell_type from final step
        final_step = pickle_list[-1]
        cells = load_cellStates(file_dir_path, final_step)
        max_cell_type = get_max_cell_type(cells)
        t = max_cell_type # usually t=max_cell_type+1, but need to adjust if we don't want HGT to interfere with intermixing calcs.
        
        # Loop through all time steps
        for file in pickle_list:
            step = read_time_step(file)
            print('Reading step ', step)
            cells = load_cellStates(file_dir_path, file)
            
            # Generate network
            G = generate_network(cells)
            G = adjust_cell_types(G, recip_types=[0], trans_types=[2])
            
            # Extra calculations
            n_cells = nx.number_of_nodes(G)
            edge_counts = intermixing.count_edge_types(G, max_cell_type)
            
            # Calculate intermixing summary stats
            entropy = intermixing.calc_entropy(edge_counts, nx.number_of_edges(G))
            contagion = intermixing.contagion_index(entropy, t)
            max_entropy = intermixing.calc_max_entropy(t)
            avg_neighbours = intermixing.mean_degree(G)
            entropy_score = entropy/max_entropy
            
            # Count populations
            populations = count_cell_types(cells, max_cell_type)
            
            # Count HGT events
            hgt_events = hgt.count_hgt_events(cells)
            hgt_efficiency = hgt.calc_hgt_efficiency(populations, recip_types=[1], trans_types=[2])
            
            # Write to spreadsheet 
            writer.writerow(np.concatenate(
                (step,
                 n_cells,
                 populations,
                 entropy,
                 max_entropy,
                 contagion,
                 avg_neighbours,
                 hgt_events,
                 hgt_efficiency,
                 entropy_score
                 ), axis = None)) 
                 
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
    
       
if __name__ == "__main__":
    main()
