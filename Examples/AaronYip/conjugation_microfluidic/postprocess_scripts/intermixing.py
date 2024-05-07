import sys
import os
import pickle
import CellModeller
import networkx as nx
from helperFunctions import create_pickle_list, get_max_cell_type, load_cellStates, read_time_step, generate_network
import csv
import math
import numpy as np
from itertools import combinations

def main():
    # Reading files and paths
    file_dir_path = sys.argv[1] 
    output_file = os.path.join(file_dir_path, 'intermixing.csv')
    
    # Process data
    pickle_list = create_pickle_list(file_dir_path)
    
    # Spreadsheet columns
    header = ['Time step', 
              'Cells',
              'Entropy',
              'Max Entropy',
              'Contagion Index',
              'Average Neighbours',
              ]
              
    with open (output_file, 'w', newline = '') as csvfile:
        # Initialize csv file
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        # Get max_cell_type from final step
        final_step = pickle_list[-1]
        cells = load_cellStates(file_dir_path, final_step)
        max_cell_type = get_max_cell_type(cells)
        
        # Loop through all time steps
        for file in pickle_list:
            step = read_time_step(file)
            cells = load_cellStates(file_dir_path, file)
            
            # Generate network
            G = generate_network(cells)
            
            # Extra calculations
            n_cells = nx.number_of_nodes(G)
            edge_counts = count_edge_types(G, max_cell_type)
            t = max_cell_type + 1
            
            # Calculate intermixing summary stats
            entropy = calc_entropy(edge_counts, nx.number_of_edges(G))
            contagion = contagion_index(entropy, max_cell_type + 1)
            max_entropy = np.log(t**2 + t) - np.log(2)
            avg_neighbours = mean_degree(G)
            
            # Write to spreadsheet 
            writer.writerow(np.concatenate(
                (step,
                 n_cells,
                 entropy,
                 max_entropy,
                 contagion,
                 avg_neighbours
                 ), axis = None))
                 
def get_intermixing_stats(G, cellTypes=['donor', 'recip', 'trans']):
    '''
    Calculate intermixing metrics for each time point in the lineage tree.
    @param  subgraphs               list of networkx Graphs with cell lineages and contacts at each time step
    @param  cellTypes               list of cellTypes
    @return intermixing_stats_dict  dict of metrics
    '''
    t = len(cellTypes)
    
    # Initialize storage variables
    intermixing_stats_dict = {'entropy': [], 'contagion': [], 'avg_neighbours_donor': [], 'avg_neighbours_recip': [], 'avg_neighbours_trans': []}
    
    # Intermixing calculations
    edge_counts = count_edge_types(G, cellTypes)
    entropy = calc_entropy(edge_counts, nx.number_of_edges(G))
    intermixing_stats_dict['entropy'].append(entropy)
    intermixing_stats_dict['contagion'].append(contagion_index(entropy, t))
    for cellType in cellTypes:
        intermixing_stats_dict[f'avg_neighbours_{cellType}'].append(mean_degree_cellType(G, 'cellType', cellType))
        
    return intermixing_stats_dict
                 
def mean_degree_cellType(G, attribute, attribute_value):
    """
    Calculate the mean degree of a nodes in a graph containing a certain attribute value 
    @param  G               networkx Graph
    @param  attribute       str; attribute of interest
    @param  attribute_value value of the attribute of interest
    @return                 mean degree of nodes with certain attribute value
    """
    # Filter nodes based on the attribute value and calculate their degrees
    degrees = [G.degree(node) for node, attr in G.nodes(data=True) if attr.get(attribute) == attribute_value]

    # Calculate the mean degree of the filtered nodes
    if len(degrees) > 0:
        return sum(degrees)/len(degrees)
    else:
        return 0
            
def count_edge_types(G, cellTypes):
    """
    Count the total number of conctacts between each cellType (single-counting)
    e.g. how many green/green contacts, green/red contacts, red/red contacs?
    
    @param  G               Graph of the cellStates
    @param  cellTypes       the largest cellType in cellStates
    @return adjacency_dict  dict containing total counts of adjacency types 
    """       
    # Generate edge list
    edge_list = []
    
    # Note: this method double-counts edges in a non-directed graph
    for n, nbrs in G.adjacency(): #n = node, nbrs = list of neighbours
        node1 = G.nodes[n]['cellType']
        for nbr in nbrs: #nbr = neighbour of node n, eattr is the value of nbr
            node2 = G.nodes[nbr]['cellType']
            edge_list.append((node1, node2))
                        
    # List of adjacency types
    adjacency_type = []
    pairs = combinations(cellTypes, 2)
    for i in pairs:
        adjacency_type.append(i) #pairs of non-equal cellType
    for i in cellTypes:
        adjacency_type.append((i,i)) #pairs of identical cellType
       
    #Create dict with each adjacency type (unordered)
    adjacency_dict = {}
    for pair in adjacency_type:
        adjacency_dict[pair] = 0
    
    # Count edge types in the graph
    for pair in adjacency_type:
        if pair[0] == pair[1]:
            adjacency_dict[pair] += edge_list.count(pair)/2 #divide by 2 since those are double-counted
        else:
            adjacency_dict[pair] += edge_list.count(pair) #no division since pairs have been listed only once and they can't be double-counted due to order
    
    return adjacency_dict
    
def calc_entropy(adjacency_counts, total_edges):
    """
    Calculate Shannon entropy of a network:
    E = -sum(p_i*ln(p_i))
    
    @param  adjacency_counts    counts of each adjacency type
    @param  total_edges         total edges in the graph
    @return entropy             Shannon entropy
    """
    if total_edges > 0: #avoid dividing by zero when no cell contacts present
        probabilities = np.array(list(adjacency_counts.values()))/total_edges
        log_probabilities = np.log(probabilities + 1e-7)
        entropy = -np.dot(probabilities, log_probabilities)
    else:
        entropy = 0
    return entropy
    
def calc_max_entropy(t):
    """
    Calculate maximum entropy of a system where order of adjacency is not preserved [e.g., (1,0) = (0,1)].
    
    @param  t           number of different cellTypes in the system
    @return max_entropy entropy of the system assuming events are uniformly distributed
    """
    return np.log(t**2 + t) - np.log(2)
    
def contagion_index(entropy, t):
    """
    Calculate contagion index for network where order is not preserved (i.e. no double-counts).
    For more information, see:
    Riitters, K.H., O'Neill, R.V., Wickham, J.D. et al. A note on contagion indices for landscape analysis. Landscape Ecol 11, 197–202 (1996). https://doi.org/10.1007/BF02071810
    
    @param  entropy         Shannon entropy of the graph 
    @param  t               Number of attribute types for a graph
    @return contagion       Contagion index when order is not preserved
    """
    max_entropy = calc_max_entropy(t)
    contagion = 1 - entropy/max_entropy
    
    return contagion

def mean_degree(G):
    """
    Calculate the average number of neighbours each cell has
    
    @param  G           Graph of the cellStates
    @return mean_degree average degree of each node
    """
    degrees = nx.average_neighbor_degree(G)
    mean_degree = np.mean(list(degrees.values()))
    return mean_degree
       
if __name__ == "__main__":
    main()
