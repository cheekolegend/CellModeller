"""
This module contains general functions for working with summary statistic data
Import modules and add functions as necessary.
"""

import glob
import os
import pickle
import re #read regex
import networkx as nx
    
def create_pickle_list(directory):  
    """
    Creates a list of pickle file names.
    
    @param  directory   string containing .pickle files
    @return pickle_list list of .pickle files
    """
    
    pickle_list = []
    for file in sorted(os.listdir(directory)):
        if file.endswith(".pickle"):
            pickle_list.append(file)
    return pickle_list
    
def create_pickle_list_full_path(directory):   
    """
    Creates a list of pickle file names with structure "directory/step-xxxx.pickle"
    
    @param  directory   string containing .pickle files
    @return pickle_list list of .pickle files
    """
    pickle_list = [pickle_file for pickle_file in glob.glob(directory + "/*.pickle")]
    return pickle_list

def load_cellStates(file_dir, pickle_file):
    """
    Loads cellStates data from a pickle file.
    
    @param  file_dir    directory containing all pickle_files
    @param  pickle_file pickle_file name
    @return cellStates  cellStates for the given pickle file       
    """
    pickle_full_path = os.path.join(file_dir, pickle_file)
    data = pickle.load(open(pickle_full_path, 'rb'))
    cells = data['cellStates']
    
    return cells
    
def load_cellStates_full_path(pickle_file_full_path):
    """
    Loads cellStates data from a pickle file given the full path.
    
    @param  pickle_file_full_path   full path to a .pickle file
    @return cellStates              cellStates for the given pickle file       
    """
    data = pickle.load(open(pickle_file_full_path, 'rb'))
    cells = data['cellStates']
    
    return cells
    
def read_time_step(pickle_file):
    """
    Read time step from a pickle file.
    
    @param  pickle_file pickle file name
    @return step        time step
    """
    match = re.search('step-(\d+)', pickle_file, re.IGNORECASE)
    step = int(match.group(1))
    
    return step
    
def get_max_cell_type(cellStates):
    """
    Loops through a cellStates dict to find the max_cell_type.
    
    @param  cellStates      cellStates dict from CellModeller
    @return max_cell_type   largest value of cellType
    """
    max_cell_type = 0
    for cell in cellStates:
        read_type = cellStates[cell].cellType
        if read_type > max_cell_type:
            max_cell_type = read_type
    
    return max_cell_type
   
def generate_network(cellStates):
    """
    Creates a Graph from the cellStates dict of a single time step.
    
    @param  cellStates  cellStates dict of a particular time step
    @return G           Graph describing contact network of all cells
    """
    
    G = nx.Graph()
    it = iter(cellStates)
    
    for it in cellStates:
        cell = cellStates[it]
        G.add_node(it, cellType=cell.cellType, color=cell.color, pos=cell.pos)
        for nb in cellStates[it].neighbours:
            G.add_edge(it, nb)
            
    return G
    

