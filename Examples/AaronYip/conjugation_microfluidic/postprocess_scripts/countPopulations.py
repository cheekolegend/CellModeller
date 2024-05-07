"""
This script reads in .pickle files containing cellStates dictionaries and outputs
a spreadsheet containing the total population of each cellType at each time step.
The script can be used for an arbitrary number of populations

Ensure that the data files have cellType starting from 0.

Instructions:
    1. Optional: input the user_max_cell_type if you don't want extra empty headings in the csv output
        e.g., for monoculture, the max_cell_type is 0
    2. Run this script and provide an argument that contains the path of the directory containing all the .pickle files.
    
    The script can also be ran as a function by providing the csv_file path as an argument to main
"""

import sys
import os
import math
import numpy as np
import pickle
from helperFunctions import create_pickle_list, get_max_cell_type, load_cellStates, read_time_step
import csv
import re
import pandas as pd

# For making data output cleaner
use_max_cell_type_custom = False #True if you want data to look nice 
max_cell_type_custom = 2 #change the RHS to the largest cellType in your data set

# Defaults for script
max_cell_type_default = 3 

def main(file_dir_path = ''):
    # Reading files and paths
    if not file_dir_path:
        file_dir_path = sys.argv[1] 
    output_file = os.path.join(file_dir_path, 'cell_type_populations.csv')
    
    # Process data
    pickle_list = create_pickle_list(file_dir_path)
    if use_max_cell_type_custom:
        run_population_counts_and_write_to_csv(file_dir_path, pickle_list, output_file, user_max_type)
    else:
        run_population_counts_and_write_to_csv(file_dir_path, pickle_list, output_file)


def count_cell_types(cells, max_cell_type = max_cell_type_default):   
    """
    Counts the total number of each cellType at a given time step.
    @param  cells           cellStates dict    
    @param  max_cell_type   int specifying the largest cellType index in the experiment
    
    @return populations     nparray with the total population for each cellType
    """
       
    # Initializations
    populations = np.zeros((max_cell_type + 1,), dtype = int)
    
    # Iterate through cellStates DataFrame and add to the populations array based on cellType
    for id, cell in cells.items():
        cell_type = cell.cellType
        for cell_type_idx in range(0, max_cell_type + 1):
            if cell_type_idx == cell_type:
                populations[cell_type] += 1

    return populations
  

def run_population_counts_and_write_to_csv(file_dir_path, pickle_list, output_file, max_cell_type = max_cell_type_default): 
    """
    Run count_cell_types at each time step, then output to csv file.
    
    @param pickle_list      list of .pickle files to process
    @param output_file      str for path of the output csv file
    @param max_cell_type    int specifying the largest cellType index in the experiment
    """
    # Write column headers
    header = ['Time step']
    for i in range(0, max_cell_type + 1):
        cell_type_string = f"cellType {i}"
        header.append(cell_type_string) 

    # Write to csv file
    with open (output_file, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
    
        #Write row of data for each time step
        for file in pickle_list:
            print('Writing : ', file)
            step = read_time_step(file)
            cells = load_cellStates(file_dir_path, file)
            populations = count_cell_types(cells, max_cell_type)
            writer.writerow(np.concatenate((step,populations), axis = None))

def count_final_populations(population_csv):
    """
    Read the last row of a population count csv file and outputs the final populations.
    
    @param  population_csv  csv file written by the run_population_counts_and_write_to_csv function
    @return final_populations pandas series containing the final population of each cellType 
    """
    data = pd.read_csv(population_csv)
    final_step = data.iloc[-1]
    final_populations = final_step[1:-1]
    return final_populations
        
if __name__ == "__main__":
    main()
