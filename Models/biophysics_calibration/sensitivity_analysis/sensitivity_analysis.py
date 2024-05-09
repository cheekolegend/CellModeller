import os
import pickle
import sys
import subprocess
import string
import shutil
import uuid

import numpy as np
import pandas as pd
import re
from CellModeller.Simulator import Simulator

# Local/custom modules
sys.path.append(os.path.join(os.path.expanduser('~'), 'CellModeller/Scripts/summaryStatistics/biophysics/'))
import anisotropy
import aspect_ratio
import density_calculation
import growth_rate_analysis2
sys.path.append(os.path.join(os.path.expanduser('~'), 'CellModeller/Scripts/summaryStatistics/'))
import helperFunctions

cellmodeller_module = "simulation_module.py"
max_cells = 1000

'''
1. Define parametric space
2. For each parameter set:
    a. Run the model and calculate summary stats
    b. Output summary stats for each parameter value
    c. Append to a df
3. Post-process
'''

def run_model(parameters):
    # Defining input/output locations
    export_path = "data/" + str(uuid.uuid4())
    sys.stdout = open(os.devnull, 'w') # Disable printing from simulations
    
    # Run CellModeller simulation
    simulate(cellmodeller_module, parameters, export_path)
    sys.stdout = sys.__stdout__ # Re-enable printing
    print("Simulation completed")
    
    # Load cellStates
    pickle_list = helperFunctions.create_pickle_list(export_path)
    cells = helperFunctions.load_cellStates(export_path, pickle_list[-1]) # load last pickle file
    
    # Extract time parameters from simulation (could hard-code to avoid unnecessary repetition)
    sim_file = os.path.abspath(cellmodeller_module)
    dt = sim_time_parameters(sim_file)
    
    # Calculate summary statistics
    summary_stats = {}
    summary_stats['knee'], summary_stats['A'], summary_stats['K'], summary_stats['C'] = growth_rate_analysis2.get_growth_rate_vs_position_parameter(cells, dt, bin_radius=2)
    summary_stats['density_parameter'] = density_calculation.main(cells)
    summary_stats['order_parameter'] = anisotropy.main(cells)
    summary_stats['aspect_ratio'] = aspect_ratio.main(cells)
    
    return summary_stats

def simulate(modfilename, params, export_path):
    """
    Same as batch.py with modifications to allow external parameters and termination based on stepNum
    
    @param modfilename  module name (e.g., "Tutorial1.py")
    @param export_path  output directory name
    @param params       dictionary of parameter values passed to the simulation
    """
    # Get module file name and append module to system path
    (path,name) = os.path.split(modfilename)
    modname = str(name).split('.')[0]
    sys.path.append(path)
    
    # Extract time parameters from simulation
    sim_file = os.path.abspath(modfilename)
    dt = sim_time_parameters(sim_file)
    
    # Run simulation
    sim = Simulator(modname, dt, saveOutput=True, outputDirName=export_path, psweep=True, params=params)
    while len(sim.cellStates) <= max_cells:
        sim.step()
        
    # Export pickle at final timestep
    sim.writePickle()
    
def sim_time_parameters(file_path):
    """
    Extract the simulation time step from the module file. 
    Must be written as dt = X.Y in the module file
    
    @param  file_path   str of file name containing parameters to be modified
    @return dt          Simulation time step specified in CellModeller module file
    """
    search_dt = f'dt = (\d+\.\d+)'
    with open(file_path, "r+") as file:
        file_contents = file.read()
        dt = re.findall(search_dt, file_contents)
    return float(dt[0])
    
def write_to_pickle(data, file_name):
    # Open a file, where you want to store the data
    file = open(file_name, 'wb')

    # Dump information to that file
    pickle.dump(data, file)

    # Close the file
    file.close()

def main():
    # Define parametric space
    gamma = np.array([1, 10, 50, 100, 1000])
    reg_param = 1.0/gamma
    
    # Sensitivity analysis
    df = pd.DataFrame(columns=['gamma', 'reg_param', 
                                'knee',
                                'growth_rate_vs_dist', 
                                'density_parameter', 
                                'order_parameter',
                                'aspect_ratio'])
    
    # For counting
    n_sims = len(gamma)*len(reg_param)
    sim_counter = 0
    
    # Main loop 
    for gamma_i in gamma:
        for reg_param_j in reg_param:
            # Run model
            parameters = {'gamma': gamma_i, 'reg_param': reg_param_j}
            try:
                summary_stat_dict = run_model(parameters)
            except:
                print(f"Run failed with params: gamma = {gamma_i}, reg_param = {reg_param_j}")
                summary_stat_dict = {'growth_rate_vs_dist': 0, 
                                     'knee': 0,
                                     'density_parameter': 0, 
                                     'order_parameter': 0, 
                                     'aspect_ratio': 0}
            # Terminal output
            sim_counter += 1
            print(f'{sim_counter}/{n_sims} completed')
            
            # Store in df
            new_row = pd.DataFrame({'gamma': gamma_i, 'reg_param': reg_param_j, 
                                    'knee': summary_stat_dict['knee'],
                                    'growth_rate_vs_dist': summary_stat_dict['K'],
                                    'density_parameter': summary_stat_dict['density_parameter'],
                                    'order_parameter': summary_stat_dict['order_parameter'],
                                    'aspect_ratio': summary_stat_dict['aspect_ratio']
                                    },
                                    index=[0])
                                   
            df = df.append(new_row, ignore_index=True) # only for pandas < 1.4.0
            # For pandas >= 1.4.0
            #df = df.concat([df, new_row], ignore_index=True)
    
    # Store df in pickle file        
    write_to_pickle(df, 'sensitivity_analysis_results3.pickle')

# Make sure we are running as a script
if __name__ == "__main__": 
    main()
