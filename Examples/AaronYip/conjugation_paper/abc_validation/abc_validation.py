import os
import sys
import re
import uuid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import pickle
import datetime
import warnings
import copy
from CellModeller.Simulator import Simulator
from Scripts.hgt_scripts import hgt_intermixing
import concurrent.futures 
import multiprocessing


np.seterr(all="ignore") #suppress warnings for cleaner output
warnings.filterwarnings("ignore") # ignore all warnings
n_attempts = 3
sim_time = 16 # Hours to simulate
imaging_frequency = 5/60

# Write down time for output
time_arr = list(np.arange(0, sim_time + imaging_frequency/2, imaging_frequency)) # 5/60 is the imaging frequency of the experiment (h)
if time_arr[-1] > sim_time:
    time_arr.pop(-1)

'''
Instructions:
0. Setup the simulation module accordingly; ensure there is a time step variable dt = x.y
0. Ensure that cellmodeller module and conjugation modules are up to date in this directory
1. Define parametric space in main()
2. Run. This does the following:
    For each parameter set:
        a. Run the model and calculate summary stats
        b. Output summary stats for each parameter value
        c. Append to a df
3. Post-process the data

To run multiple simulations for the same parameter value, run this script multiple times,
each time assigning a new name to the output file.
'''

def run_model(cellmodeller_module, scene, parameters):
    """
    The "model" run by pyabc. Every time a simulation is run, the following occurs:
    1. Create a unique export path
    2. Run simulation and store data in export path
    3. Calculate and return summary statistics from the simulation
    
    The function must only have "parameters" as an argument to conform to pyabc's framework. 
    
    @param  parameters      dict of parameters specified in the prior distributions
    @return summary_stats   dict of calculated summary statistics
    """
    for attempt in range(n_attempts):
        try:           
            # Extract time parameters from simulation (could hard-code to avoid unnecessary repetition)
            sim_file = os.path.abspath(cellmodeller_module)
            dt = sim_time_parameters(sim_file)
            
            # Defining input/output locations
            folder_id = str(uuid.uuid4())
            export_path = "data/" + folder_id
            #export_path = folder_id
            
            # Send data to file that CellModeller module can read
            os.makedirs(f'scene_io/{folder_id}')
            with open(f'scene_io/{folder_id}/scene.txt',"w") as file:
                file.write('%d' % scene) 
                file.close()
                            
            # Run CellModeller simulation
            start = time.time()
            #sys.stdout = open(os.devnull, 'w') # Disable printing from simulations
            sim = simulate(cellmodeller_module, parameters, export_path, max_cells=None, max_time=sim_time)
            #sys.stdout = sys.__stdout__ # Re-enable printing
            print(f'{cellmodeller_module} at {datetime.datetime.now()} complete. Wall time: {(time.time() - start)/60} min' )
                            
            # Calculate summary stats
            population_df = hgt_intermixing.main(file_dir_path=export_path, dt=dt, img_freq=imaging_frequency)
            
            # Return summary statistics
            df_calcs(population_df)    
            population_df['scene'] = scene
            """
            summary_stats = {"intermixing_score": population_df["intermixing_score"].to_numpy(), 
                    'event_counts': population_df['event_counts'].to_numpy(),
                    'donor_frac': population_df['donor_frac'].to_numpy(), 
                    'recip_frac': population_df['recip_frac'].to_numpy(), 
                    'trans_frac': population_df['trans_frac'].to_numpy(),
                    "scene": cellmodeller_module,
                    'time': time_arr}
            """                    
            del sim 
            break
                 
        except:
            population_df = pd.DataFrame()
            print('RETRY!')
        
    return population_df

def simulate(modfilename, params, export_path, max_cells=None, max_time=None):
    """
    Same as batch.py with modificationsto allow external parameters and termination based on stepNum
    
    @param modfilename  module name (e.g., "Tutorial1.py")
    @param export_path  output directory name
    @param max_cells    number of cells where simulation terminates (int)
    @param max_time     simulation time where simulation terminates (float)
    """
    # Get module file name and append module to system path
    (path,name) = os.path.split(modfilename)
    modname = str(name).split('.')[0]
    sys.path.append(path)
    
    # Extract time parameters from simulation
    sim_file = os.path.abspath(modfilename)
    dt = sim_time_parameters(sim_file)
    
    # Create Simulator object
    sim = Simulator(modname, dt, saveOutput=True, outputDirName=export_path, psweep=True, params=params)
    
    # Run simulation and terminate when simulation reaches a certain number of cells
    if max_cells:
        while len(sim.cellStates) <= max_cells:
            sim.step() # Could use try/except to delete sims if necessary
    if max_time:
        while sim.stepNum*dt <= max_time:
            sim.step()  
        
def sim_time_parameters(file_path):
    """
    Extract the simulation time step from the module file. 
    Must be written as dt = X.Y and sim_time = A.B in the module file
    
    @param  file_path   str of file name containing parameters to be modified
    """
    search_dt = f'dt = (\d+\.\d+)'
    search_sim_time = f'sim_time = (\d+\.\d+)'
    with open(file_path, "r+") as file:
        file_contents = file.read()
        dt = re.findall(search_dt, file_contents)
        sim_time = re.findall(search_sim_time, file_contents)
    return float(dt[0])
    
def df_calcs(df, experiment=False):
    if experiment==False:
        correction = 0
    else:
        correction = 5/60
        
    df['Time (h)'] = df['Time (h)'] - correction
    df['donor_frac'] = df['donors']/(df['donors'] + df['recips'] + df['trans'])
    df['recip_frac'] = df['recips']/(df['donors'] + df['recips'] + df['trans'])
    df['trans_frac'] = df['trans']/(df['donors'] + df['recips'] + df['trans'])
    df['intermixing_score'] = 1 - df['contagion']
    
def write_to_pickle(data, file_name):
    # Open a file, where you want to store the data
    file = open(file_name, 'wb')

    # Dump information to that file
    pickle.dump(data, file)

    # Close the file
    file.close()
    
def run_models(module, scene, params):
    #local_df = pd.DataFrame()
    #local_counter = 0

    summary_stat_df = run_model(module, scene, params)
    
    '''
    # Terminal output
    local_counter += 1
    print(f'{local_counter} completed for scene {scene}')
    '''

    # Store in df
    #local_df = pd.concat([local_df, summary_stat_df], ignore_index=True)
    
    return summary_stat_df

# Make sure we are running as a script
if __name__ == "__main__": 
    multiprocessing.set_start_method('spawn') # Required to make CM work with multiprocessing
    cellmodeller_module = "conjugation_abc.py"
    scenes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18]
      
    parameters = {'conj_freq': 0.5142786401713151,
                  'delay': 0.6288688280654475,
                  'm_max': 3.0371970337762533,
                  't_half': 0.5001704428814664,
                  'death_rate_donor': 0.31819885071066445,
                  'death_rate_trans': 0.26802855404172576,
                  'death_rate_recip': 0.06508671512237757
                  }
      
    # Initialize storage
    df = pd.DataFrame()  

    # Using ThreadPoolExecutor or ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(run_models, cellmodeller_module, scene, parameters) for scene in scenes]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            df = pd.concat([df, result], ignore_index=True)  

    df.to_csv('abc_validation.csv', index=False)      
