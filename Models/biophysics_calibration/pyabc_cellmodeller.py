"""
This is the main script that runs pyabc and CellModeller. 

Instructions:
1. Set cellmodeller_module to the name of the simulation module (e.g., Tutorial1a.py)
2. Set max_cells to the number of cells where the simulation will end
3. Set the summary statistics to calculate in the model() function
4. Define the ABC calibration settings in the main function
5. Set experimental values for summary statistics in the main function;
    ensure the chosen summary statistics are the same as the simulated ones
6. Define prior distributions for each parameter in the main function
7. Run by entering the command: python pyabc_cellmodeller.py
"""

# Standard python modules
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
import uuid

# Installed modules
from CellModeller.Simulator import Simulator
import pyabc

# Local/custom modules
sys.path.append(os.path.join(os.path.expanduser('~'), 'CellModeller/Scripts/summaryStatistics/biophysics/'))
import anisotropy
import aspect_ratio
import density_calculation
import growth_rate_analysis
sys.path.append(os.path.join(os.path.expanduser('~'), 'CellModeller/Scripts/summaryStatistics/'))
import helperFunctions

cellmodeller_module = "simulation_module.py"  
max_cells = 1000

def model(parameters):
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
    summary_stats['growth_rate_vs_centroid'] = growth_rate_analysis.main(cells, dt)
    summary_stats['density_parameter'] = density_calculation.main(cells)
    summary_stats['order_parameter'] = anisotropy.main(cells)
    summary_stats['aspect_ratio'] = aspect_ratio.main(cells)
    
    # Write summary stats to file for convenience (optional) 
    file = open(export_path + "/summary_stats.txt","w")
    for key, value in summary_stats.items(): 
        file.write('%s: %.3f\n' % (key, value)) 
    file.close()

    return summary_stats

def simulate(modfilename, params, export_path):
    """
    Same as batch.py with modificationsto allow external parameters and termination based on stepNum
    
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
       
def distance_calculation(sim_stats, exp_stats):
    """
    Calculate Euclidean distance between simulations and experimental observations
    
    @param  sim_stats   Dict of simulated summary statistics
    @param  exp_stats   Dict of experimental summary statistics
    @return distance    Euclidean distance between simuluations and experiment
    """
    distance_list = []
    for x in sim_stats.keys():
        diff = np.abs(sim_stats[x] - exp_stats[x])
        distance_list.append(diff)

    distance = np.linalg.norm(distance_list)

    return distance    
    
if __name__ == '__main__':
    # Define ABC-SMC settings
    n_cores = 8
    population_size = 10
    min_epsilon = 0.05
    max_populations = 10

    # Define experimental data
    exp_summary_stats = {}
    exp_summary_stats['growth_rate_vs_centroid'] = 0.05
    exp_summary_stats['density_parameter'] = 0.8
    exp_summary_stats['order_parameter'] = 0.5
    exp_summary_stats['aspect_ratio'] = 0.7
    
    # Define prior distribution for each parameter [lb, ub]
    param_config = {'gamma': [1, 100], 'reg_param': [1e-2, 1]}
    
    prior_distributions = {}
    for parameter_name in param_config.keys():
        param_low = param_config[parameter_name][0]
        param_hi = param_config[parameter_name][1]
        width = abs(param_hi - param_low)

        # generate dictionary containing keywords for pyabc Distribution() class
        prior_distributions[parameter_name] = {"type": "uniform", "args": (param_low, width), "kwargs": {}}

    # create instance of `Distribution` class
    prior = pyabc.Distribution()
    prior = prior.from_dictionary_of_dictionaries(prior_distributions)
    
    # Create abc model
    abc = pyabc.ABCSMC(model, 
                       prior, 
                       distance_function=distance_calculation, 
                       population_size=population_size, 
                       sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_cores))
    
    # Create database                   
    db_path = "results.db"
    abc.new("sqlite:///" + db_path, exp_summary_stats)
    
    # Run
    history = abc.run(minimum_epsilon=min_epsilon, 
                      max_nr_populations=max_populations) 
                      
