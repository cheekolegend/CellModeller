import os
import sys
import re
import uuid
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pyabc
from CellModeller.Simulator import Simulator

cellmodeller_module = "Tutorial_1a.py"    

def model(parameters):
    export_path = "data/" + str(uuid.uuid4())
    sys.stdout = open(os.devnull, 'w') # Disable printing from simulations
    simulate(cellmodeller_module, parameters, export_path)
    sys.stdout = sys.__stdout__ # Re-enable printing
    print("Simulation completed")
    return {"data": parameters["mu"]}
   
def distance(x, x0):
    return abs(x["data"] - x0["data"])

def simulate(modfilename, params, export_path):
    """
    Same as batch.py with modificationsto allow external parameters and termination based on stepNum
    
    @param modfilename  module name (e.g., "Tutorial1.py")
    @param export_path  output directory name
    @param params       dictionary of parameter values passed to the simulation
    """
    (path,name) = os.path.split(modfilename)
    modname = str(name).split('.')[0]
    sys.path.append(path)
    
    # Extract time parameters from simulation
    sim_file = os.path.abspath(modfilename)
    (dt, sim_time) = sim_time_parameters(sim_file)
    
    sim = Simulator(modname, dt, saveOutput=True, outputDirName=export_path, psweep=True, params=params)
    while sim.stepNum*dt <= sim_time:
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
    return float(dt[0]), float(sim_time[0])
    
if __name__ == '__main__':
    # Define prior distribution for each parameter
    lower_bound = 0
    scale = 1
    prior = pyabc.Distribution(mu=pyabc.RV("uniform", lower_bound, scale))
    
    # Define ABC-SMC settings
    n_cores = 8  
    abc = pyabc.ABCSMC(model, 
                       prior, 
                       distance, 
                       population_size=10, 
                       sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_cores))
    db_path = "results.db"
    observation = 0.8
    abc.new("sqlite:///" + db_path, {"data": observation})
    
    # Run
    
    history = abc.run(minimum_epsilon=0.05, 
                      max_nr_populations=10) 
