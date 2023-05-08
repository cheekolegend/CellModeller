import os
import sys
import subprocess
import string
import shutil
import re
from CellModeller.Simulator import Simulator

"""
Instructions:
1. Specify n_simulations
2. From terimnal, run multi_batch_dt.py and give the module name as an argument

Optional: To run multiple simulations in parallel, just run the command again in a new terminal.
"""

n_simulations = 5

def simulate(modfilename):
    (path,name) = os.path.split(modfilename)
    modname = str(name).split('.')[0]
    sys.path.append(path)
    
    # Extract time parameters from simulation
    sim_file = os.path.abspath(modfilename)
    (dt, sim_time) = sim_time_parameters(sim_file)

    sim = Simulator(modname, dt, saveOutput = True)
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
    
def main():
    # Get module name to load
    if len(sys.argv)<2:
        print("Please specify a model (.py) file")
        exit(0)
    else:
        moduleName = sys.argv[1]

    for i in range(n_simulations):
        simulate(moduleName)

# Make sure we are running as a script
if __name__ == "__main__": 
    main()
