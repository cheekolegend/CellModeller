import os
import sys
import re
import uuid
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabc
import random
import time
import datetime
import warnings
from CellModeller.Simulator import Simulator
from Scripts.hgt_scripts import hgt_intermixing
from pyabc.sampler import RedisEvalParallelSampler
from pyabc.sampler import RedisEvalParallelSamplerServerStarter

# Add all subdirectories from LineageAnalysis to path
root_directory = os.path.join(os.path.expanduser('~'), 'LineageAnalysis')
for dirpath, dirnames, filenames in os.walk(root_directory):
    for dirname in dirnames:
        sys.path.append(os.path.join(dirpath, dirname))

# Imports from LineageAnalysis package
import matingpairs_sims      

# Folds for cross-validation
fold1 = [1,2,3,]
fold2 = [4,5,6]
fold3 = [7,8,9]
fold4 = [10,11,12,13,]
fold5 = [14,16,17,18]

max_sim_time = 8
scenes = fold5 + fold1 + fold2 + fold3
imaging_frequency = 5/60
sim_times = np.ones(17)*max_sim_time#[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] 
pickle_interval = 5
scene_idxs = list(range(len(scenes)))
cellmodeller_module = "conjugation_abc.py"
np.seterr(all="ignore") #suppress warnings for cleaner output
warnings.filterwarnings("ignore") # ignore all warnings
n_attempts = 3
carrying_capacity = 1100 # max cells in the trap

def model(parameters):
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
            # Select scene to calibrate to
            i = random.choice(scene_idxs)
            scene = scenes[i]
            sim_time = sim_times[i]
            
            # Extract time parameters from simulation (could hard-code to avoid unnecessary repetition)
            sim_file = os.path.abspath(cellmodeller_module)
            dt = sim_time_parameters(sim_file)
            
            # Defining input/output locations
            folder_id = str(uuid.uuid4())
            export_path = "data/" + folder_id
            
            # Send data to file that CellModeller module can read
            os.makedirs(f'scene_io/{folder_id}')
            with open(f'scene_io/{folder_id}/scene.txt',"w") as file:
                file.write('%d' % scene) 
                file.close()
                
            # Run CellModeller simulation
            start = time.time()
            sys.stdout = open(os.devnull, 'w') # Disable printing from simulations
            sim = simulate(cellmodeller_module, parameters, export_path, max_cells=None, max_time=sim_time)
            sys.stdout = sys.__stdout__ # Re-enable printing
            print(f'{parameters} at {datetime.datetime.now()} complete. Wall time: {(time.time() - start)/60} min' )
            
            with open(f'{export_path}/scene.txt',"w") as file:
                file.write('%d' % scene) 
                file.close()

            #print(f'Simulation failed for parameters: {parameters}')
            #shututil.rmtree(export_path) 
                
            # Calculate summary stats
            population_df = hgt_intermixing.main(file_dir_path=export_path, dt=dt, img_freq=imaging_frequency)
            '''
            # Write summary stats to file for convenience (optional) 
            summary_stats_export = {'aspect_ratio': aspect_ratio,
                                    'anisotropy': anisotropy,
                                    'density': density,
                                    'growth_rate_exp_deviation': growth_rate_exp_deviation,
                                    'convexity': convexity,
                                    'growth_rate': growth_rate}
            file = open(export_path + "/summary_stats.txt","w")
            for key, value in summary_stats_export.items(): 
                file.write('%s: %.3f\n' % (key, value)) 
            file.close()
            '''
            
            # Return summary statistics
            df_calcs(population_df)    
            summary_stats = {"intermixing_score": population_df["intermixing_score"].to_numpy(), 
                    'recip_frac': population_df['recip_frac'].to_numpy(),
                    'donor_frac': population_df['donor_frac'].to_numpy(),
                    'trans_frac': population_df['trans_frac'].to_numpy(),
                    #'conj_eff_pop': population_df['conj_eff_pop'].to_numpy(),
                    'conj_prob': population_df['conj_prob'].to_numpy(),
                    'cells': population_df['cells'].to_numpy(),
                    "scene": scene}
                               
            del sim 
            break
          
        except:
            print('RETRY!')
            summary_stats = {"intermixing_score": np.zeros(96), 
                    'donor_frac': np.zeros(96), 'recip_frac': np.zeros(96), 'trans_frac': np.zeros(96), 'conj_prob': np.zeros(96), 'cells': np.zeros(96),
                    "scene": scene}

    return summary_stats

def populationDistance(x, x0):
    """
    Sum of squared errors for each population fraction.
    x = simulation data
    x0 = experiment data
    """
    exp_data = x0["population_data"][x0["population_data"]["experiment_id"] == "scene" + str(x["scene"])]
    distance = np.sum([(x['recip_frac'][-1] - exp_data["recip_frac"].to_numpy()[-1])**2,
                       (x['donor_frac'][-1] - exp_data["donor_frac"].to_numpy()[-1])**2,
                       (x['trans_frac'][-1] - exp_data["trans_frac"].to_numpy()[-1])**2])
                       
    return distance
    
def intermixingDistance(x, x0):
    """
    Sum of squared errors for intermixing score
    x = simulation data
    x0 = experiment data
    """
    exp_data = x0["population_data"][x0["population_data"]["experiment_id"] == "scene" + str(x["scene"])]
    distance = np.sum((x["intermixing_score"][-1] - exp_data["intermixing_score"].to_numpy()[-1])**2)
    
    return distance
    
def conjEfficiencyDistance(x, x0):
    exp_data = x0["population_data"][x0["population_data"]["experiment_id"] == "scene" + str(x["scene"])]
    distance = np.sum((x["conj_prob"][-1] - exp_data["conj_prob"].to_numpy()[-1])**2)
    
    return distance
    
def cellsDistance(x, x0):
    exp_data = x0["population_data"][x0["population_data"]["experiment_id"] == "scene" + str(x["scene"])]
    distance = np.sum(((x["cells"][-1] - exp_data["cells"].to_numpy()[-1])/carrying_capacity)**2)
    
    return distance

def simulate(modfilename, params, export_path, max_cells=None, max_time=None):
    """
    Same as batch.py with modificationsto allow external parameters and termination based on stepNum
    
    @param modfilename  module name (e.g., "Tutorial1.py")
    @param export_path  output directory name
    @param params       dictionary of parameter values passed to the simulation
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
    df['intermixing_score'] = 1 - df['contagion']
    df['conj_eff_pop'] = df['trans']/(df['recips'] + df['trans'])
    df['recip_frac'] = df['recips']/(df['recips'] + df['donors'] + df['trans'])
    df['donor_frac'] = df['donors']/(df['recips'] + df['donors'] + df['trans'])
    df['trans_frac'] = df['trans']/(df['recips'] + df['donors'] + df['trans'])
    df['cells'] = df['donors'] + df['recips'] + df['trans']
    
if __name__ == '__main__':
    dt = sim_time_parameters(os.path.abspath(cellmodeller_module))
    
    # Read in experimental data
    df = pd.read_csv('conjugation_population_analysis.csv')   
    df_calcs(df, experiment=True)
    df = df[df['Time (h)'] <= max_sim_time]
    
    # Define experimental data to calibrate to
    observation = {'population_data': df}
        
    # Define prior distribution for each parameter [lb, ub]
    param_config = {'conj_freq': [0, 1], 
                    'delay': [0, 1.5],
                    'm_max': [1, 5],
                    't_half': [1/60, 1],
                    'death_rate_donor': [0, 0.5],
                    'death_rate_trans': [0, 0.5],
                    'death_rate_recip': [0, 0.2]
                    }
    
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
    
    # Define ABC-SMC settings
    n_cores = 8  
    #initial_epsilon = 0.4   #epsilon for first population
    #median_multiplier = 1.0 #default
    population_size = 40
    
    #distance = pyabc.AdaptiveAggregatedDistance([populationDistance, intermixingDistance, conjEfficiencyDistance]) # Requires calibration sample
    distance = pyabc.AggregatedDistance([populationDistance, intermixingDistance, conjEfficiencyDistance, cellsDistance]) 
    
    #redis_sampler = RedisEvalParallelSampler(host="111.111.111.111", port=6379, password="conjugation_is_awesome", look_ahead=True, max_n_eval_look_ahead_factor=4, log_file='redis_log.csv')
    redis_sampler = RedisEvalParallelSamplerServerStarter(password="conjugation_is_awesome", look_ahead=True, max_n_eval_look_ahead_factor=4, workers=pyabc.nr_cores_available(), log_file='redis_log.csv')
    
    #initial_epsilon = 9.18210405 # From running 8 sims from priors
    abc = pyabc.ABCSMC(model, 
                       prior, 
                       distance, 
                       population_size=population_size, 
                       #eps=pyabc.QuantileEpsilon(alpha=0.5, weighted=True),
                       #eps=pyabc.QuantileEpsilon(initial_epsilon=9.18210405, alpha=0.5, weighted=True),
                       eps=pyabc.epsilon.MedianEpsilon(),
                       sampler=redis_sampler,                       
                       #sampler=pyabc.sampler.MulticoreParticleParallelSampler(n_procs=n_cores)
                       #sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_cores)
                       #sampler=pyabc.sampler.SingleCoreSampler()
                       )
    db_path = "results.db"
    abc.new("sqlite:///" + db_path, observation)
    #abc.load("sqlite:///" + db_path, abc_id=0, observed_sum_stat=observation)
    
    # Run
    history = abc.run(minimum_epsilon=0.001, 
                      max_nr_populations=3)
                      
    # Output parameter values                  
    df, w = history.get_distribution()
    with open("stats_output.txt", "w") as f:
        for key in param_config.keys():
            array = df[key].to_numpy()
            f.write(f'Mean {key}: {np.average(array, weights=w)}\n')
            f.write(f'Std {key}: {df[key].std()}\n')
