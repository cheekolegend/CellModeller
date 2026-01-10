import os
import sys
import pyabc
from pyabc.visualization import plot_data_callback
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add all subdirectories from LineageAnalysis to path
root_directory = os.path.join(os.path.expanduser('~'), 'LineageAnalysis')
for dirpath, dirnames, filenames in os.walk(root_directory):
    for dirname in dirnames:
        sys.path.append(os.path.join(dirpath, dirname))

# Imports from LineageAnalysis package
import matingpairs_sims 

plt.rcParams['font.size'] = 12
#colors = ['b','grey','r','chocolate','m','gold']
plt.style.use('tableau-colorblind10')

def ABCsimulation(params):
    return None

### 1-D KDE ###
def plot_1d(history, limits, key, xlabel=None, **kwargs):
    """
    Plot posterior distributions of an ABC run
    @param history  ABC database
    @param limits   dict containing upper/lower bounds of priors
    @param exact    Exact value of the observed data
    @param key      Key in exact dict
    
    The figure is saved in the same directory as this script
    """

    xmin = limits[key][0]
    xmax = limits[key][1]
    
    fig = plt.figure(figsize=(4,3))
    #fig = plt.figure(figsize=(4,6)) # for creating the legend
   
    #Plot prior distributions
    y = 1/(xmax-xmin) # uniform prior is equal to 1/max(param) for each parameter 
    plt.plot(np.linspace(xmin,xmax,100), y*np.ones(100,), linestyle='--', color='k', lw=3, label='Prior')
    
    ax = plt.gca()
    c = 0
    for t in range(0, history.n_populations, 1):
    #for t in range(0, 4, 1):
        df, w = history.get_distribution(t=t)
        pyabc.visualization.plot_kde_1d(
            df, w, 
            x=key, ax=ax,
            xmin=xmin,
            xmax=xmax,              
            numx=10000,
            label="Round {}".format(t+1),
            lw=3, 
            #color=colors[c],
            )  
                          
        ax.set_ylabel("") 
        c+=1
        
    if xlabel:
        plt.xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_ylim(bottom=0)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, prop={'size': 12}) # for creating legend          
    plt.savefig("%s_1D_KDE-t=%s.svg" % (key,t), bbox_inches='tight', dpi=300)
    
def plot_epsilons(history):
    fig = plt.figure(figsize=(4,3))
    eps = np.array(history.get_all_populations()['epsilon'][1:])
    x = np.arange(1, history.max_t + 2)
    #eps = np.array(history.get_all_populations()['epsilon'][1:5])
    #x = np.arange(1, 5)
    ax = plt.gca()
    ax.plot(x, eps, 'ko-')
    ax.set_xlabel("Calibration Round")
    ax.set_ylabel("Error")
    ax.set_ylim(bottom=0)
    plt.xticks(x)
    fig.set_tight_layout(True)
    plt.savefig("epsilons.svg")
    
def plot_data_populations(sum_stat, weight, ax, **kwargs):
    """Plot a single trajectory"""
    # sum_stat_key is the name of the summary statistic I want to plot
    ax.plot(measurement_times, sum_stat['donor_frac'], color='red', alpha=0.2)
    ax.plot(measurement_times, sum_stat['recip_frac'], color='green', alpha=0.2)
    ax.plot(measurement_times, sum_stat['trans_frac'], color='orange', alpha=0.2)
    
def plot_mean_populations(sum_stats, weights, ax, **kwargs):
    """Plot mean over all samples"""
    weights = np.array(weights)
    weights.fill(1) # Include this to take the unweighted sample mean
    weights /= weights.sum()
    data_donor= np.array([sum_stat['donor_frac'] for sum_stat in sum_stats])
    mean_donor = (data_donor*weights.reshape((-1, 1))).sum(axis=0)
    data_recip = np.array([sum_stat['recip_frac'] for sum_stat in sum_stats])
    mean_recip = (data_recip*weights.reshape((-1, 1))).sum(axis=0)
    data_trans = np.array([sum_stat['trans_frac'] for sum_stat in sum_stats])
    mean_trans = (data_trans*weights.reshape((-1, 1))).sum(axis=0)
    ax.plot(measurement_times, mean_donor, color='red', linestyle='dashed', label='Sim. Donors')
    ax.plot(measurement_times, mean_recip, color='green', linestyle='dashed', label='Sim. Recipients')
    ax.plot(measurement_times, mean_trans, color='orange', linestyle='dashed', label='Sim. Transconjugants')
    
def plot_mean_intermixing(sum_stats, weights, ax, **kwargs):
    """Plot mean over all samples"""
    weights = np.array(weights)
    weights.fill(1) # Include this to take the unweighted sample mean
    weights /= weights.sum()
    data = np.array([sum_stat['intermixing_score'] for sum_stat in sum_stats])
    mean = (data*weights.reshape((-1, 1))).sum(axis=0)
    ax.plot(measurement_times, mean, color='k', linestyle='dashed', label='Simulation Mean')
    
def plot_data_intermixing(sum_stat, weight, ax, **kwargs):
    """Plot a single trajectory"""
    # sum_stat_key is the name of the summary statistic I want to plot
    ax.plot(measurement_times, sum_stat['intermixing_score'], color='grey', alpha=0.2)
    
def plot_accepted_runs(history, ylabel, plot_data_func, plot_mean_func, plot_stat=None, t=None):
    plt.rcParams['font.size'] = 32
    fig = plt.figure(figsize=(9,11))
    _, ax = plt.subplots(figsize=(9,11))
    
    if t == None:
        t = history.max_t
    
    ax = plot_data_callback(history, 
                            plot_data_func, 
                            plot_mean_func, 
                            ax=ax,
                            t=t
                            )
    
    # plot experimental data
    if plot_stat == 'intermixing':
        plt.plot(measurement_times, df_mean['intermixing_score'].to_numpy(), color='k', label='Experiment Mean') 
    elif plot_stat == 'populations':
        plt.plot(measurement_times, df_mean['donor_frac'].to_numpy(), color='red', label='Exp. Donors') 
        plt.plot(measurement_times, df_mean['recip_frac'].to_numpy(), color='green', label='Exp. Recipients') 
        plt.plot(measurement_times, df_mean['trans_frac'].to_numpy(), color='orange', label='Exp. Transconjugants') 
    
    #plt.scatter(measurement_times, measurements, color='k', label='Experiment mean')
    plt.xlabel('Time (h)')
    plt.ylabel(ylabel)
    plt.legend(loc='upper left', ncol=1, prop={'size': 20})
    plt.savefig(f"{plot_stat}.png", bbox_inches='tight')
    
def plot_violins(mp_stats_exp, mp_stats_sim, cols=['donor_length_frac','donor_growth_rate','overlap_area','angle','max_contact_time']):
    plt.rcParams['font.size'] = 16
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    axs = axs.ravel()
    
    #ylims = [[0.4, 1.1], [-0.5, 2], [0,10], [0, 3.14/2], [0,10]]
    ylims = [[0,10]] 
    
    '''         
    xlabels = ['Donor Cell Cycle State', 
               'Donor Growth Rate (1/h)',
               'Overlap Area (μm²)', 'Angle (Radians)', 'Max. Contact Time (Frames)']
    '''
    xlabels = ['Max. Contact Time (Frames)']
    
    for i, col in enumerate(cols): 
        array1 = mp_stats_exp[col]
        array2 = mp_stats_sim[col]

        df1 = pd.DataFrame({'value': array1})
        df1['type'] = 'Experiment'
        df1['variable'] = col
        df2 = pd.DataFrame({'value': array2})
        df2['type'] = 'Simulation'
        df2['variable'] = col
        df = pd.concat([df1, df2])
        
        legend=False
        if i == 0:
            legend=True
        sns.violinplot(data=df, x='variable', y='value', hue='type', split=True, inner='quart', dodge=True, linewidth=1.5, ax = axs[i], legend=legend, bw_adjust=0.9)
        
        if i == 0:
            sns.move_legend(axs[i], "upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False, title=None)  
        
        axs[i].set_ylim(bottom=ylims[i][0], top=ylims[i][1])
        axs[i].set_xticklabels([xlabels[i]])
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')
    
    # Number of samples
    n_exp = len(array1)
    n_sim = len(array2)   
    fig.suptitle(f'n_exp = {n_exp}, n_sim = {n_sim}') 

    plt.savefig("violins.png", bbox_inches='tight', dpi=300)

if __name__ == '__main__':   
    # Create "Null" ABCSMC object that has the corresponding parameter priors used in ABC run 
    lower_bound = 0.02
    scale = 0.2
    prior = pyabc.Distribution(mu=pyabc.RV("uniform", lower_bound, scale))
    abc = pyabc.ABCSMC(ABCsimulation, prior)

    # Load database
    db_path = ("sqlite:///" + "results.db")
    run_id = 1
    history = abc.load(db_path, run_id)
            
    # Define prior distribution for each parameter [lb, ub]
    limits = {'conj_freq': [0, 1], 
                    'delay': [0, 2],
                    'm_max': [1, 5],
                    't_half': [1/60, 1],
                    'death_rate_donor': [0, 0.5],
                    'death_rate_trans': [0, 0.5],
                    'death_rate_recip': [0, 0.2],
                    }
    
    
    ### 1-D KDE ###
    xlabels = {'conj_freq': 'Transfer Gene Activation Frequency (h$^{-1}$)', 
                    'delay': 'Transconjugant Maturation Time (h)',
                    'm_max': 'Transitory Derepression Multiplier',
                    't_half': 'Transitory Derepression Half-life (h)',
                    'death_rate_donor': 'Donor Death Rate (h$^{-1}$)',
                    'death_rate_trans': 'Transconjugant Death Rate (h$^{-1}$)',
                    'death_rate_recip': 'Recipient Death Rate (h$^{-1}$)',
                    }
    for key in limits.keys():
        plot_1d(history, limits, key, xlabel=xlabels[key])

    ### Epsilons ###
    plot_epsilons(history)
    
    ### Statistics for parameters ###
    #df, w = history.get_distribution(t=3)
    df, w = history.get_distribution() # Stats for final calibration round
    for key in limits.keys():
        array = df[key].to_numpy()
        print(f'Mean {key}: {np.average(array, weights=w)}')
        print(f'Std {key}: {df[key].std()}')

    '''
    ### Accepted runs ###
    measurement_times = sorted(df['Time (h)'].unique())
    plot_accepted_runs(history, 'Intermixing Score', plot_data_intermixing, plot_mean_intermixing, plot_stat='intermixing')
    plot_accepted_runs(history, 'Population Fraction', plot_data_populations, plot_mean_populations, plot_stat='populations')
    #plot_violins(exp_mp_stats, sim_mp_stats, cols=mp_stats_list)
    '''
    
    '''
    _, ax = plt.subplots()
    ax = plot_data_callback(history, 
                            plot_data, 
                            plot_mean, 
                            ax=ax
                            )
    plt.scatter(measurement_times, measurements, color="C1", label='Experiment')
    plt.xlabel('Time (h)')
    plt.ylabel('Conjugation Efficiency [T/(T + R)]')
    plt.title('')
    plt.legend()
    plt.show()
    '''
        
    
    
