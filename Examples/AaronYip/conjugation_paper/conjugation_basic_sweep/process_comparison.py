import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind, tukey_hsd
import matplotlib.pyplot as plt
from itertools import combinations
from postprocess_scripts import hgt_intermixing
from run_comparison import df_calcs

def cellmodeller_to_summarystats(data_dir, simulation_modules, dt, csv_name):
    '''
    Create .csv if it wasn't created successfully and load summary stats into a DataFrame

    data_dir    directory containing subdirectories with simulation data
    simulation_modules  list of different model names tested; must match cellmodeller_module name
    dt          time step in hours
    csv_name    name of the outputted csv file  
    '''
    if os.path.isfile(csv_name):
        master_df = pd.read_csv(csv_name)
    else:
        master_df = pd.DataFrame()
        
        # Write down time for output
        time_arr = list(np.arange(0, sim_time + 5/60/2, 5/60)) # 5/60 is the imaging frequency of the experiment (h)
        if time_arr[-1] > sim_time:
            time_arr.pop(-1)
        
        # Load all subfolders in "data_dir"
        subdirs = [os.path.join(data_dir, subdir) for subdir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subdir))]

        # [Start loop; go through each simulation]
        for i, subdir in enumerate(subdirs):
            print(f"Processing folder {i}/{len(subdirs)}")
            # Loop through all files in the current subdirectory
            for file_name in os.listdir(subdir):
                # Check if the file name matches any of the simulation modules
                for module in simulation_modules:
                    if module == file_name:
                        cellmodeller_module = module
                        break

            # Calculate summary stats
            print(subdir)
            population_df = hgt_intermixing.main(file_dir_path=subdir, dt=dt)
            population_df['scene'] = cellmodeller_module
            
            # Return summary statistics
            df_calcs(population_df)    
            
            # Store in df
            new_row = pd.DataFrame(population_df)
            master_df = pd.concat([master_df, new_row], ignore_index=True)
        
        master_df.to_csv(csv_name, index=False) 
    
    return master_df    
    
def ttest_matrix(df):
    # Get the maximum 'time' value for each 'scene' group to identify the end-point
    max_time_df = df.groupby('scene')['Time (h)'].max().reset_index()

    # Merge the max_time_df with the original df to filter out the rows corresponding to the end-point
    end_point_df = pd.merge(df, max_time_df, on=['scene', 'Time (h)'])

    # Step 2: Extract the 'intermixing_score' values for each group
    groups = end_point_df.groupby('scene')['intermixing_score']

    # Extract unique group names
    group_names = groups.groups.keys()

    # Step 3: Perform pairwise t-tests
    results = {}
    for group1, group2 in combinations(group_names, 2):
        group1_scores = end_point_df[end_point_df['scene'] == group1]['intermixing_score']
        group2_scores = end_point_df[end_point_df['scene'] == group2]['intermixing_score']
        
        t_stat, p_value = ttest_ind(group1_scores, group2_scores)
        results[(group1, group2)] = p_value

    # Step 4: Apply Bonferroni correction
    num_comparisons = len(results)
    bonferroni_alpha = 0.05 / num_comparisons

    # Step 5: Display the results with Bonferroni correction
    print("bonferroni alpha: ", bonferroni_alpha)
    for pair, p_value in results.items():
        print(f"Comparison: {pair[0]} vs {pair[1]}")
        print(f"p-value: {p_value:.4f}")
        if p_value < bonferroni_alpha:
            print("Result: Significant difference (reject H0)")
        else:
            print("Result: No significant difference (fail to reject H0)")
        print("-" * 40)
        
def run_tukey_endpoints(df):
    # Get the maximum 'time' value for each 'scene' group to identify the end-point
    max_time_df = df.groupby('scene')['Time (h)'].max().reset_index()

    # Merge the max_time_df with the original df to filter out the rows corresponding to the end-point
    end_point_df = pd.merge(df, max_time_df, on=['scene', 'Time (h)'])
    
    stats = ['intermixing_score', 'conj_efficiency', 'conj_per_contact_cumsum']
    for stat in stats:
        groups = end_point_df.groupby('scene')[stat]
        group_array = []
        name_array = []
        
        for name, group in groups:
            group_array.append(group.to_numpy())
            name_array.append(name)

        # Perform Tukey's HSD test; number of groups corresponds to number of models tested
        group0 = group_array[0]
        group1 = group_array[1]
        group2 = group_array[2]
        group3 = group_array[3]
        group4 = group_array[4]
        group5 = group_array[5]
        tukey_results = tukey_hsd(group0,group1,group2,group3,group4, group5)

        # Display the results
        print('Names in ascending order are: ', name_array)
        print('Results for ', stat)
        print(tukey_results)    
        
        # Box plots
        fig, ax = plt.subplots(1, 1)
        ax.boxplot([group0,group1,group2,group3,group4,group5])
        ax.set_xticklabels(["Model 0", "Model 1", "Model 2", "Model 3", "Model 4", "Model 5"]) 
        ax.set_ylabel(stat) 
        plt.show()
    
if __name__ == "__main__": 
    data_dir = 'data'
    simulation_modules = ['conjugation_basic', 'conjugation_basic_death', 'conjugation_delay', 'conjugation_contact', 'conjugation_derepression', 'conjugation_kernel', ]
    dt = 0.008333333 # 0.5/60 h
    sim_time = 8.0 #h
    csv_name = 'model_comparison.csv'
    df = cellmodeller_to_summarystats(data_dir, simulation_modules, dt, csv_name)
    
    # Summary stats to compare: [conj_per_contact_per_time, conj_efficiency, intermixing_score]
    #df['conj_per_contact_per_time'] = df['event_counts']/df['donor_trans_cts']/dt
    #df['conj_efficiency'] = df['trans']/df['donor_trans_cts']
        
    # Get the maximum 'time' value for each 'scene' group to identify the end-point
    max_time_df = df.groupby('scene')['Time (h)'].max().reset_index()

    # Merge the max_time_df with the original df to filter out the rows corresponding to the end-point
    end_point_df = pd.merge(df, max_time_df, on=['scene', 'Time (h)'])

    # Step 2: Extract the 'intermixing_score' values for each group
    group1 = end_point_df[end_point_df['scene'] == 'conjugation_basic']['intermixing_score']
    group2 = end_point_df[end_point_df['scene'] == 'conjugation_contact']['intermixing_score']
    group3 = end_point_df[end_point_df['scene'] == 'conjugation_derepression']['intermixing_score']
    group4 = end_point_df[end_point_df['scene'] == 'conjugation_kernel']['intermixing_score']

    print(group1)

    # Step 3: Perform ANOVA
    f_statistic, p_value = f_oneway(group1, group2, group3, group4)

    # Step 4: Display the results
    print("F-statistic:", f_statistic)
    print("p-value:", p_value)

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print("The means of the groups are significantly different (reject H0).")
    else:
        print("The means of the groups are not significantly different (fail to reject H0).")
    
    # Run t-test matrix    
    run_tukey_endpoints(df)
    
    # Make box plots of each summary stat
    
    
    
