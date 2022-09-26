# Standard modules
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import sys

# Installed modules
import CellModeller
from kneed import KneeLocator
from sklearn.linear_model import LinearRegression

# Local/custom modules
from lje.src.lownerJohnEllipse import plot_ellipse, welzl
   
def get_radial_position(cell, centerX, centerY):
    """
    Convert cell position to radial coordinates relative to colony centroid
    
    @param  cell    cellStates object
    @param  centerX X-coordinate of colony centroid
    @param  centerY Y-coordinate of colony centroid
    @return r       Radial position of cell relative to colony centroid
    """
    
    r = math.sqrt((cell.pos[0] - centerX)**2 + (cell.pos[1] - centerY)**2)
    return r
    
def define_bin_ranges(bin_radius, r):
    """
    Define the radial ranges for bins to group cells by.
    
    @param  bin_radius  Length of the annuli to group cells by
    @param  r           Radial distance from centroid (nparray)
    @return bin_ranges  Bin intervals (distances) to average over (nparray)
    """
    # Get upper bound for the colony radius
    max_r = round(np.max(r))
    
    # Creat bin ranges
    bin_ranges = np.arange(0, math.ceil(max_r), bin_radius) #0-10, 10-20, etc.
    
    return bin_ranges
    
def bin_by_radius(r, measurement, bin_radius, centerX, centerY):
    """
    Place cell measurements into bins based on radial position in a colony
    
    @param  r           nparray of radial distances from colony centroid
    @param  measurement measurement of interest (float)
    @param  bin_radius  Thickness of band to average over (um)
    @param  centerX     Center of colony (X-coord)
    @param  centerY     Center of colony (Y-coord)
    @return df          dataFrame containing: ['Radius'], ['Measurement'], ['bin']
    """ 
    # Assign bins to each r and measurement
    bin_ranges = define_bin_ranges(bin_radius, r)
    bin_indices = np.digitize(r, bin_ranges) #Allocate each r to a bin number; first bin = 1 #https://www.adamsmith.haus/python/answers/how-to-put-data-into-bins-using-numpy-in-python
    
    # Create a dataFrame of radial position, measurement, and bin index
    df = pd.DataFrame()
    df['Radius'] = r.tolist()
    df['Growth Rate'] = measurement.tolist()
    df['bin'] = bin_indices.tolist()
    
    return df
    
def get_binned_statistics(df):
    """
    Calculate the mean and standard deviation of binned data
    
    @param  df          dataFrame with a column ['bin']; the column contains integers
    @return mean, std   Group dataFrames: mean and standard deviation of measurements (float)
    """
    # Group each cell by bin
    df_groupby_bin = df.groupby(df['bin'])
    
    # Calculate statistics for all other measurements
    mean = df_groupby_bin.mean()
    std = df_groupby_bin.std()
    
    return mean, std
    
def filter_by_cell_age(cells, min_age=2):
    """
    Create a dict with cells that have age >= min_age
    
    @param  cells               cellStates dict
    @param  min_age             minimum age to include cell in the filtered list; recommended age is 2
    @return filtered_cellstates cellStates dict with cells filtered by age
    """
    filtered_cellstates = {}
    for i, cell in cells.items():
        if cell.cellAge >= min_age:
            filtered_cellstates[i] = cell
    
    return filtered_cellstates
    
def get_all_centroids(cells):
    """
    Obtain centroids for all cells
    
    @param  cells           cellStates dict
    @return all_centroids   nparray (n_cells, 2) of cell centroids
    """
    # Initialize storage variables
    n_cells = len(cells.keys())
    all_centroids = np.zeros((n_cells, 2))
    
    # Obtain centroids for all cells
    for i, cell in enumerate(cells.values()):
        all_centroids[i][0] = cell.pos[0]
        all_centroids[i][1] = cell.pos[1]
        
    return all_centroids
    
def get_colony_centroid(cells):
    '''
    Fit an ellipse around colony using the welzl function
    
    @param  cells            cellStates dict
    @return centerX, centerY colony centroid coordinates [x0, y0]
    '''
    # Get cell centroids
    cell_centroids = get_all_centroids(cells)
    
    # Required check for larger colonies
    if len(cell_centroids) >= 1000:
        sys.setrecursionlimit(10000) # necessary to run welzl for colonies with n_cells > 1000  
    
    # Fit ellipse     
    center, major, minor, theta = welzl(cell_centroids)  
    
    centerX = center[0]
    centerY = center[1]  
    
    return centerX, centerY   
    
def find_knee(x, y, curvature_type="convex", direction_type="increasing"):
    """
    Finds point of maximum curvature in x, y data
    https://raghavan.usc.edu/papers/kneedle-simplex11.pdf
    https://pypi.org/project/kneed
    
    @param  x               X-coordinates of data
    @param  y               Y-coordinates of data
    @param  curvature_type  (string) "convex" or "concave"  
    @param  direction_type  (string) "increasing" or "decreasing"
    @return knee            X-coordinate of the knee
    """
    
    kneedle = KneeLocator(x, y, S=1.0, curve=curvature_type, direction=direction_type)
    knee = kneedle.knee
    
    return knee
    
def fit_line(x, y):
    """
    Fit a line to [x, y] data. Get regression coefficients and R^2.
    
    @param  x           (1D nparray) X data
    @param  y           (1D nparray) Y data
    @return slope       Slope of linear fit
    @return intercept   Intercept of linear fit
    @return r_squared   R^2 of linear fit
    """
    # Necessary to reshape data for LinearRegression model
    x_reshape = x.reshape((-1,1))
    
    # Linear regression
    model = LinearRegression().fit(x_reshape, y)
    r_squared = model.score(x_reshape, y)
    model_prediction = model.predict(x_reshape)
    slope = model.coef_[0] # slope
    intercept = model.intercept_ # intercept
    r_squared = model.score(x_reshape, y)
    
    return slope, intercept, r_squared
    
def main(cells, dt, bin_radius=4):
    """
    The main function.
    Calculates the slope of growth_rate vs. radial cell position in the colony in growth portions of the colony.
    This function is to be called by pyabc.
    
    @param  dt          Time step
    @param  bin_radius  Width to perform radial average in um (int)
    @param  cells       cellStates dict
    @return slope       Slope of growth_rate vs. position in colony
    """
    # Get colony centroid
    centerX, centerY = get_colony_centroid(cells)
    
    # Filter cells by age to exclude cells that were just born
    cells_filtered = filter_by_cell_age(cells, min_age=2)
    
    # Initialize storage variables
    r = np.zeros(len(cells_filtered))
    growth_rate = np.zeros(len(cells_filtered))
    
    # Collect radial positions and growth rates
    for i, cell in enumerate(cells_filtered.values()):
        r[i] = get_radial_position(cell, centerX, centerY)
        growth_rate[i] = cell.strainRate_rolling/dt #cell.normalized_growth_rate
        if math.isnan(growth_rate[i]):
            growth_rate[i] = 0
    
    # Create a dataFrame with each cell placed in a bin corresponding to radial position
    df = bin_by_radius(r, growth_rate, bin_radius, centerX, centerY)
    
    # Calculate statistics of binned measurements
    mean_df, std_df = get_binned_statistics(df)   
    
    # Get bin ranges for plotting purposes
    bin_ranges = define_bin_ranges(bin_radius, r)
        
    # Find knee, then get range of points to fit line
    knee = find_knee(bin_ranges, mean_df['Growth Rate'])
    try:
        knee_index = np.where(bin_ranges == knee)[0][0] 
    except:
        print("No knee, fit to entire curve")
        knee_index = 0
    bin_ranges_linear = bin_ranges[knee_index:]
    growth_rate_linear = mean_df['Growth Rate'].to_numpy()[knee_index:]
    
    # Fit line
    slope, intercept, r_squared = fit_line(bin_ranges_linear, growth_rate_linear)
    
    return slope

if __name__ == "__main__":
    '''
    For testing purposes only
    '''
    dt = 0.02
    
    # Load cellStates
    pickle_full_path = 'step-00420.pickle'
    data = pickle.load(open(pickle_full_path, 'rb'))
    cells = data['cellStates']   
    
    slope = main(cells, dt)
    print(slope)
