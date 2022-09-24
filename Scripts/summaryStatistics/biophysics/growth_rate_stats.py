'''
Algorithm for plotting growth rate vs nearest location to colony edge:

Obtain equation for ellipse surrounding colony
For each cell:
    Obtain equation of line along cell's long axis
    Find intersections of line and ellipse
    Calculate shortest distance between centroid and intersections; select shortest distance
    
Plot instantaneous growth rate (strain rate) vs. shortest distance to ellipse boundary for all cells

Instructions: 
Call the main() function, giving a cellStates dict as an argument.
Optional: include path to a directory to export plots
'''
# Standard modules
import math
import numpy as np
import os
import pickle #only needed for testing
import sys

# Installed modules 
import CellModeller
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from sklearn.linear_model import LinearRegression

# Local/custom modules
from lje.src.lownerJohnEllipse import plot_ellipse, welzl

def show_ellipse_plot(ellipse, centroids):
    """
    Plot the fitted ellipse with individual cells as points
    
    @param ellipse      tuple (center, major, minor, theta)
    @param centroids    2D array containing x, y coordinates of each cell
    """
    plot_ellipse(ellipse, str='k--')
    plt.scatter(centroids[:, 0], centroids[:, 1])
    plt.show()    
    
def plot_intersection(ellipse, cell, intersection):
    """
    Draws a plot of an ellipse, line, and intersection(s).
    For validation purposes.
    
    @param ellipse      tuple with the following: (center, major, minor, theta)
    @param cell         cellState object
    @param intersection array of intersections between the line and ellipse [(x1,y1), (x2,y2)]  
    """
       
    # Values for plotting line
    slope = cell.dir[1]/cell.dir[0]
    cell_x = cell.pos[0]
    cell_y = cell.pos[1]
    major_axis = ellipse[1]
    x_vals = np.arange(-major_axis, major_axis)
    y_vals = slope*(x_vals - cell_x) + cell_y
       
    # Converting intersection points to list
    intersection_x = []
    intersection_y = []
    for pair in intersection:
        intersection_x.append(pair[0])
        intersection_y.append(pair[1])
    
    # Plots
    plt.plot(x_vals, y_vals, 'b') # line
    plt.scatter(intersection_x, intersection_y) # intersections
    plt.scatter(cell_x, cell_y)
    plot_ellipse(ellipse, str='k--') # ellipse
    
    plt.show()
    
def func(x, *data):
    """
    System of equations to solve for intersection of a line and ellipse
    
    @param  x        List of variables we are solving for
    @param  data     Parameters being passed into the function: center, major, minor, theta, cell
    @return eqs      List of equations
    """
    
    # unpack the data
    center, major, minor, theta, cell = data 
    
    # Parameters for equation of a line
    cell_x = cell.pos[0]
    cell_y = cell.pos[1]
    slope = cell.dir[1]/cell.dir[0]
    
    # Other parameters for ellipse equation
    h = center[0] #shift in x
    k = center[1] #shift in y
    
    eqs = [((x[0]-h)*np.cos(theta) + (x[1] - k)*np.sin(theta))**2/major**2 + ((x[0] - h)*np.sin(theta) - (x[1] - k)*np.cos(theta))**2/minor**2 - 1,
            slope*(x[0] - cell_x) + cell_y - x[1]]
    
    return eqs    

def shortest_distance(point1, point2, cell_centroid):
    """
    Calculate shortest distance between a cell centroid and two points
    
    @param  point1          
    @param  point2
    @param  cell_centroid
    @return                 smallest distance between (point1, cell_centroid) or (point2, cell_centroid)
    """
    
    dist1 = math.dist(point1, cell_centroid)
    dist2 = math.dist(point2, cell_centroid)
    
    return min(dist1, dist2)
    
def solve_roots(function_inputs):
    """
    Finds the roots of the function func(x)
    
    @param  function_inputs tuple consisting of 5 args (first four are outputs from welzl, fifth is CellState object)
    @return [root1, root2]  list of intersections between line and ellipse
    """
    center = function_inputs[0]
    major = function_inputs[1]
    x0_1 = [-major, center[1]] # LHS of ellipse
    x0_2 = [major, center[1]] # RHS of ellipse
    root1 = fsolve(func, x0_1, args = function_inputs)
    root2 = fsolve(func, x0_2, args = function_inputs)
    
    # If case the solver fails to find two unique roots, try again with different initial guesses
    if root1.all == root2.all:
        # Reset initial guesses to bottom left corner and top right corner
        x0_1 = [-major, -major]
        x0_2 = [major, major]
        root1 = fsolve(func, x0_1, args = function_inputs)
        root2 = fsolve(func, x0_2, args = function_inputs)
        
    return [root1, root2]
    
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
                
def get_growth_rates(cells, dt):    
    """
    Obtain growth rates for all cells with age > 1
    
    @param  cells           cellStates dict
    @return growth_rates    nparray of growth rates (dL/dt)
    """                
    # Initialize storage variables
    n_cells = len(cells.keys())
    growth_rates = np.zeros(n_cells)

    # Get single-cell properties
    for i, cell in enumerate (cells.values()):
        growth_rates[i] = cell.strainRate_rolling/dt
    
    return growth_rates

def get_centroids(cells):    
    """
    Obtain growth rates for all cells with age > 1
    @param  cells           cellStates dict
    @return centroids       nparray of centroids (n_cells, 2)
    """                
    # Initialize storage variables
    n_cells = len(cells.keys())
    centroids = np.zeros((n_cells, 2))
    
    # Collect centroid values from cellStates dict
    for i, cell in enumerate(cells.values()):
        centroids[i][0] = cell.pos[0]
        centroids[i][1] = cell.pos[1]
    
    return centroids    
     
def get_distances_to_colony_edge(cells, center, major, minor, theta):
    """
    Obtain distances to colony edge for a list of cells
    
    @param  cells   cellStates dict
    @param  center  center of the fitted ellipse (x, y)
    @param  major   major axis length of the fitted ellipse
    @param  minor   minor axis length of the fitted ellipse
    @parap  theta   angle of rotation of the fitted ellipse
    @return dist    nparray of shortest distances to colony border along each cell's major axis
    """

    # Initialize storage variables
    n_cells = len(cells)
    dist = np.zeros(n_cells)
    
    # Fit ellipse around colony
    center, major, minor, theta = welzl(all_centroids)   
    
    # For each cell, calculate min distance to colony border in direction of cell's major axis    
    for i, cell in enumerate(cells.values()):
        cell_centroid = [cell.pos[0], cell.pos[1]]
        function_inputs = (center, major, minor, theta, cell)
        roots = solve_roots(function_inputs)
        dist[i] = shortest_distance(roots[0], roots[1], cell_centroid)
        
    return dist
    
def plot_growth_vs_distance_to_colony_edge(growth_rates, dist, export_path):
    """
    Save a plot of the data in the export_path
    
    @param  growth_rates    nparray of growth rates
    @param  dist            nparray of distances to colony edge
    @param  export_path     directory where figure will be exported
    """
    dist_reshape = dist.reshape((-1,1))
    
    # Mean growth rate
    mean_growth_rate = np.mean(growth_rates)
    std_growth_rate = np.std(growth_rates)
    
    # Linear regression
    model = LinearRegression().fit(dist_reshape, growth_rates)
    r_squared = model.score(dist_reshape, growth_rates)
    model_prediction = model.predict(dist_reshape)
    slope = model.coef_[0] # slope
    
    # Plot results
    plt.scatter(dist, growth_rates, color = 'black', s=1)
    plt.plot(dist_reshape, model_prediction, '--b')
    
    # Plot annotation
    plt.figure(1)
    plt.xlabel("Minimum Distance to Colony Border Along Cell's Major Axis ($\mathrm{\mu}$m)")
    plt.ylabel('Growth Rate (1/h)')
    plt.ylim(bottom=-0.05, top=np.max(growth_rates)+0.05)
    linear_fit_text = 'y = {slope: .3f}x + {intercept: .3f}'.format(slope = slope, intercept = model.intercept_)
    plt.text(20, np.max(growth_rates) - 0.05, linear_fit_text)
    plt.text(20, np.max(growth_rates) - 0.1, f'R^2 = {r_squared: .2f}')
    plt.text(20, np.max(growth_rates) - 0.15, f'Growth rate = {mean_growth_rate: .2f} $\pm$ {std_growth_rate: .2f}')  
    plt.savefig(export_path + "growth_vs_distance_to_colony_edge.png", bbox_inches='tight')  
    
def main(cells, dt, fig_export_path=''):
    """
    The main function that will be used by pyabc
    
    Takes as input a cellStates dict and outputs the slope of the linear fit to
    distance to colony border along cell's main axis vs. growth rate
    
    @param  cells               cellStates dict
    @param  dt                  simulation time step (h)
    @param  fig_export_path     optional: path to export a plot of growth rate vs. distance to colony edge
    @return dist_vs_growth_rate slope of linear fit to distance vs. growth rate
    """
    # Create cellStates dict filtered for cellAge > 1
    cells_age_filtered = filter_by_cell_age(cells, min_age=2)
    n_cells_age_filtered = len(cells_age_filtered.keys())   

    # Get single-cell properties
    growth_rates = get_growth_rates(cells_age_filtered, dt)
    all_centroids = get_centroids(cells)
        
    # Fit ellipse around colony
    if len(cells.keys()) >= 1000:
        sys.setrecursionlimit(10000) # necessary to run welzl for colonies with n_cells > 1000   
    center, major, minor, theta = welzl(all_centroids)   
    
    # For each cell, calculate min distance to colony border in direction of cell's major axis    
    dist = get_distances_to_colony_edge(cells_age_filtered, center, major, minor, theta)
        
    # Linear fit of growth rate vs distance to colony edge along cell's major axis
    dist_reshape = dist.reshape((-1,1))
    model = LinearRegression().fit(dist_reshape, growth_rates)
    dist_vs_growth_rate = model.coef_[0] # slope
    
    # Optional: save a plot to fig_export_path
    if fig_export_path:
        plot_growth_vs_distance_to_colony_edge(growth_rates, dist, fig_export_path)
        print('Growth rate plot exported to: ', fig_export_path)
    
    return dist_vs_growth_rate

if __name__ == '__main__':   
    '''
    For testing purposes only
    '''
    
    # Specify time step 
    dt = 0.02 # h  
    
    # Import data
    pickle_full_path = 'step-00420.pickle'
    data = pickle.load(open(pickle_full_path, 'rb'))
    cells = data['cellStates']   
    
    # Export location of plots
    export_path = "data/"  
       
    # Run calculation
    dist_vs_growth_rate = main(cells, dt, export_path)
    
    # Plotting the cell positions with cell_ids
    '''    
    plt.figure(1)
    for i, cell_id in enumerate(zero_growth_cell_ids):
        plt.annotate(cell_id, (zero_growth_cell_pos[i][0], zero_growth_cell_pos[i][1]), fontsize=6)
    show_ellipse_plot((center, major, minor, theta), all_centroids)
    '''
    
