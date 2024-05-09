# Standard modules
import math
import matplotlib.pyplot as plt
import numpy as np
import sys

# Installed modules
import CellModeller
from scipy.optimize import minimize_scalar, curve_fit, root
from sklearn.linear_model import LinearRegression

# Local/custom modules
from lje.src.lownerJohnEllipse import plot_ellipse, welzl

"""
The ellipse is rotated to get it in form of a standard, translated ellipse. 
The displacement of the ellipse center due to rotation is accounted for.

Algorithm for getting distance to edge:
1. Setup distance equation from arbitrary point to ellipse f(x)
2. Minimize the function
"""

'''
Math to find shortest distance to ellipse edge
'''
def rotate_point(p, theta):
    """
    Rotate a point p by angle theta
    
    @param  p           point with coordinates [x,y]
    @param  theta       angle of rotation (radians)
    @return p_rotated   point rotated by rotation matrix
    """
    p_rotated = [p[0]*np.cos(theta) - p[1]*np.sin(theta), 
                 p[0]*np.sin(theta) + p[1]*np.cos(theta)]
    
    return p_rotated
   
def distance2_func(t, *data):
    """
    Distance squared function of a standard, translated ellipse:
    x = x_shift + a*cos(t)
    y = y_shift + b*sin(t)
    0 <= t <=2*pi 
    
    @param  t       independent variable for parametric equations
    @param  *data   rotated_center, major, minor, rotated_cell_centroid
    @return d2      distance squared
    """
    # unpack the data
    rotated_center, major, minor, rotated_cell_centroid = data 
    
    # Parameters for equation of a line
    x0 = rotated_cell_centroid[0]
    y0 = rotated_cell_centroid[1]
    
    # Shift in center
    h = rotated_center[0] #shift in x
    k = rotated_center[1] #shift in y
    
    # Parametric form of ellipse equation
    x = h + major*np.cos(t)
    y = k + minor*np.sin(t)
    
    # Distance squared
    d2 = (x0 - x)**2 + (y0 - y)**2
    
    return d2
    
def get_optimum_point(center, major, minor, theta, cell_centroid):
    """
    Find the point on ellipse boundary closest to a bacterium.
    1. Rotate cell centroid and ellipse translation.
    2. Minimize distance squared function.
    3. Output point on ellipse with minimum distance to the bacterium. 
    
    @param  center          x,y coordinates of the ellipse center
    @param  major           major axis length of ellipse
    @param  minor           minor axis length of ellipse
    @param  theta           angle of rotation of ellipse (radians)
    @param  cell_centroid   x,y coordinates of bacterium
    @return optimum_point   point on ellipse boundary closest to the bacterium [x, y]
    """
    # Rotate cell centroid and ellipse center to put in standard form
    rotated_cell_centroid = rotate_point(cell_centroid, -theta)
    rotated_center = rotate_point(center, -theta)
    
    # Find minimum for distance squared functions
    function_inputs = (rotated_center, major, minor, rotated_cell_centroid)
    lb = 0
    ub = 2*math.pi
    options = {'maxiter': 1000}
    sol = minimize_scalar(distance2_func, method='Bounded', bounds=[lb,ub], args = function_inputs, options=options)
    t = sol.x

    # Getting coordinates of optimum point
    optimum_point_std = [rotated_center[0] + major*np.cos(t), rotated_center[1] + minor*np.sin(t)]
    optimum_point = rotate_point(optimum_point_std, theta)
    
    return optimum_point
    
'''
Filtering and data fetching
'''
def filter_by_cell_age(cells, min_age=2):
    """
    Create a dict with cells that have age >= min_age.
    
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
    Obtain centroids for all cells.
    
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
    Fit an ellipse around colony using the welzl function.
    
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
    
'''
Regression
'''    
def perform_linear_regression(x, y):
    """
    Fit a line to [x, y] data. Get regression coefficients and R**2.
    
    @param  x           (1D nparray) X data
    @param  y           (1D nparray) Y data
    @return slope       Slope of linear fit
    @return intercept   Intercept of linear fit
    @return r_squared   R**2 of linear fit
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
    
def exp_func(t, K):
    """
    Standard negative exponential function
    """
    return np.exp(-K * t)
    
def fit_exp(t, y):
    """
    Fit exponential function to data
    """
    opt_parms, parm_cov = curve_fit(exp_func, t, y, maxfev=10000)
    K = opt_parms
    
    return K   
    
"""
Main function
"""
def get_growth_parameter(cells, dt, show_plots=False):
    """
    The main function that fits the parameter describing growth vs. distance to colony border.
    
    @param  cells       cellStates dict
    @param  dt          simulation time step
    @param  show_plots  allows showing plots; set to True for viewing
    @return K           parameter describing decay in growth rate vs distance
    """
    # Fit ellipse     
    cell_centroids = get_all_centroids(cells)
    if len(cells.keys()) >= 1000:
        sys.setrecursionlimit(10000) # necessary to run welzl for colonies with n_cells > 1000  
    center, major, minor, theta = welzl(cell_centroids)  
    
    # Filter cells by age to exclude cells that were just born
    cells_filtered = filter_by_cell_age(cells, min_age=2)
    
    # Initialize storage variables
    d = np.zeros(len(cells_filtered)) # distance to ellipse boundary
    growth_rate = np.zeros(len(cells_filtered))
    points = []
    
    # Collect radial positions and growth rates
    for i, cell in enumerate(cells_filtered.values()):
        cell_centroid = [cell.pos[0], cell.pos[1]]
        optimum_point = get_optimum_point(center, major, minor, theta, cell_centroid)
        points.append(optimum_point)
        d[i] = math.dist(optimum_point, cell_centroid)
        growth_rate[i] = cell.strainRate_rolling/dt / cell.growthRate # growth rate normalized to max_growth
        if math.isnan(growth_rate[i]):
            growth_rate[i] = 0

    # Perform regression
    K = fit_exp(d, growth_rate)
    
    if show_plots:
        d_plot = np.linspace(0, np.max(d), num=100)
        linear_fit = exp_func(d_plot, K)
        
        # Show fit to data
        plt.figure(0)
        plt.scatter(d, growth_rate, s=4)
        plt.plot(d_plot, linear_fit, '--k', label='Linear fit')
        plt.xlabel('Distance to ellipse boundary (um)')
        plt.ylabel('Normalized growth rate')
        
        # Show minimum distances
        fig = plt.figure(1)
        ax = fig.add_subplot()
        ellipse = (center, major, minor, theta)
        show_ellipse_plot(ellipse, cell_centroids)
        plot_min_distances(cells_filtered, points)
        ax.axis('square')

        orthogonality = test_orthogonality(center, major, minor, theta, cells_filtered, points)
        print(f'Mean of dot products between cell-to-ellipse vector and ellipse tangent = {orthogonality}')
        
        plt.show()
     
    return K
    
"""
Extra functions for testing and visualization
"""    
def show_ellipse_plot(ellipse, centroids):
    """
    Plot the fitted ellipse with individual cells as points
    
    @param ellipse      tuple (center, major, minor, theta)
    @param centroids    2D array containing x, y coordinates of each cell
    """
    plot_ellipse(ellipse, str='k--')
    plt.scatter(centroids[:, 0], centroids[:, 1])
        
def plot_min_distances(cells, points):
    """
    Plot lines from cells to points
    """  
    for i, cell in enumerate(cells.values()):
        x = [cell.pos[0], points[i][0]]
        y = [cell.pos[1], points[i][1]]
        plt.plot(x, y)
        
def test_orthogonality(center, major, minor, theta, cells, opt_points):
    """
    For each cell, perform the dot product between the (1) vector pointing from a cell to 
    the closest point on ellipse boundary and (2) vector tangent to the ellipse. 
    The lines should be perpendicular if the point on the ellipse boundary has the minimum
    distance from the cell.
    
    Expected result: v_cell_to_point \cdot v_tangent_at_point = 0
    
    @param  center          x,y coordinates of the ellipse center
    @param  major           major axis length of ellipse
    @param  minor           minor axis length of ellipse
    @param  theta           angle of rotation of ellipse (radians)
    @param  cells           cellStates dict
    @return opt_points      list of point on ellipse boundary closest to the bacterium [x, y]
                            (in same order as the cellStates dict)
    """
    # Rotate ellipse translation coordinates
    center = rotate_point(center, -theta)
    
    # Loop to calculate orthogonality for each cell and optimum point
    orthogonality = []
    for i, cell in enumerate(cells.values()):
        # Collect coordinates
        ellipse_x = opt_points[i][0]
        ellipse_y = opt_points[i][1]
        cell_x = cell.pos[0]
        cell_y = cell.pos[1]
        
        # Rotate by -theta to get into std form
        rotated_point = rotate_point(opt_points[i], -theta)
        rotated_cell_centroid = rotate_point([cell_x, cell_y], -theta)
        
        # Calculate t given x and y
        t = np.arctan(major/minor*(rotated_point[1] - center[1])/(rotated_point[0] - center[0]))
        
        # Get derivatives wrt t
        dydt = minor*np.cos(t)
        dxdt = -major*np.sin(t)
        
        # Calculate dy/dx by doing dy/dt / dx/dt
        dydx = dydt/dxdt
        
        # Define vector for ellipse
        ellipse_tangent = [1, dydx] 
        
        # Get vector from cell to opt_point
        line_x = rotated_cell_centroid[0] - rotated_point[0]
        line_y = rotated_cell_centroid[1]  - rotated_point[1]
        point_to_ellipse = [line_x, line_y]
        
        # v_cell_to_point \cdot v_tangent_at_point
        orthogonality.append(np.dot(ellipse_tangent, point_to_ellipse))
    
    # Uncomment to see all dot products
    '''
    for val in orthogonality:
        print(f'Dot product w/ cell to optimum point and ellipse tangent = {val:.3f}')
    '''
        
    return np.mean(orthogonality)
