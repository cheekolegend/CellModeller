# Standard modules
import sys
import numpy as np
import pickle #only needed for testing

# Installed modules 
import CellModeller

# Local/custom modules
from lje.src.lownerJohnEllipse import plot_ellipse, welzl

def get_centroids(cells):    
    """
    Obtain cell controids
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
     
def calc_aspect_ratio(a, b):
    """
    Compute the aspect ratio of an ellipse

    @param  a               (float) one axis of the ellipse
    @param  b               (float) other axis of the ellipse
    @return aspect_ratio    (float) short/long axis
    """
    aspect_ratio = np.minimum(a, b)/np.maximum(a, b)

    return aspect_ratio

def main(cells):
    """
    The main function that finds the aspect ratio of a colony.
    To be called by pyabc.
    
    @param  cells   cellStates dict
    @param  center  center of the fitted ellipse (x, y)
    @param  major   major axis length of the fitted ellipse
    @param  minor   minor axis length of the fitted ellipse
    @parap  theta   angle of rotation of the fitted ellipse
    @return dist    nparray of shortest distances to colony border along each cell's major axis
    """
    if len(cells.keys()) >= 1000:
        sys.setrecursionlimit(10000) # necessary to run welzl for colonies with n_cells > 1000   
        
    # Find cell centroids
    centroids = get_centroids(cells)
    
    # Fit ellipse around colony
    center, major, minor, theta = welzl(centroids)
    
    # Calculate aspect ratio
    aspect_ratio = calc_aspect_ratio(minor, major)
    
    return aspect_ratio   
    
if __name__ == '__main__':    
    '''
    For testing purposes only
    '''
    # Load cellStates
    pickle_full_path = 'step-00420.pickle'
    data = pickle.load(open(pickle_full_path, 'rb'))
    cells = data['cellStates']   
    
    aspect_ratio = main(cells)
    print(aspect_ratio)
    
    
