'''
This is a demo simulation for growth in a microfluidic device, where cell growth is determined by a limiting nutrient. The nutrient field is consumed proportionally to the cell growth rate.

The nutrient field u is calculated based on the species transport equation, where du/dt and advection are neglected:

D*laplace(u) = -mu_max*u*VolumeFraction()/(u+K)

This model is valid if cell growth is the limiting time scale. The nutrient field can be calculated separately to a pseudo steady-state.
The volume fraction is computed based on the cell center (i.e. a point within a mesh element). The element size must be >= the length scale of a bacterium.

The geometry is as follows:

|--------------open-----------------|
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|                                   |
|___________________________________|

Domain size is 80 X 80 um. 
In the solver, the domain is extended by two elements to ensure that cells do not touch the boundary elements.
Bacteria on the open boundary elements will make the solver fail as we will be placing a source term into elements with Dirichlet BCs, causing a very steep gradient that the solver cannot handle.

The dolfin solver is based on the work of William P.J. Smith: Proceedings of the National Academy of Sciences Jan 2017, 114 (3) E280-E286; DOI: 10.1073/pnas.1613007114
Github: https://github.com/WilliamPJSmith

The PDESolver setup for this module is in monod_growth_1open_DolfinPDESolver.py.
'''

import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.GUI import Renderers
import numpy
import math
import os #needed for outputting files in PDEsolver

cell_cols = {0:[0,1.0,0], 1:[1.0,0,0], 2:[0,0,1.0]} #RGB cell colours
cell_lens = {0:3.5, 1:3.5, 2:3.5} #target cell lengths
cell_growr = {0:1, 1:1, 2:1} #growth rates

# Domain geometry and discretization
len_x = 80.0
len_y = 80.0
num_x = 10.0
num_y = 10.0

# Picklesteps
save_every = 10

# Growth variables; arbitrarily chosen for now
sim_mu_max = 1.0
sim_K = 0.2
sim_D = 2.5*math.pow(10,3) #um2/h

def setup(sim):
    # Set biophysics, signalling, and regulation models
    sim.dt = 0.02
    
    friction_factor = 1e6 #default is 10
    biophys = CLBacterium(sim, jitter_z=False, max_planes=3, gamma=friction_factor, max_cells=5000)
    
    # Solver parameters:
    # u0 = Field concentration at open boundaries
    # mu_max = maximum growth rate
    # K = Half-saturation constant
    solverParams = dict(N_x = num_x, \
                        N_y = num_y, \
                        L_x = len_x, \
                        L_y = len_y, \
                        u0 = 1, \
                        D = sim_D, \
                        mu_max = sim_mu_max, \
                        K = sim_K, \
                        mesh_type = 'crossed', \
                        rel_tol = 1e-6, \
                        pickleSteps = save_every)

    # use this file for reg too
    regul = ModuleRegulator(sim, sim.moduleName)
    #regul = ModuleRegulator(sim, __file__)
    # Only biophys and regulation
    sim.init(biophys, regul, None, None, solverParams)
 
    planeWeight = 1.0
    biophys.addPlane((0,0,0), (1,0,0), planeWeight) 		    # left side
    biophys.addPlane((len_x,0,0), (-1,0,0), planeWeight) 		# right side
    biophys.addPlane((0,0,0), (0,1,0), planeWeight) 			# base
 
    # Specify the initial cell and its location in the simulation
    sim.addCell(cellType=0, pos=(len_x/6.0,len_y-10,0), dir=(0,1,0))
    sim.addCell(cellType=1, pos=(len_x/3.0,len_y-20,0), dir=(0,1,0))
    sim.addCell(cellType=2, pos=(len_x/2.0,len_y-30,0), dir=(0,1,0))


    # Add some objects to draw the models
    therenderer = Renderers.GLBacteriumRenderer(sim)
    sim.addRenderer(therenderer)
    sim.pickleSteps = save_every
    

def init(cell):
    # Specify mean and distribution of initial cell size
    cell.targetVol = cell_lens[cell.cellType] + random.uniform(0.0,0.5)
    # Specify growth rate of cells
    cell.growthRate = cell_growr[cell.cellType] # Ideally initialize based on initial nutrient field, but okay for demo
    cell.color = cell_cols[cell.cellType]
    

def update(cells): 
    #Iterate through each cell; remove cells exiting open boundaries; flag cells for division;
    for (id, cell) in cells.items():
        if cell.pos[1] > (len_y): #cells must be removed from sim while still within FEniCS domain.
            cell.deathFlag = True
        if cell.volume > cell.targetVol and cell.deathFlag == False:
            cell.divideFlag = True
            
            
def solvePDEandGrowth(sim,cells):
    '''
    The algorithm is as follows:  
        get cell centers and volumes
        call PDE solver; return nutrient field u_local 
        update growthrates for each cell based on their u_local
    
    We need to loop through u_local in the same order as we did for centers and vols to remain consistent with assigning u(x,y) to the corresponding cell.
    Cannot use cell.idx as WPJS did because sim.next_idx does not account for removed cells, it just keeps growing, so it will return an indexing error
    
    FYI: cells is equivalent to sim.cellStates 
    '''

    filename = os.path.join(sim.outputDirPath, 'step-%05i.pvd' % sim.stepNum)
    directory = ""
    centers = []
    vols = []
    
    # Create lists of centers and volumes from cellStates   
    for cell in cells.values():
        vol_calc = cell.length*2*cell.radius + 3.141592**cell.radius #cell area
        centers.append(cell.pos) #TODO: convert cell.pos lists to tuples?
        vols.append(vol_calc)
    
    # Convert to numpy.array (required to work with FEniCS)
    centers = numpy.array(centers)
    vols = numpy.array(vols)
    
    u_local = sim.solver.SolvePDE(centers, vols, filename, directory, sim.stepNum)

    # Set growth rates        
    for i,cell in enumerate(cells.values()):
        u_this = u_local[i]
        cell.growthRate = sim_mu_max*u_this/(u_this + sim_K)
    

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    d1.targetVol = cell_lens[parent.cellType] + random.uniform(0.0,0.5)
    d2.targetVol = cell_lens[parent.cellType] + random.uniform(0.0,0.5)
    
    
def kill(cell):
    """
    This function specifies how to change a cell's state when it dies.
    """ 
    # define what happens to a cell's state when it dies
    cell.growthRate = 0.0       # dead cells can't grow any more
    cell.divideFlag = False     # dead cells can't divide
