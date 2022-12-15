"""
The purpose of this simulation is to demonstrate adhesion to walls.
Two cells are placed against separate walls, and an equal and constant force is applied to both of them.
The sticky cell should move slower than the non-sticky cell if wall adhesion is working.

        |                     |
        |                     |
        |                     |
        |sticky     non-sticky|
        |cell             cell|
wall    |                     |     wall
        | ^                ^  |
        | |                |  |
        | Force         Force |
        |                     |

Instructions:
1. Adjust "force" variable to desired force
2. Adjust cell_adh variables. cellType 0 (red) is non-sticky, cellType 1 (green) is sticky
3. Run
"""

from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacteriumAdhesion import CLBacteriumAdhesion
from CellModeller.GUI import Renderers

# Simulation settings
max_cells = 10
dt = 0.025 #h
sim_time = 10.0

# Cell properties
cell_col = [[1.0,0.0,0.0],[0.0,1.0,0.0]]
cell_radius = 0.5
cell_adh = [0., 0.] # cell adhesion "spring" constants
force = 0.005
cell_forces = [(0,force,0,0), (0,force,0,0)]

def setup(sim):
    # Set biophysics models; externalForces = True to allow applying external forces to cells for testing
    biophys = CLBacteriumAdhesion(sim, dt=dt, max_planes=2, jitter_z=False, cgs_tol=1e-5, compNeighbours=False, max_contacts=32, gamma=100.0, max_cells=max_cells, externalForces=True)
    
    # Initialize planes at x = 0 and x = 3.5
    biophys.addPlane((0,0,0), (1,0,0), 1) #left boundary
    biophys.addPlane((9.5,0,0), (-1,0,0), 1) #right boundary
    
    # Initialize module regulator
    regul = ModuleRegulator(sim, sim.moduleName)
    
    # Only biophys and regulation
    sim.init(biophys, regul, None, None)

    # Specify the initial cells, their location in the simulation, and applied forces
    sim.addCell(cellType=1, pos=(0.49,0,0), dir=(0,1,0), cellForce=(-force,force,0,0)) #sticky  
    sim.addCell(cellType=0, pos=(9*2*cell_radius,0,0), dir=(0,1,0), cellForce=(force,force,0,0)) #not sticky

    # Add some objects to draw the models
    therenderer = Renderers.GLBacteriumRenderer(sim)
    sim.addRenderer(therenderer)
    
    # Save data
    sim.pickleSteps = 10

# Adhesion logic - here the adhesion strength is the weakest cellAdh of the two contacting cells
# For logic on wall adhesion, see build_tangent_matrix function in AdhKernel.cl
def adhLogicCL():
    return """return min(adh_str1, adh_str2);"""
        
def init(cell):
    cell.targetVol = 3.5
    cell.growthRate = 0
    cell.color = cell_col[cell.cellType]
    cell.cellAdh = cell_adh[cell.cellType] # Stickiness defined here

def update(cells):
    #Iterate through each cell and flag cells that reach target size for division
    pass

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    pass
