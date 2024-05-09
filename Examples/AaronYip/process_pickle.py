'''
My first attempt at reading a pickle file. Pipeline taken from contactGraphy.py. 

After reading in the pickle file as 'data', data contains most of the simulation data. Lists containing data about cells are denoted by a cell_id number
    'cellStates'    list of CellState objects   The cellState of each cell
    'stepNum'       int                         Time step
    'lineage'       list of ints                (presumably) the parent of the cell
    'moduleStr'     string                      The simulation code
    'moduleName'    string                      Name of the simulation file (without .py)
    'specData'      array(dtype=float32)        Species concentration (within cell?)
    'sigGridOrig'
    'sigGridDim'
    'sigGridSize'
    'sigGrid'                                   Extracellular species concentrations
    'sigData'       array([[sig1,sig2],...],dtype=float32)  Intracellular species concentrations
    
TODO: get the following cell data (see def draw_cells(self) in Draw2DPDF.py) from the pickle
            p = state.pos
            d = state.dir
            l = state.length
            r = state.radius
'''

import sys
import os
import pickle

fname = 'test_contact.pickle'
data = pickle.load(open(fname,'rb'))

cs = data['cellStates']
it = iter(cs)
n = len(cs)
cell_type={}
pos_dict={}
growth_rate={}
for it in cs:
    cell_type[cs[it].idx] = cs[it].cellType
    pos_dict[cs[it].idx] = cs[it].pos[0:2]
    growth_rate[cs[it].idx] = cs[it].growthRate
modname = data['moduleName']
moduleStr = data['moduleStr']

print(cs)

#print attribute values for one cell
index = next(iter(cs))
print(vars(cs[index]))

lineage = data['lineage']
print(lineage)
