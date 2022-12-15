CellModeller4
=============
Multicellular modelling framework, created by Tim Rudge, PJ Steiner, and Jim Haseloff, University of Cambridge

## Microfluidic Environments
This fork of CellModeller can simulate growth in microfluidic traps. This is permitted by the following functionalities:

- Cell removal from the simulation
- PDE solver with user-defined boundary conditions

Cell removal is handled by deleting the cell id from sim.CellStates and excluding these cells from the biophysics handling. See [this tutorial](Examples/AaronYip/cell_removal).

Scalar fields are handled by [FEniCS](https://fenicsproject.org/) 2019.1.0, an open-source PDE solver. The CellModeller engine and FEniCS solver are fully coupled. Please download FEniCS from their website. Try running [this tutorial](Examples/AaronYip/monod_growth_1open).

Cells and fields can be visualized simultaneously in [Paraview](https://www.paraview.org/) by converting .pickle files to .vtp format. Some parts of this are still in progress.

All the above updates are based on the work of @WilliamPJSmith.

/!\ Simulations interfaced with FEniCS cannot be run in the GUI.

## Approximate Bayesian Computation
This fork is interfaced to work with [pyabc](https://pyabc.readthedocs.io/en/latest/). See [this tutorial](Examples/AaronYip/pyabc_cellmodeller) for a minimal example. 

To setup a calibration, setup the setparams function like so in the CellModeller module file:

```
def setparams(param_dict):
    global your_parameter
    your_parameter = param_dict['your_parameter']
```

This defines which parameters will be calibrated. Afterwards, setup the the [ABC simulation script](Examples/AaronYip/pyabc_cellmodeller/pyabc_cellmodeller.py) in a similar fashion to the tutorials in the [pyabc documentation](https://pyabc.readthedocs.io/en/latest/examples/parameter_inference.html)

## Adhesion Module
The adhesion module originally published by [Kan et al. (2018)](https://doi.org/10.1098/rsif.2018.0406) has been updated to work with the current biophysics module. Adhesion to walls has been added as a new feature. See the [example simulations](Examples/AaronYip/adhesion) for more information.

## Degradation of Extracellular Molecules
A term for reaction has been added to the [Signalling module](https://github.com/cheekolegend/CellModeller/blob/master/CellModeller/Signalling/GridDiffusion.py).
