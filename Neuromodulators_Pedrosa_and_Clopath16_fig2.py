# ============================================================================================================
# Neuromodulators_Pedrosa_and_Clopath16_fig2.py -- Simulates a feedforward network of excitatory neurons as in
#
# Ref: Pedrosa V and Clopath C (2017) The Role of Neuromodulators in Cortical Plasticity. 
# A Computational Perspective. Front. Synaptic Neurosci. 8:38. doi: 10.3389/fnsyn.2016.00038
#
# This code executes 3-Neuromodulation_and_plasticity_Activity_vs_Learning_rate.py
# to simulate a feedforward network of neurons to compare the effect of modulating the learning rate 
# with the effect of modulating the neuronal activity.
# Finally, this code executes Make_figs2.py to generate the final figures.
# -----------------------------------------------------------------------
#
# Author: Victor Pedrosa <v.pedrosa15@imperial.ac.uk>
# Imperial College London, London, UK - Dec 2016
# ====================================================================================================


# Import modules -------------------------------------------------------------------------------------
import subprocess
import numpy as np
import os
from time import time as time_now


# Create new directories to store data ---------------------------------------------------------------
newpath = r'Data_Activity_vs_Learning_rate/' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
    
newpath = r'Figures/' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

# start to count the time spent with simulations -----------------------------------------------------
time_in = time_now() 


# ====================================================================================================
# Define the arguments to be used when calling the main code.
# Each pair of arguments corresponds to the amplitude of synaptic plasticity for pre-post and post-pre
# events.
# ====================================================================================================
args = [[0.02, 1.], [0.01, 1.], [0.02, 10.]] 


# ====================================================================================================
# Run the simulations
# ====================================================================================================

# Run the main code for homogeneous stimulation ------------------------------------------------------
for arg in args:
	subprocess.call('python 3-Neuromodulation_and_plasticity_Activity_vs_Learning_rate.py {0} {1}'.format(arg[0],arg[1]),shell=True)




# stop counting the time and show the total time spent -----------------------------------------------
time_end = time_now()
total_time = (time_end-time_in)/60. # [min]

print('\n')
print('Simulation finally finished!')
print('Total time = {0:.2f} minutos'.format(total_time))


# Run the code to generate the figures ---------------------------------------------------------------

subprocess.call('python Make_figs2.py',shell=True)
