# ============================================================================================================
# Make_figs.py -- Generates the figures with the data produced by 1-Neuromodulation_and_plasticity.py
# and 2-Neuromodulation_and_plasticity_with_special_input.py
#
# Ref: Pedrosa V and Clopath C (2017) The Role of Neuromodulators in Cortical Plasticity. 
# A Computational Perspective. Front. Synaptic Neurosci. 8:38. doi: 10.3389/fnsyn.2016.00038
# -----------------------------------------------------------------------
#
# Author: Victor Pedrosa <v.pedrosa15@imperial.ac.uk>
# Imperial College London, London, UK - Dec 2016
# ============================================================================================================


# ------------------------------------- Import modules -------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import os
import numpy as np
from matplotlib.gridspec import GridSpec

plt.rc('xtick', labelsize=17)
plt.rc('ytick', labelsize=17)

colors = [[0,0,1],[1,0,0],[0,0.7,0],[1,0,1]]
labels = ['Rule DP','Rule PP','Rule UP','Rule DU']


fig = plt.figure(num=1,figsize=(15, 4), dpi=100, facecolor='w')
gs1 = GridSpec(1, 3)


# ================================================================================================================
# Use the data generated with 1-Neuromodulation_and_plasticity.py 
# ================================================================================================================

# ----------------------------------------------------------------------------------------------------------------
# Plot the weights after a fixed time

# Choose the directory and rearrange the files
Dir = 'Data_hom_stimulation/'
fnames = os.listdir(Dir)
fnames.sort()
fnames = [fnames[i] for i in [1,3,2,0]]

# Create the first subplot to show the weights
ax1 = plt.subplot(gs1[0, 0])
plt.xlabel('Neuron',fontsize='20')
plt.ylabel('Synaptic Weight', fontsize='20')
plt.xlim((1,10))
plt.ylim((0,2.0))

n = 12 # Choose the time at wi=hich the weights will be shown (time(T) = n*5 seconds)

# Initial weights:
W0 = np.mean(np.load(Dir+fnames[0]),axis=0)[0] 

# Mean synaptic weights at time T for the four different STDP rules
WsAllTrials0 = np.mean(np.load(Dir+fnames[0])[:,n],axis=0) 
WsAllTrials1 = np.mean(np.load(Dir+fnames[1])[:,n],axis=0)  
WsAllTrials2 = np.mean(np.load(Dir+fnames[2])[:,n],axis=0)  
WsAllTrials3 = np.mean(np.load(Dir+fnames[3])[:,n],axis=0)

# Standard deviation of synaptic weights at time T
WsAllTrials0_sd = np.std(np.load(Dir+fnames[0])[:,n],axis=0)  
WsAllTrials1_sd = np.std(np.load(Dir+fnames[1])[:,n],axis=0)  
WsAllTrials2_sd = np.std(np.load(Dir+fnames[2])[:,n],axis=0)  
WsAllTrials3_sd = np.std(np.load(Dir+fnames[3])[:,n],axis=0)  

# Plot all the data ---------------------------------------------------------------------------------------------
xdata = np.arange(1,11,1)

plt.plot(xdata,WsAllTrials0,color=colors[0],label=labels[0],lw=2)
plt.plot(xdata,WsAllTrials1,color=colors[1],label=labels[1],lw=2)
plt.plot(xdata,WsAllTrials2,color=colors[2],label=labels[2],lw=2)
plt.plot(xdata,WsAllTrials3,color=colors[3],label=labels[3],lw=2)

plt.plot(xdata,W0,color='k',label='Initial',lw=2)

plt.fill_between(xdata,WsAllTrials0-WsAllTrials0_sd,WsAllTrials0+WsAllTrials0_sd,alpha=0.2,color=colors[0])
plt.fill_between(xdata,WsAllTrials1-WsAllTrials1_sd,WsAllTrials1+WsAllTrials1_sd,alpha=0.2,color=colors[1])
plt.fill_between(xdata,WsAllTrials2-WsAllTrials2_sd,WsAllTrials2+WsAllTrials2_sd,alpha=0.2,color=colors[2])
plt.fill_between(xdata,WsAllTrials3-WsAllTrials3_sd,WsAllTrials3+WsAllTrials3_sd,alpha=0.2,color=colors[3])

plt.legend(fontsize=10,loc=0)


# ================================================================================================================
# Use the data generated with 2-Neuromodulation_and_plasticity_with_special_input.py 
# ================================================================================================================

# ----------------------------------------------------------------------------------------------------------------
# Plot the weights after a fixed time

# Choose the directory and rearrange the files
Dir = 'Data_RF_shift/'
fnames = os.listdir(Dir)
fnames.sort()
fnames = [fnames[i] for i in [1,3,2,0]]

# Create the second subplot to show the weights
ax2 = plt.subplot(gs1[0, 1])
plt.xlabel('Neuron',fontsize='20')
plt.ylabel('Synaptic Weight', fontsize='20')
plt.xlim((1,10))
plt.ylim((0,2.0))

n = 12 # Choose the time at wi=hich the weights will be shown (time(T) = n*5 seconds)

# Initial weights:
W0 = np.mean(np.load(Dir+fnames[0]),axis=0)[0] 

# Mean synaptic weights at time T for the four different STDP rules
WsAllTrials0 = np.mean(np.load(Dir+fnames[0])[:,n],axis=0) 
WsAllTrials1 = np.mean(np.load(Dir+fnames[1])[:,n],axis=0)  
WsAllTrials2 = np.mean(np.load(Dir+fnames[2])[:,n],axis=0)  
WsAllTrials3 = np.mean(np.load(Dir+fnames[3])[:,n],axis=0)

# Standard deviation of synaptic weights at time T
WsAllTrials0_sd = np.std(np.load(Dir+fnames[0])[:,n],axis=0)  
WsAllTrials1_sd = np.std(np.load(Dir+fnames[1])[:,n],axis=0)  
WsAllTrials2_sd = np.std(np.load(Dir+fnames[2])[:,n],axis=0)  
WsAllTrials3_sd = np.std(np.load(Dir+fnames[3])[:,n],axis=0)  

# Plot all the data ---------------------------------------------------------------------------------------------
xdata = np.arange(1,11,1)

plt.plot(xdata,WsAllTrials0,color=colors[0],label=labels[0],lw=2)
plt.plot(xdata,WsAllTrials1,color=colors[1],label=labels[1],lw=2)
plt.plot(xdata,WsAllTrials2,color=colors[2],label=labels[2],lw=2)
plt.plot(xdata,WsAllTrials3,color=colors[3],label=labels[3],lw=2)

plt.plot(xdata,W0,color='k',label='Initial',lw=2)

plt.fill_between(xdata,WsAllTrials0-WsAllTrials0_sd,WsAllTrials0+WsAllTrials0_sd,alpha=0.2,color=colors[0])
plt.fill_between(xdata,WsAllTrials1-WsAllTrials1_sd,WsAllTrials1+WsAllTrials1_sd,alpha=0.2,color=colors[1])
plt.fill_between(xdata,WsAllTrials2-WsAllTrials2_sd,WsAllTrials2+WsAllTrials2_sd,alpha=0.2,color=colors[2])
plt.fill_between(xdata,WsAllTrials3-WsAllTrials3_sd,WsAllTrials3+WsAllTrials3_sd,alpha=0.2,color=colors[3])

plt.legend(fontsize=10,loc=0)


# ----------------------------------------------------------------------------------------------------------------
# Plot the evolution of the input specificity (different between weight from inputs 4 and 7)

# Create the third subplot
ax3 = plt.subplot(gs1[0, 2])
plt.xlabel('time (s)',fontsize='20')
plt.ylabel('Input specificity', fontsize='20')
plt.xlim((1,1000))

for j in range(len(fnames)):
    WsAllTrials = np.load(Dir+fnames[j])
    WsAllTrials = np.mean(WsAllTrials,axis=0)
    Wpref_in = WsAllTrials[:-1,6]
    Wpref_new = WsAllTrials[:-1,3]
    
    timeVec = np.linspace(0,1000,Wpref_new.shape[0])
    plt.plot(timeVec,Wpref_new-Wpref_in,color=colors[j],label=labels[j],lw=2)
    plt.show()

plt.legend(fontsize=10,loc=0)

# ----------------------------------------------------------------------------------------------------------------
# Choose the directory and save the figures
plt.savefig('Figures/Fig1_Receptive_field_plasticity.png',dpi=300)

plt.show()