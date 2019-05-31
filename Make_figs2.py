# ============================================================================================================
# Make_figs2.py -- Generates the figures with the data produced by 
# 3-Neuromodulation_and_plasticity_Activity_vs_Learning_rate.py
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


Dir = 'Data_Activity_vs_Learning_rate/'

# ================================================================================================================
# Plot the synaptic weights for small and large learning rates 
# ================================================================================================================

# Files with the data to be loaded
f1 = 'Syn_weights_all_trials_learning_rate=00.0100_activity=01.0000.npy'
f2 = 'Syn_weights_all_trials_learning_rate=00.0200_activity=01.0000.npy'

# Create the figure
fig = plt.figure(num=2,figsize=(10, 4), dpi=100, facecolor='w')
gs1 = GridSpec(1, 2)

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# Plot the weights for small alpha and large alpha

# Load the initial and the final weights
n = -1
W0 = np.load(Dir+f1)[:,0]
W1 = np.load(Dir+f1)[:,n] # small alpha
W2 = np.load(Dir+f2)[:,n] # large alpha

# Redefine the weights as a weighted sum of the synaptic weights and re-scale it
W0_in = W0
sigma = .5
for i in range(10):
    NormDist = 1./np.sum(np.array([np.exp(-(x-i)**2/((2.*sigma)**2)) for x in range(10)]))
    InputWeights = NormDist*np.array([np.exp(-(x-i)**2/((2.*sigma)**2)) for x in range(10)])
    W0[:,i] = np.dot(W0_in,InputWeights) 
W0 = np.mean(W0,axis=0)
W0 = W0/np.max(W0)

W1_in = W1
sigma = .5
for i in range(10):
    NormDist = 1./np.sum(np.array([np.exp(-(x-i)**2/((2.*sigma)**2)) for x in range(10)]))
    InputWeights = NormDist*np.array([np.exp(-(x-i)**2/((2.*sigma)**2)) for x in range(10)])
    W1[:,i] = np.dot(W1_in,InputWeights) 

for i in range(W1.shape[0]):
    W1[i] = W1[i]/np.max(W1[i])

W2_in = W2
sigma = .5
for i in range(10):
    NormDist = 1./np.sum(np.array([np.exp(-(x-i)**2/((2.*sigma)**2)) for x in range(10)]))
    InputWeights = NormDist*np.array([np.exp(-(x-i)**2/((2.*sigma)**2)) for x in range(10)])
    W2[:,i] = np.dot(W2_in,InputWeights) 

for i in range(W2.shape[0]):
    W2[i] = W2[i]/np.max(W2[i])


# Calculate the mean weight over all the trials
W1_mean = np.mean(W1,axis=0)
W2_mean = np.mean(W2,axis=0)

# Plot the weights (initial and final)
xdata = np.arange(1,11,1)

ax1 = plt.subplot(gs1[0, 0])
plt.xlabel('Input',fontsize='20')
plt.ylabel('Synaptic Weight', fontsize='20')
plt.xlim((1,10))
plt.ylim((0,1.01))

color = 'r'
plt.plot(xdata,W0,color='k',label='Initial',lw=2)
plt.plot(xdata,W1_mean,color=color,label=r'Small $\alpha$',lw=2)

color = 'b'
plt.plot(xdata,W2_mean,color=color,label=r'Large $\alpha$',lw=2)
plt.legend(fontsize=10,loc=0)


# ================================================================================================================
# ================================================================================================================


# ================================================================================================================
# Plot the synaptic weights for small and large values of presynaptic activity 
# ================================================================================================================

f1 = 'Syn_weights_all_trials_learning_rate=00.0200_activity=01.0000.npy'
f2 = 'Syn_weights_all_trials_learning_rate=00.0200_activity=10.0000.npy'


# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# Plot the weights for small nu and large nu

# Load the initial and the final weights
n = -1
W0 = np.load(Dir+f1)[:,0,:10]
W1 = np.load(Dir+f1)[:,n,:10]
W2 = np.load(Dir+f2)[:,n,:10]

# Redefine the weights as a weighted sum of the synaptic weights and re-scale it
W0_in = W0
sigma = .5
for i in range(10):
    NormDist = 1./np.sum(np.array([np.exp(-(x-i)**2/((2.*sigma)**2)) for x in range(10)]))
    InputWeights = NormDist*np.array([np.exp(-(x-i)**2/((2.*sigma)**2)) for x in range(10)])
    W0[:,i] = np.dot(W0_in,InputWeights) 

W0 = np.mean(W0,axis=0)
W0 = W0/np.max(W0)

W1_in = W1
sigma = .5
for i in range(10):
    NormDist = 1./np.sum(np.array([np.exp(-(x-i)**2/((2.*sigma)**2)) for x in range(10)]))
    InputWeights = NormDist*np.array([np.exp(-(x-i)**2/((2.*sigma)**2)) for x in range(10)])
    W1[:,i] = np.dot(W1_in,InputWeights) 

for i in range(W1.shape[0]):
    W1[i] = W1[i]/np.max(W1[i])


W2_in = W2
sigma = .5
for i in range(10):
    NormDist = 1./np.sum(np.array([np.exp(-(x-i)**2/((2.*sigma)**2)) for x in range(10)]))
    InputWeights = NormDist*np.array([np.exp(-(x-i)**2/((2.*sigma)**2)) for x in range(10)])
    W2[:,i] = np.dot(W2_in,InputWeights) 

for i in range(W2.shape[0]):
    W2[i] = W2[i]/np.max(W2[i])


# Calculate the mean weight over all the trials
W1_mean = np.mean(W1,axis=0)
W1_sd = np.std(W1,axis=0)

W2_mean = np.mean(W2,axis=0)
W2_sd = np.std(W2,axis=0)

# Plot the weights (initial and final)
xdata = np.arange(1,11,1)

ax1 = plt.subplot(gs1[0, 1])
plt.xlabel('Input',fontsize='20')
plt.ylabel('Synaptic Weight', fontsize='20')
plt.xlim((1,10))
plt.ylim((0,1.01))

color = 'r'
plt.plot(xdata,W0,color='k',label='Initial',lw=2)
plt.plot(xdata,W1_mean,color=color,label=r'Small $\nu$',lw=2)

color = 'b'
plt.plot(xdata,W2_mean,color=color,label=r'Large $\nu$',lw=2)
plt.legend(fontsize=10,loc=0)

# ----------------------------------------------------------------------------------------------------------------
# Choose the directory and save the figures
plt.savefig('Figures/Fig2_Receptive_field_plasticity_Activity_vs_Learning_rate.png',dpi=300)

plt.show()
