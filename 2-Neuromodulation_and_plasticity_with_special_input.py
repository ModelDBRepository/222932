# ============================================================================================================
# ============================================================================================================
# 2-Neuromodulation_and_plasticity_with_special_input.py -- Simulates a feedforward network of excitatory 
# neurons as in
#
# Ref: Pedrosa V and Clopath C (2017) The Role of Neuromodulators in Cortical Plasticity. 
# A Computational Perspective. Front. Synaptic Neurosci. 8:38. doi: 10.3389/fnsyn.2016.00038
# -----------------------------------------------------------------------
#
# Author: Victor Pedrosa <v.pedrosa15@imperial.ac.uk>
# Imperial College London, London, UK - Dec 2016
# -----------------------------------------------------------------------
# 
# Arguments to call the code:
#   arg1 = a_plus	-> Amplitude of plasticity for pre-post events
#   arg2 = a_minus	-> Amplitude of plasticity for post-pre events
# ============================================================================================================


# ------------------------------------- Import modules -------------------------------------------------------

import numpy as np
import scipy.io as sio
import multiprocessing as mp
from time import time as time_now
import sys as sys
import Filt_Gaussian_Noise as FGN

args = sys.argv
time_in = time_now()



# ============================================================================================================
# +++++++++++++++++++++++++++++++++++++++++ Parameters +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ============================================================================================================

# Parameters +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
NE = 10				# Number of presynaptic neurons
NT = NE+1        		# Total number of neurons (presynaptic + 1 postsynaptic)
tau_m = 50. 			# [ms] Membrane voltage time constant
R = 1. 				# [Ohm] Resistance of the leaky I-F model for excitatory neurons
t_max = 1.e6	 		# [ms] Max time of simulation
El = 0.				# [mV] Resting potential (leaky)
Vth = 10. 			# [mV] Threshold potential
Vres = 0. 			# [mV] Reset potential
Vspike = 10.			# [mV] Extra potential to indicate a spike

# Parameters for STDP (excitatory):
a0 = -0.00000 			# Activity independent term
a1_pre = 0.0 	 		# Pre-synaptic activity term (non-Hebbian)
a1_post = -0.0			# Post-synaptic activity term
a_plus = float(args[1])		# Coefficient related to pre->post activity
a_minus = float(args[2])	# Coefficient related to post->pre activity
tau1 = 8. 			# [ms] Decay time for pre->post activity
tau2 = 8. 			# [ms] Decay time for post->pre activity

# Simulation parameters ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dt = 1. 			# [ms] Simulation time step
numSteps = np.int(t_max/dt)	# Number of steps in simulation

# Connections (synapses) +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
EsynE = 30.			# [mV] Reversal potential for synaptic

# Synaptic conductance
tauSynEx = 15.			# [ms] Time constant for postsynaptic potential
gBarEx = 1. 			# Peak amplitude for EPSPs


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++ Define the parameters for the time dependent firing rate of presynaptic neurons +++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

tauFilt = 5.		# [ms] Time constant for filtering the Gaussian noise
Mean_FRate = 30.	# [Hz] Mean firing rate for the input neurons

# Define the training input
SpecialInput = 1.*np.ones(NE)
SpecialInput[3] = 2.

# ============================================================================================================
# ++++++++++++++++++++++ Calculate all the currents on the postsynaptic neuron +++++++++++++++++++++++++++++++
# ============================================================================================================


# Total synaptic current - vector with the total synaptic current --------------------------------------------
def Isyn(u_in,g_synE,W):
	alphaE = (-1)*g_synE*(u_in-EsynE)	# For excitatory synapses
	Itotal = np.sum(W*alphaE)		# For all the synapses
	return Itotal

# External current on the postsynaptic neuron ----------------------------------------------------------------

I_pre = 0.
def Iext(I_pre):
	I = I_pre - (I_pre - np.random.normal(0,50))/5.
	return I


# ============================================================================================================
# ++++++++++++++ Define the calculations to be used in each step of simulation +++++++++++++++++++++++++++++++
# ============================================================================================================

# Define new parameters for speeding up the simulation -------------------------------------------------------

step_tauSynEx = dt/tauSynEx
step_tau_m = dt/tau_m
step_tau1 = dt/tau1
step_tau2 = dt/tau2

A_plus = np.ones(NE)*a_plus
A_minus = np.ones(NE)*a_minus


# Simulation step (one integration step) +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def SimuStep(u_in,g_synE,Delta_pre_post,Delta_post_pre,W,t,I_pre,PreFiring_t):
	
	u_inPost = u_in	 		# Membrane potential for the postsynaptic neuron
	
	# Verify which input neurons fired at this step:
	pre_spikes_test_Ex = PreFiring_t
	
	# ============================== Synaptic conductance ==================================================
	# Update the synaptic conductances for the neurons that fired:
	g_synE_out = g_synE + gBarEx*pre_spikes_test_Ex
	# Evolve the synaptic conductances for all the neurons:
	g_synE_out = g_synE_out - g_synE_out*step_tauSynEx
	
	
	# ============================== Membrane Potential ====================================================
	I_now = Iext(I_pre)
	# Reset the potential for the neurons that fired and Evolve all the potentials:
	Isyn_total = Isyn(u_inPost,g_synE,W) # save the synaptic currents
	
	u_in_new = u_in - u_in*(u_in>Vth) + Vres*(u_in>Vth) 
	u_out = u_in_new + (-u_in_new + R*I_now + Isyn_total)*step_tau_m
	
	# For those that fired, make them fire!
	u_out = u_out + Vspike*(u_out>Vth)
	
	
	# ======================= Synaptic traces for STDP =====================================================
	Delta_pre_post_out = Delta_pre_post - Delta_pre_post*step_tau2 + (u_inPost>Vth)
	Delta_post_pre_out = Delta_post_pre - Delta_post_pre*step_tau1 + pre_spikes_test_Ex
	
	
	# =============================== Synaptic weight ======================================================
	
	# Define the vectors to update the synaptic weights vector (excitatory)
	
	W_out = W + Delta_pre_post*pre_spikes_test_Ex*A_minus \
		+ Delta_post_pre*(u_inPost>Vth)*A_plus
	
	# Use hard bounds to assure that 0 < W < 2
	up_bound = 2.
	W_out = W_out - (W_out - up_bound)*(W_out>up_bound) - (W_out)*(W_out<0.)
	 
	return u_out , g_synE_out, Delta_pre_post_out, Delta_post_pre_out , W_out, I_now



# ============================================================================================================
# +++++++++++++++++++++++++++++++++++++++ Run the main program +++++++++++++++++++++++++++++++++++++++++++++++
# ============================================================================================================

subsampling = 5000

def OneTrial(void):
	# Function to run the simulation for one trial
	
	# ====================================================================================================
	# +++++++++++++++++++++++++++++++++ Initialization of variables ++++++++++++++++++++++++++++++++++++++
	# ====================================================================================================
	
	
	Delta_pre_post = 0.		# Synaptic traces for pre-post activity
	Delta_post_pre = np.zeros(NE)	# Synaptic traces for post-pre activity
	Vmemb = 0.			# [mV] Initial membrane potential for the postsynaptic neuron
	W = np.load('W0_new.npy')	# Initial synaptic weights
	g_synE = np.zeros(NE) 		# [mS] Initial value for the g function 
	
	I_pre = 0.
	
	Ws = np.zeros((int(numSteps/subsampling)+1,NE))# Matrix to track the evolution of the weight matrix
	Ws[0,:] = W
	
	
	# ====================================================================================================
	# +++++++++++++++++++++++ Generation of presynaptic spike trains +++++++++++++++++++++++++++++++++++++
	# ====================================================================================================
	
	np.random.seed() # to ensure randomness when using multiple processes
	
	# Generate NE independent filtered gaussian noise series
	FilteredGaussianNoise = FGN.FiltGNoise(Mean_FRate,tauFilt,NE,numSteps,dt)
	FilteredGaussianNoise = FilteredGaussianNoise*SpecialInput
	# Re-scale the Filtered Gaussian noise to the time bin (in order to calculate the probability of firing):
	FilteredGaussianNoise = FilteredGaussianNoise*dt
	# Create correlation between the inputs ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	FilteredGaussianNoiseFinal =  1.*FilteredGaussianNoise
	sigma = .5
	for i in range(NE):
		NormDist = 1./np.sum(np.array([np.exp(-(x-i)**2/((2.*sigma)**2)) for x in range(NE)]))
		InputWeights = NormDist*np.array([np.exp(-(x-i)**2/((2.*sigma)**2)) for x in range(NE)])
		FilteredGaussianNoiseFinal[:,i] = np.dot(FilteredGaussianNoise,InputWeights) 
	
	# Finally generate the presynaptic spike trains
	PreFiring = np.random.random((numSteps,NE)) < FilteredGaussianNoiseFinal
	
	
	# ====================================================================================================
	# +++++++++++++++++ Calculate the evolution of the variables +++++++++++++++++++++++++++++++++++++++++
	# ====================================================================================================
	
	for l in range(numSteps-1):
		time = dt*l
		Vmemb,  g_synE,  Delta_pre_post, Delta_post_pre, W , I_pre\
			= SimuStep(Vmemb,g_synE,Delta_pre_post,Delta_post_pre,W,time,I_pre,PreFiring[l])
		
		if np.mod(l,subsampling)==0:
			Ws[int(l/subsampling)+1,:] = W
	
	
	return Ws 



# Using parallel computing
NTrials = 20
quant_proc = np.max((mp.cpu_count()-1,1))
pool = mp.Pool(processes=quant_proc)

# run the main code for NTrials trials -----------------------------------------------------------------------
print('\n Running simulation (training input):')
print(' a_plus = {0:.4f} \t a_minus = {1:.4f}'.format(a_plus,a_minus))

AllTrials = pool.map(OneTrial,np.zeros(NTrials))
WsAllTrials = np.array([AllTrials[i] for i in range(NTrials)])

# save the data ----------------------------------------------------------------------------------------------
np.save('Data_RF_shift/Syn_weights_all_trials_aplus={0:07.4f}_aminus={1:07.4f}'.format(a_plus,a_minus,subsampling),WsAllTrials)


time_out = time_now()
time_total = time_out - time_in

print(' Total time = {0:.3f} segundos'.format(time_total))
