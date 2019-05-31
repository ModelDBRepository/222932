# Defines the function to create a time-filtered Gaussian noise.
#
# FiltGNoise(Mean_FRate,tauFilt,N,numSteps,dt):
# 1. [Hz] Mean firing rate
# 2. [ms] Filtering time constant
# 3. Number of neurons
# 4. Number of steps
# 5. [ms] time step

import numpy as np
def FiltGNoise(Mean_FRate,tauFilt,N,numSteps,dt):
	GaussianNoise = np.random.normal(0,1,(numSteps,N))	# Gaussian noise
	FilteredGaussianNoise = GaussianNoise*0.		# Initialization of the filtered Gaussian noise
	
	
	# Make the time-filtered Gaussian noise signal
	step_tauFilt = dt/tauFilt # just to make the simulation faster
	for t in range(numSteps)[1:]:
		FilteredGaussianNoise[t] = FilteredGaussianNoise[t-1] - (FilteredGaussianNoise[t-1]-GaussianNoise[t-1])*step_tauFilt
	
	# Normalize it to a maximum value of 1:
	for i in range(N):
		FilteredGaussianNoise[:,i] = FilteredGaussianNoise[:,i]/np.max(FilteredGaussianNoise[:,i])
	
	
	# Apply the rate threshold:
	rth = 0.2
	FilteredGaussianNoise = FilteredGaussianNoise - rth
	# Rectify it:
	FilteredGaussianNoise[FilteredGaussianNoise<0.] = 0.
	
	
	# Make the mean firing rate equal to Mean_FRate:
	for i in range(N):
		FilteredGaussianNoise[:,i] = FilteredGaussianNoise[:,i]/np.mean(FilteredGaussianNoise[:,i])*(Mean_FRate/1000.)
	
	return FilteredGaussianNoise