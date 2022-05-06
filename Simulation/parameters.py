import automatedParameters
import numpy as np

# ----------START------------
# SIMULATION PARAMETERS
# ---------------------------


# Weights for each neurone can be generated in one of two ways:
# 1) 'templateNeurone': By copying the weights of a neurone with theta_0=0 and then "rolling" the weights. This leads to a matrix diagonal that is equal.
# 2) 'independentNeurone': By generating weights between neurones independent of each other.
# NOTE: Option 2 leads to bizarre behaviour. (i.e., a weight matrix that is not equal along the diagonal)
generateWeightsMethod = 'templateNeurone'

# Network size
# +1 is added here to account for indexing differences between Matlab and Python.
# NOTE: To copy the results of the paper, ensure this is set to 37. Results are bizarre otherwise.
numberOfUnits = (360)+1

f_max = 40  # Hz


# Taken from Section 3, Basic Dynamic Model, in-text.
# Time steps in
tau = 20  # msec

# Total time to simulate for
totalSimulationTime = 1200  # msec


# Possible options
# noise
# tuningCurve
# steadyState
# slightlyAwayFromSteadyState
initialCondition = 'tuningCurve'

# ----------END------------
# SIMULATION PARAMETERS
# ---------------------------


# //////////////////////////////////


# ----------START------------
# (INV) SIGMOID FUNCTION PARAMETERS
# ---------------------------

# Taken from Figure 4.
# "Determined by scaling condition: sigma(1-c) = f_max = 40Hz"
# NOTE: Is referenced in paper as both alpha and a?!
alpha = 6.34

# Taken from Figure 4.
beta = 0.8

# Taken from Figure 4.
b = 10

# Taken from Figure 4.
c = 0.5

# ---------------------------
# (INV) SIGMOID FUNCTION PARAMETERS
# -----------END-------------


# //////////////////////////////////


# ----------START------------
# TUNING CURVE PARAMETERS
# ---------------------------

# TODO: Taken from ?
K = 8

# TODO: Taken from ?
# To produce Figure 2...
#A = 2.53
A = 1

# TODO: Taken from ?
# To produce Figure 2...
# B = 34.8/np.exp(K)
# To produce Figure ?...
B = (f_max - A)/np.exp(K)


# ---------------------------
# TUNING CURVE PARAMETERS
# -----------END-------------


# //////////////////////////////////

# ----------START------------
# WEIGHT DISTRIBUTION PARAMETERS
# ---------------------------

# NOTE: penaltyForMagnitude = lambda in paper, but the word is reserved in Python.
penaltyForMagnitude_0 = 1e-03

# Taken from Figure 5
epsilon = 0.06

# Used to generate odd weights
# Options:
# sinusoid
# derivative
oddWeightFunction = 'derivative'

# TODO: Is this the same as alpha as defined in Figure 4>
alphaSinusoid = 0.0037

# Used for odd weights
gamma = np.rad2deg(-0.063)

# ---------------------------
# WEIGHT DISTRIBUTION PARAMETERS
# -----------END-------------


# //////////////////////////////////

# Do not edit below this line
outputDirectory, cwd, randomGenerator, thetaSeries, timeSeries = automatedParameters.generate()
