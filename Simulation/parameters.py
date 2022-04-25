import automatedParameters
import numpy as np

# ----------START------------
# SIMULATION PARAMETERS
# ---------------------------


# Weights for each neurone can be generated in one of two ways:
# 1) 'templateNeurone': By copying the weights of a neurone with theta_0=0 and then "rolling" the weights. This leads to a matrix diagonal that is equal.
# 2) 'independentNeurone': By generating weights between neurones independent of each other.
# NOTE: Option 2 leads to bizarre behaviour.  
generateWeightsMethod = 'templateNeurone'

# Network size
numberOfUnits = (360) + 1


f_max = 40 #Hz

# Actual theta of current head direction (in degrees)
actualTheta = 0

# Taken from Section 3, Basic Dynamic Model, in-text.
# Time steps in
tau = 10 #msec

# Total time to simulate for
totalSimulationTime = 800  # msec

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
penaltyForMagnitude_0 = 10**(-2)

# Taken from Figure 5
epsilon = 0.1

# ---------------------------
# WEIGHT DISTRIBUTION PARAMETERS
# -----------END-------------


# //////////////////////////////////

# Do not edit below this line
outputDirectory, cwd, randomGenerator, thetaSeries, timeSeries = automatedParameters.generate()
