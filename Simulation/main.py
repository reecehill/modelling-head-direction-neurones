import parameters as p
import commonFunctions as f
import plottingFunctions as pf
import weightHandler as w
from matplotlib.pyplot import show as showFigures
from matplotlib.pyplot import close as closeAllFigures
from shutil import copyfile

import numpy as np

closeAllFigures()

figure1 = pf.plotTuningCurve()

# Generate a population of neurones that have unique, discretised directions
weightsForAllNeurones = w.generateWeightsForAllNeurones()
figure2 = pf.plotWeightDistribution(weightsForAllNeurones)

# Inject noise into this neuronal population
weightsForAllNeurones_Noise = w.injectNoise(weightsForAllNeurones)
figure3 = pf.plotWeightDistribution(weightsForAllNeurones_Noise, hasNoise=True)


# Copy parameters.py to output directory.
copyfile(p.cwd+'/parameters.py', str(p.outputDirectory)+'/parameters.py')


showFigures()
