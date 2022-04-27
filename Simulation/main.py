import mathEquations as e
from copy import deepcopy
import time
start = time.time()
import parameters as p
import plottingFunctions as pf
import populationHandler as ph
import weightHandler as wh
from matplotlib.pyplot import show as showFigures
from matplotlib.pyplot import close as closeAllFigures
from shutil import copyfile
import numpy as np

closeAllFigures('all')

# Plot curves before simulation
figure0 = pf.plotSigmoidFunction()
figure1 = pf.plotTuningCurve()


# Generate a population of neurones with weights as if there were no head movement (θ_odd=0) 
neuronalPopulation = ph.generatePopulation()
figure2 = pf.plotSampledNeuroneWeightDistributions(neuronalPopulation)

# Plot matrix of noiseless weights
figure3 = pf.plotWeightDistribution(neuronalPopulation.getAllWeights())
figure4 = pf.solveDuDt(neuronalPopulation)

# Copy the neuronal population, and inject noise into its weights
neuronalPopulation_noise = deepcopy(neuronalPopulation).injectNoiseIntoWeights(meanOfNoise=0)
figure5 = pf.plotWeightDistribution(neuronalPopulation_noise.getAllWeights(), hasNoise=True)
figure6 = pf.solveDuDt(neuronalPopulation_noise)

# Now begin to make the model dynamic - add the odd weights to noiseless population.
neuronalPopulation_dynamic = deepcopy(neuronalPopulation).setupOddWeights()
figure7 = pf.plotWeightDistribution(
    neuronalPopulation_dynamic.getAllWeights())
figure8 = pf.solveDuDt(neuronalPopulation_dynamic)

# Copy parameters.py to output directory.
copyfile(p.cwd+'/parameters.py', str(p.outputDirectory)+'/parameters.py')
print("Finished, time elapsed: "+str(time.time() - start)+' seconds')
print("Now showing graphs...")
showFigures()
