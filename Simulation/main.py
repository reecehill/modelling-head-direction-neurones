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
#figure2 = pf.plotSampledNeuroneWeightDistributions(neuronalPopulation)

# Plot matrix of noiseless weights
#figure3 = pf.plotWeightDistribution(neuronalPopulation.getAllWeights(), title="static model without noise")
#figure4 = pf.solveDuDt(neuronalPopulation, title="static model without noise")
print("Neuronal Population (noiseless) completed: "+str(time.time() - start)+' seconds')




# Copy the neuronal population, and inject noise into its weights
#neuronalPopulation_noise = deepcopy(neuronalPopulation).injectNoiseIntoWeights(meanOfNoise=0)
#figure5 = pf.plotWeightDistribution(neuronalPopulation_noise.getAllWeights(
#), hasNoise=True, title="static model with noise")
#figure6 = pf.solveDuDt(neuronalPopulation_noise, title="static model with noise")
#print("Neuronal Population (noisy) completed: " + str(time.time() - start)+' seconds')
#
## Now begin to make the model dynamic - add the odd weights to noiseless population.
#neuronalPopulation_dynamic = deepcopy(neuronalPopulation).setupOddWeights()
#figure7 = pf.plotZerothNeuroneOddAndEvenWeights(neuronalPopulation_dynamic)
#figure8 = pf.plotWeightDistribution(
#    neuronalPopulation_dynamic.getAllWeights(),  title="dynamic model")
#figure9 = pf.solveDuDt(neuronalPopulation_dynamic, title="dynamic model")
#print("Neuronal Population (dynamic, sinusoidal) completed: " + str(time.time() - start)+' seconds')

# Take a static population, add add a local-view detector input and plot output.
neuronalPopulation_static2 = deepcopy(neuronalPopulation)
figure10 = pf.plotEffectOfAdditionalTimedUInput(neuronalPopulation)


# Copy parameters.py to output directory.
copyfile(p.cwd+'/parameters.py', str(p.outputDirectory)+'/parameters.py')
print("Finished, total time elapsed: "+str(time.time() - start)+' seconds')
print("Now showing graphs...")
showFigures()
