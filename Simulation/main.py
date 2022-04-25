import time
start = time.time()
import parameters as p
import plottingFunctions as pf
import populationHandler as ph
import weightHandler as wh
from matplotlib.pyplot import show as showFigures
from matplotlib.pyplot import close as closeAllFigures
from shutil import copyfile
from scipy.integrate import ode
import numpy as np

closeAllFigures('all')

figure1 = pf.plotTuningCurve()

# Generate a population of neurones with weights as if there is no head movement (θ_odd=0) 
neuronalPopulation = ph.generatePopulation()
figure3 = pf.plotSampledNeuroneWeightDistributions(neuronalPopulation)



# Gather the neuronal population's weights into a matrix.
weightsForAllNeurones = neuronalPopulation.getAllWeights()
figure2 = pf.plotWeightDistribution(weightsForAllNeurones)
test = pf.plotTest(neuronalPopulation)

# Inject noise into this neuronal population
weightsForAllNeurones_Noise = wh.injectNoise(weightsForAllNeurones)
figure3 = pf.plotWeightDistribution(weightsForAllNeurones_Noise, hasNoise=True)




# Copy parameters.py to output directory.
copyfile(p.cwd+'/parameters.py', str(p.outputDirectory)+'/parameters.py')
print("Finished, time elapsed: "+str(time.time() - start)+' seconds')
print("Now showing graphs...")
showFigures()
