import parameters as p
import weightHandler as wh
import numpy as np
from classes.Neurone import Neurone


class NeuronalPopulation:
    def __init__(self, thetaOffset=0):
        # ThetaOffset is used to add gamma to generate a temporary population.
        # It is possible to generate neurones in one of two ways. See parameters.py for more information.
        if(p.generateWeightsMethod == 'independentNeurone'):
            # Get each neurone's weight independently by looping through all theta values.
            self.neurones = np.array([Neurone(theta_0=theta) for theta in p.thetaSeries])
            print("Weights generated")

        elif(p.generateWeightsMethod == 'templateNeurone'):
            # Instantiate a neurone with preferred direction 0, then roll its weights to offset them according to thetaIndex.
            self.neurones = np.array([Neurone(theta_0=5+thetaOffset)
                                     for theta in p.thetaSeries])

            for neuroneIndex, neurone in enumerate(self.neurones):
                neurone.theta_0 = p.thetaSeries[neuroneIndex]
                neurone.rollWeights(neuroneIndex)

    def getAllWeights(self):
        # Returns w(θ,t)
        # See: Equation 2.
        summedWeights = [np.add(neurone.evenWeights, neurone.oddWeights)
                         for neurone in self.neurones]
        return np.array(summedWeights)

    def injectNoiseIntoWeights(self, meanOfNoise):
        stdOfNoise = p.epsilon * np.mean(np.abs(self.getAllWeights()))
        noise = p.randomGenerator.normal(
            loc=meanOfNoise, scale=stdOfNoise, size=self.getAllWeights().shape)
        for neuroneIndex, neurone in enumerate(self.neurones):
            neurone.evenWeights = neurone.evenWeights + noise[neuroneIndex]
        return self

    def setupOddWeights(self):
        if(p.oddWeightFunction=='sinusoid'):
            [neurone.setOddWeights() for neurone in self.neurones]

        elif(p.oddWeightFunction == 'derivative'):
            temporaryNeuronalPopulation = NeuronalPopulation(
                thetaOffset=p.gamma)
            [neurone.setOddWeights(temporaryNeuronalPopulation.neurones[neuroneIndex]) for neuroneIndex, neurone in enumerate(self.neurones)]
       
        return self
