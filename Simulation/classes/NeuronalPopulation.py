import parameters as p
import weightHandler as wh
import numpy as np
from classes.Neurone import Neurone

class NeuronalPopulation:
  def __init__(self):
      # It is possible to generate neurones in one of two ways. See parameters.py for more information.
      if(p.generateWeightsMethod == 'independentNeurone'):
        # Get each neurone's weight independently by looping through all theta values.
        self.neurones = np.array([Neurone(theta_0=theta, evenWeights=wh.generateWeightsForOneNeurone(
            theta_0=theta)) for theta in p.thetaSeries])
        
      elif(p.generateWeightsMethod == 'templateNeurone'):
      # Instantiate a neurone with preferred direction 0, then roll its weights to offset them according to thetaIndex.
        self.neurones = np.array([Neurone(theta_0=theta, evenWeights=np.roll(wh.generateWeightsForOneNeurone(theta_0=p.actualTheta), thetaIndex)) for thetaIndex, theta in enumerate(p.thetaSeries)])

  def getAllWeights(self):
    # Returns w(θ,t)
    # See: Equation 2. 
    return np.array(np.add([neurone.evenWeights for neurone in self.neurones],[neurone.oddWeights for neurone in self.neurones]))