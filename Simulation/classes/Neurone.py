import numpy as np
import mathEquations as e
import parameters as p
import weightHandler as wh
import matplotlib.pyplot as plt


class Neurone:
  def __init__(self, theta_0, isTemporaryNeurone=False):
    self.isTemporaryNeurone = isTemporaryNeurone
    self.theta_0 = theta_0
    self.tuningCurve = e.getTuningCurve(theta_0=theta_0)
    self.evenWeights = wh.generateWeightsForOneNeurone(neurone=self)
    # Begin with oddWeights as zeros, and get them later.
    self.oddWeights = np.zeros(self.evenWeights.shape)
    
    
    # Create a vector of zeros. This stores the f (firing rate, in Hz) of each neurone over time.
    self.firingActivity = np.zeros(p.timeSeries.size)
    
    # Average net input received by neurone, over time.
    self.uActivity = np.zeros(p.timeSeries.size)

  def getWeights(self):
    return (self.evenWeights + self.oddWeights)
  
  def rollWeights(self, byAmount):
    unrolledWeights = self.evenWeights
    rolledWeights = np.roll(unrolledWeights, byAmount)
    self.evenWeights = rolledWeights
    self.tuningCurve = np.roll(self.tuningCurve, byAmount)
    return self
  
  def setOddWeights(self, temporaryNeurone={}):
    self.oddWeights = e.getOddWeights(self, temporaryNeurone)
    return self