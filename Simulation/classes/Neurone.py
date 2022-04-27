import numpy as np
import mathEquations as e
import parameters as p
import weightHandler as wh
class Neurone:
  def __init__(self, theta_0):
    self.theta_0 = theta_0
    self.tuningCurve = e.getTuningCurve(theta_0=theta_0)
    self.evenWeights = wh.generateWeightsForOneNeurone(neurone=self)
    self.oddWeights = np.zeros(self.evenWeights.shape)
    
    
    # Create a vector of zeros. This stores the f (firing rate, in Hz) of each neurone over time.
    self.firingActivity = np.zeros(p.timeSeries.size)
    
    # Average net input received by neurone, over time.
    self.uActivity = np.zeros(p.timeSeries.size)

  def getWeights(self):
    return self.evenWeights + self.oddWeights
  
  def rollWeights(self, byAmount):
    self.evenWeights = np.roll(self.evenWeights, byAmount)
    return self
  
  def setOddWeights(self):
    self.oddWeights = e.getOddWeights(self.evenWeights)
    return self