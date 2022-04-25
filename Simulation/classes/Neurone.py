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
    
    
    # Create a matrix of zeros. The first dimension is the time (t), and the second are the weights (even and odd), the third are respective matrices. 
    self.firingActivity = np.zeros(
        (len(p.timeSeries), 2, self.evenWeights.size))
    self.firingActivity[0][0] = self.evenWeights
    self.firingActivity[0][1] = self.oddWeights

  def getWeights(self):
    return self.evenWeights + self.oddWeights
