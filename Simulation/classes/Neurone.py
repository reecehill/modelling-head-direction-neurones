import numpy as np
import mathEquations as e
import parameters as p

class Neurone:
  def __init__(self, theta_0, evenWeights):
    self.theta_0 = theta_0
    self.evenWeights = evenWeights
    self.oddWeights = np.zeros(evenWeights.shape)
    self.tuningCurve = e.getTuningCurve(theta_0=theta_0)
    
    # Create a matrix of zeros. The first dimension is the time (t), and the second are the weights (even and odd), the third are respective matrices. 
    self.firingActivity = np.zeros((len(p.timeSeries), 2, evenWeights.size))
    self.firingActivity[0][0] = self.evenWeights
    self.firingActivity[0][1] = self.oddWeights
  def getWeights(self):
    return self.evenWeights + self.oddWeights
