import numpy as np
import parameters as p

# Returns firing rate, f, as a function of head direction, theta.
# It has a Guassian-like shape.
def getTuningCurve(theta_0):
  # See: Equation 1
  
  # For every possible theta, get the difference between it and this neurone's preferred angle.
  anglesInDegrees = p.thetaSeries - theta_0
  # Convert angles to radians as required by np.cos()
  anglesInRadians = np.radians(anglesInDegrees)
  
  f = p.A + p.B*np.exp(p.K * np.cos(anglesInRadians))
  return f


# Returns the output of a single neuronal unit (neurone), sigma, as a function of its input, x. 
# Could be considered the "activation" function of a neurone?
# Is an adaptation of the conventional sigmoid function, for reasons explained in section 4.2
def getSigmoid(x):
  # See: Equation 4
  sigma = p.alpha * np.log( (1 + np.exp(p.b * (x + p.c))) ** p.beta)
  return sigma



# Returns the input given to a single neuronal input (neurone), x, given its output, sigma.
# Is the inverse of getSigmoid().
def getInverseSigmoid(sigma):
  # Mathematically derived, by Parrivesh, from Equation 4.
  x = 1/p.b * np.log( np.exp(( sigma/p.alpha )**(1/p.beta)) -1) - p.c
  return x

def getDuDt(u, t, w, f):
  # TODO: Confirm, is f() the sigmoid function here? This is not clear in the paper.
  # See: Equation 2. 
  
  #f = getSigmoid(u);
  duDt = 1/p.tau * ( -u + np.matmul(w,f) )
  return duDt


def getF(u):
  # See: Section 3, Basic Dynamic Model - in-text
  # TODO: Returns the "average firing rate of all neurones that share the same preferred direction"  
  # NOTE: inputCurrent(u) -> neuronesOfSharedPreference(sigma) -> outputCurrent(f)
  f = getSigmoid(u)
  return f
  
def getU(f):
  # See: Section 3, Basic Dynamic Model - in-text
  # NOTE: inputCurrent(u) -> neuronesOfSharedPreference(sigma) -> outputCurrent(f)
  # TODO: Returns the "average net inputs/synaptic currents" for neurones that share the same preferred direction. Is this true?
  u = getInverseSigmoid(f)
  return u
