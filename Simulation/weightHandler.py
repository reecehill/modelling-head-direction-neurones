import numpy as np
import parameters as p
import mathEquations as e

import matplotlib.pyplot as plt
from sys import exit

def generateWeightsForOneNeurone(theta_0):
    # For this function, see: Section 4.3, Synaptic Weight Distribution

    # Get the average firing rates, f, of this neurone, at each head direction, theta.
    firingRatesByTheta = e.getTuningCurve(theta_0)

    # Get the mean net input into this neurone, at each head direction given the respective firing rates.
    # NOTE: Here, we use the principle: inputCurrent(u) -> neuronesOfSharedPreference(sigma) -> outputCurrent(f)
    netInputsByTheta = e.getU(firingRatesByTheta)

    # Transform by Fast Fourier Transformations
    firingRatesByThetaFft = np.fft.fft(firingRatesByTheta)
    netInputsByThetaFft = np.fft.fft(netInputsByTheta)

    # Find squared firing rates (f)
    # TODO: Parrivesh finds abs before squaring, is this necessary?
    firingRatesByThetaFft_squared = np.square(
        np.absolute(firingRatesByThetaFft))

    # Calculate the amount by which large weights will be penalised.
    penaltyForMagnitude = p.penaltyForMagnitude_0 * \
        firingRatesByThetaFft_squared.max()

    # !!!!!
    # DEBUGGING
    # In order to run this script, you must run main.py
    # The script will only run the below code once, then all execution will be stopped (so you won't get 300+ figures!)

    u_hat = netInputsByThetaFft
    f_hat = firingRatesByThetaFft
    Lambda = p.penaltyForMagnitude_0 * \
        np.square(np.abs(firingRatesByThetaFft)).max()
    w_hat = np.divide((np.matmul(u_hat, f_hat)),
                      (Lambda + np.square(np.abs(f_hat))))
    # Compute weight matrix in Fourier ?space.
    # Taken from Section 4.3, Synaptic Weights Distribution, in-text, final equation.
    numerator = np.multiply(netInputsByThetaFft, firingRatesByThetaFft)
    denominator = penaltyForMagnitude + firingRatesByThetaFft_squared
    weightsFft = np.divide(numerator, denominator)
    plt.figure()
    plt.title('netInputsByTheta')
    plt.plot(np.linspace(
        p.thetaSeries[0], p.thetaSeries[-1], p.numberOfUnits), netInputsByTheta)

    plt.figure()
    plt.title('firingRatesByThetaFft')
    plt.plot(np.linspace(
        p.thetaSeries[0], p.thetaSeries[-1], p.numberOfUnits), firingRatesByThetaFft)

    plt.figure()
    plt.title('weightsFft')
    plt.plot(np.linspace(
        p.thetaSeries[0], p.thetaSeries[-1], p.numberOfUnits), weightsFft)

    plt.figure()
    plt.title('w_hat')
    plt.plot(np.linspace(
        p.thetaSeries[0], p.thetaSeries[-1], p.numberOfUnits), np.abs(w_hat))
    plt.show()
    exit() # No code will run after this.

    # !!!!!
    # END DEBUGGING


    # Convert weights into ?normal domain.
    # TODO: Notice that we get rid of the imaginary part. Is this right? It matches result of Parrivesh, so assume it's a difference in Python.
    weights = np.fft.ifft(weightsFft).real

    # Scaling factor weights*N
    # NOTE: This may actually be weights/N
    return weights*p.numberOfUnits


def injectNoise(weightsForAllNeurones):
  meanOfNoise = 0
  stdOfNoise = p.epsilon * np.mean(np.abs(weightsForAllNeurones))
  noise = p.randomGenerator.normal(
      loc=meanOfNoise, scale=stdOfNoise, size=weightsForAllNeurones.shape)
  weightsForAllNeurones_Noise = weightsForAllNeurones + noise
  return weightsForAllNeurones_Noise
