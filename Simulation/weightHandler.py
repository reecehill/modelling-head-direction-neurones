import numpy as np
from numpy import fft as fft
from pandas import read_csv
import parameters as p
import mathEquations as e

import matplotlib.pyplot as plt
from sys import exit


def generateWeightsForOneNeurone(neurone):
    # For this function, see: Section 4.3, Synaptic Weight Distribution

    # Get the average firing rates, f, of this neurone, at each head direction, theta.
    firingRatesByTheta = neurone.tuningCurve

    # Get the mean net input into this neurone, at each head direction given the respective firing rates.
    # NOTE: Here, we use the principle: inputCurrent(u) -> neuronesOfSharedPreference(sigma) -> outputCurrent(f)
    netInputsByTheta = e.getInverseSigmoid(firingRatesByTheta)

    # Transform by Fast Fourier Transformations
    firingRatesByThetaFft = fft.fft(
        firingRatesByTheta, axis=0).astype(np.complex128)
    netInputsByThetaFft = fft.fft(
        netInputsByTheta, axis=0).astype(np.complex128)

    # Find squared firing rates (f)
    # TODO: Parrivesh finds abs before squaring, is this necessary?
    firingRatesByThetaFft_squared = np.square(
        np.abs(firingRatesByThetaFft, dtype=np.float64))

    # Calculate the amount by which large weights will be penalised.
    f_hat_max2 = firingRatesByThetaFft_squared.max()
    penaltyForMagnitude = p.penaltyForMagnitude_0 * \
        f_hat_max2

    # !!!!!
    # DEBUGGING
    # In order to run this script, you must run main.py
    # The script will only run the below code once, then all execution will be stopped (so you won't get 300+ figures!)

    u_hat = netInputsByThetaFft
    f_hat = firingRatesByThetaFft
    Lambda = p.penaltyForMagnitude_0 * \
        np.square(np.abs(firingRatesByThetaFft)).max()
    out = np.divide((np.multiply(u_hat, f_hat)),
                    (Lambda + np.square(np.abs(f_hat))))
    # Compute weight matrix in Fourier ?space.

    # Taken from Section 4.3, Synaptic Weights Distribution, in-text, final equation.
    numerator = np.multiply(netInputsByThetaFft, firingRatesByThetaFft).astype('complex64')
    denominator = (penaltyForMagnitude + \
        firingRatesByThetaFft_squared).astype('complex64')
    #weightsFft = numerator / denominator
    weightsFft = np.divide(numerator, denominator, casting='no')
    np.savetxt(p.outputDirectory+'/weightFFt.csv', weightsFft, delimiter=',')

#
 #   plt.figure()
 #   plt.title('w_hat')
 #   plt.plot(np.linspace(
 #       p.thetaSeries[0], p.thetaSeries[-1], p.numberOfUnits), np.abs(w_hat))
 #   plt.show()
 #   exit() # No code will run after this.

    # !!!!!
    # END DEBUGGING

    # Convert weights into ?normal domain.
    # TODO: Notice that we get rid of the imaginary part. Is this right? It matches result of Parrivesh, so assume it's a difference in Python.
    # Scaling factor weights*N
    # NOTE: This may actually be weights/N
    weights = (fft.ifft(weightsFft, axis=0)) * p.numberOfUnits

    plt.figure()
    fig, axArray = plt.subplots(2, 2)
    plt.sca(axArray[0, 0])
    plt.title('netInputsByTheta')
    plt.plot(np.linspace(
        p.thetaSeries[0], p.thetaSeries[-1], p.numberOfUnits), netInputsByTheta)

    plt.sca(axArray[0, 1])
    plt.title('firingRatesByThetaFft')
    plt.plot(np.linspace(
        p.thetaSeries[0], p.thetaSeries[-1], p.numberOfUnits), firingRatesByThetaFft)

    plt.sca(axArray[1, 0])
    plt.title('weightsFft')
    plt.plot(p.thetaSeries, np.absolute(
        weightsFft, dtype=np.float64))

    plt.sca(axArray[1, 1])
    plt.title('out and weightsFft')
    filename = p.outputDirectory+'/../out.csv'
    data = np.genfromtxt(filename, delimiter=',', dtype=complex).T[0]
    plt.suptitle(neurone.theta_0)
    plt.plot(p.thetaSeries, np.abs(weightsFft.real), label='weightsFft')
    plt.plot(p.thetaSeries, np.abs(data.real), label='output')
    plt.legend()
    plt.show()
    exit()

    return weights


def injectNoise(weightsForAllNeurones):
    meanOfNoise = 0
    stdOfNoise = p.epsilon * np.mean(np.abs(weightsForAllNeurones))
    noise = p.randomGenerator.normal(
        loc=meanOfNoise, scale=stdOfNoise, size=weightsForAllNeurones.shape)
    weightsForAllNeurones_Noise = weightsForAllNeurones + noise
    return weightsForAllNeurones_Noise
