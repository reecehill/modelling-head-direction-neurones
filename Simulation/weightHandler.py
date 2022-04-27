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
    firingRatesByThetaFft_squared_max = firingRatesByThetaFft_squared.max()

    #firingRatesByThetaFft_squared_max = 5.952236665565973e+04
    # Calculate the amount by which large weights will be penalised.
    penaltyForMagnitude = p.penaltyForMagnitude_0 * \
        firingRatesByThetaFft_squared_max

    # Compute weight matrix in Fourier ?space.
    # Taken from Section 4.3, Synaptic Weights Distribution, in-text, final equation.
    numerator = np.multiply(netInputsByThetaFft,
                            firingRatesByThetaFft).astype('complex64')
    denominator = (penaltyForMagnitude +
                   firingRatesByThetaFft_squared).astype('complex64')
    weightsFft = np.divide(numerator, denominator, casting='no')
    np.savetxt(p.outputDirectory+'/weightFFt.csv', weightsFft, delimiter=',')

    # Convert weights into ?normal domain.
    # TODO: Notice that we get rid of the imaginary part. Is this right? It matches result of Parrivesh, so assume it's a difference in Python.
    # Scaling factor weights*N

    # NOTE: Scaling factor is removed and yet it matches data of Markus.
    weights = (fft.ifft(weightsFft, axis=0))

    # This is not a parameter as is for debugging only.
    if(neurone.theta_0 == 5):
        showPlotForWeightsFft = False
        if(showPlotForWeightsFft):
            plt.figure()
            fig, axArray = plt.subplots(2, 2)
            plt.sca(axArray[0, 0])
            plt.title('netInputsByTheta')
            plt.plot(p.thetaSeries, netInputsByTheta,
                     label='netInputsByTheta', marker='o')
            filename = p.outputDirectory+'/../../expectedUTargetLambda1e-3.csv'
            u_target = np.genfromtxt(
                filename, delimiter=',', dtype=complex, skip_header=1).T[0]
            if(netInputsByTheta.size == u_target.size):
                plt.plot(p.thetaSeries, u_target, label='u_target')
            plt.legend()

            plt.sca(axArray[0, 1])
            plt.title('firingRatesByThetaFft')
            filename = p.outputDirectory+'/../../expectedFHatLambda1e-3.csv'
            f_hat = np.genfromtxt(filename, delimiter=',',
                                  dtype=complex, skip_header=1).T[0]
            plt.plot(p.thetaSeries, firingRatesByThetaFft.real,
                     label='firingRatesByThetaFft', marker='o')
            if(firingRatesByThetaFft.size == f_hat.size):
                plt.plot(p.thetaSeries, f_hat.real, label='f_hat')
            plt.legend()

            plt.sca(axArray[1, 0])
            plt.title('out and weightsFft')
            filename = p.outputDirectory+'/../../expectedWeightsFftLambda1e-3.csv'
            out = np.genfromtxt(filename, delimiter=',',
                                dtype=complex, skip_header=1).T[0]
            plt.plot(p.thetaSeries, np.abs(weightsFft.real),
                     label='weightsFft', marker='o')
            if(weightsFft.size == out.size):
                plt.plot(p.thetaSeries, np.abs(out.real), label='output')
            plt.legend()

            plt.sca(axArray[1, 1])
            plt.title('weights versus w_mat[0]: -1e-3 only')
            filename = p.outputDirectory+'/../../expectedWeightsLambda1e-3.csv'  # no e5 available!
            weights_mat = np.genfromtxt(
                filename, delimiter=',', dtype=np.float64, skip_header=1)
            weights_vector = weights_mat[0]
            plt.plot(p.thetaSeries, weights.real,
                     label='weights)', marker='o')
            if(weights.size == weights_vector.size):
                plt.plot(p.thetaSeries, weights_vector, label='w_mat[0]')
            plt.legend()

            plt.suptitle(r"$\theta_0$" "=%f," r" $\lambda_0$" "=%e," r" N" "=%d" % (
                neurone.theta_0, p.penaltyForMagnitude_0, p.numberOfUnits))
            plt.show()
            exit()

    return weights.real
