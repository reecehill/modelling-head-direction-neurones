import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pl
import parameters as p
import mathEquations as e
import numpy as np
from scipy.integrate import solve_ivp
import weightHandler as wh

mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelpad'] = 6
mpl.rcParams['savefig.dpi'] = 1000
mpl.rcParams['figure.figsize'] = (19, 10)


def plotSigmoidFunction():
    uValues = np.linspace(-1, 1.5, 300)
    fig = plt.figure()
    plt.plot(uValues, e.getSigmoid(uValues))
    return fig


def plotTuningCurve():
    # To replicate Figure 2.
    # K = 8.08
    # A = 2.53 Hz
    # Be^k = 34.8 Hz
    theta_0 = 0
    fig = plt.figure("Paper - Figure 2")
    ax = fig.add_subplot()
    dataY = e.getTuningCurve(theta_0=theta_0)
    dataX = theta_0 - p.thetaSeries
    plt.plot(dataX, dataY)
    plt.suptitle("Activity of a HD cell in the anterior thalamus")
    plt.title(r"K=%d, A=%d, B=%d" % (p.K, p.A, p.B))
    plt.xlabel(
        r"Difference between head direction versus preferred direction , $\theta - \theta_0$")
    plt.ylabel("Firing rate, f (Hz)")
    plt.tight_layout()

    # Add padding to x and y axis.
    plt.ylim(0, dataY.max()+(dataY.max()*0.1))
    plt.xlim(p.thetaSeries.min(), p.thetaSeries.max())

    # Suffix x-axis labels with degree sign.
    ax.xaxis.set_major_formatter('{x:1.0f}°')

    # Ensure axes ticks match paper.
    ax.set_xticks([-180, -90, 0, 90, 180])

    plt.savefig(p.outputDirectory+'/figures/tuning-curve.svg', dpi=350)
    return fig


def plotSampledNeuroneWeightDistributions(neuronalPopulation):
    fig, ax = plt.subplots(nrows=3, ncols=4, squeeze=True)
    fig.set_tight_layout(True)
    neuroneIdsToSample = np.linspace(
        0, len(neuronalPopulation.neurones)-1, num=12)

    rowId = 0
    columnId = 0
    for neurone in neuronalPopulation.neurones[neuroneIdsToSample.astype(int)]:
        ax[rowId, columnId].plot(p.thetaSeries, neurone.getWeights())
        maxYIndex = np.argmax(neurone.getWeights())
        ax[rowId, columnId].axvline(p.thetaSeries[maxYIndex], color='red')
        ax[rowId, columnId].set_title(
            r'$\theta_0=%d°$' "\n" r'Strongest connection to: %d°' % (neurone.theta_0, p.thetaSeries[maxYIndex]))
        ax[rowId, columnId].set_xlabel(
            r'Neurone(s) with preferred head direction, $\theta$')
        ax[rowId, columnId].set_ylabel('Weight to neurone')
        if(columnId == 3):
            rowId = rowId + 1
            columnId = 0
        else:
            columnId += 1

    #plt.xlabel("common X")
    #plt.ylabel("common Y")
    plt.suptitle(
        'Weights of 12 neurones, each with unique preferred head directions')
    plt.tight_layout()
    plt.savefig(p.outputDirectory + '/figures/12-weight-plots.svg', dpi=700)
    return fig


def plotWeightDistribution(weights, hasNoise=False, title=""):
    # NOTE: This produces figures that are sensitive to theta_0 (i.e., the preferred head direction).
    # TODO: This function could do with some experimentation, where neuronal population weights are not rolled, but rather set according to varying theta_0.
    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.imshow(weights)
    ticks = [{
        'location': int(i),
        'label': str(np.ceil(p.thetaSeries[int(i)]))+'°'
    } for i in np.linspace(0, len(p.thetaSeries)-1, 9)]
    # ax.invert_yaxis()

    ax.set_xticks([tick['location'] for tick in ticks],
                  [tick['label'] for tick in ticks])
    ax.set_yticks([tick['location'] for tick in ticks],
                  [tick['label'] for tick in ticks])
    plt.xlabel(r"HD cell's preferred direction, $\theta$ (degrees)")
    plt.ylabel(r"HD cell's preferred direction, $\theta$ (degrees)")

    fig.colorbar(im)
    if(hasNoise):
        plt.title(r"N=%d, K=%d, A=%d, B=%d, " "$\lambda$" "=%f" %
                  (p.numberOfUnits, p.K, p.A, p.B, p.penaltyForMagnitude_0))
        plt.suptitle(
            "Strength of connections (weights) between neurones" + title)
        plt.savefig(p.outputDirectory +
                    '/figures/weights-heatmap-noisy.svg', dpi=350)
    else:
        plt.title(r"N=%d, K=%d, A=%d, B=%d" % (p.numberOfUnits, p.K, p.A, p.B))
        plt.suptitle(
            "Strength of connections (weights) between neurones " + title)
        plt.savefig(p.outputDirectory +
                    '/figures/weights-heatmap-noiseless.svg', dpi=350)

    return fig


def plotZerothNeuroneOddAndEvenWeights(neuronalPopulation_dynamic):
    fig, ax = plt.subplots(nrows=3, ncols=1, squeeze=True)
    for neurone in neuronalPopulation_dynamic.neurones:
        if(neurone.theta_0 == 0):
            ax[0].set_title('A neurone\'s evenWeights')
            ax[0].plot(p.thetaSeries, neurone.evenWeights)

            ax[1].set_title('A neurone\'s oddWeights')
            ax[1].plot(p.thetaSeries, neurone.oddWeights)

            ax[2].set_title('A neurone\'s allWeights')
            ax[2].plot(p.thetaSeries, neurone.getWeights())
    return fig


def solveDuDt(neuronalPopulation, title=""):
    # Solve for time
    fig = plt.figure()
    plt.suptitle(
        'Firing activity (f) of each neurone in the population over time')
    ax = fig.gca()

    # Labelling X-Axis
    ax.set_xlabel('Time')

    # Labelling Y-Axis
    ax.set_ylabel('Neurones firing rate (Hz)')
    # Labelling Z-Axis
    # ax.set_zlabel('Time')

    t0 = p.timeSeries[0].astype('float64')
    tf = p.timeSeries[-1].astype('float64')

    if(p.initialCondition == 'noise'):
        firingActivityOfAllNeurones = p.randomGenerator.normal(
            loc=10, scale=3, size=p.numberOfUnits)

    elif(p.initialCondition == 'tuningCurve'):
        firingActivityOfAllNeurones = e.getTuningCurve(theta_0=90)

    elif(p.initialCondition == 'steadyState'):
        firingActivityOfAllNeurones = np.ones(
            p.numberOfUnits) * e.getF(-0.4635)

    elif(p.initialCondition == 'slightlyAwayFromSteadyState'):
        firingActivityOfAllNeurones = np.ones(p.numberOfUnits) * e.getF(-0.6)

    firingActivityOfAllNeurones = np.abs(firingActivityOfAllNeurones)

    uActivityOfAllNeurones = e.getU(firingActivityOfAllNeurones)
    neuronalWeights = neuronalPopulation.getAllWeights()

    sol = solve_ivp(e.getDuDt, (t0, tf), uActivityOfAllNeurones,
                    args=[neuronalWeights], dense_output=True, t_eval=p.timeSeries)

    usAtTimeT = np.transpose(sol.sol(p.timeSeries))
    fsAtTimeT = e.getF(usAtTimeT)
    plt.plot(p.timeSeries, fsAtTimeT)

    fig = plt.figure()
    plt.suptitle('Population firing activity over time - ' + title + "\n" r"N=%d, K=%d, A=%d, B=%d, " "$\lambda$" "=%f" %
                 (p.numberOfUnits, p.K, p.A, p.B, p.penaltyForMagnitude_0))
    ax = fig.gca(projection="3d")
    # Labelling X-Axis
    ax.set_xlabel('Firing rate $f$ (Hz)')

    # Labelling Y-Axis
    ax.set_zlabel('Time (ms)')
    ax.yaxis.set_major_formatter('{x:1.0f}°')

    # Labelling Z-Axis
    ax.set_ylabel('Population of HD Cells')
    #plt.xlim(0, 40)

    tsToSample = np.linspace(0, len(p.timeSeries)-1, 10).astype('int')
    for tToSample in tsToSample:
        x, y, z,  = p.thetaSeries, fsAtTimeT[tToSample], p.timeSeries[tToSample]
        plt.plot(y, x, z, color='black')

    #yValues = p.thetaSeries[np.argmax(fAtTimeT, axis=1)]
    ax.invert_xaxis()
    ax.invert_zaxis()
    plt.savefig(p.outputDirectory +
                '/figures/dudt-over-time.svg', dpi=350)
    return fig


def plotEffectOfAdditionalTimedUInput(neuronalPopulation):
    tsToSample = np.linspace(0, len(p.timeSeries)-1, 50).astype('int')
    externalCurrentStart = 200  # msec
    externalCurrentStop = 700  # msec
    additionalUInputAtTimeT = np.zeros(
        shape=(p.timeSeries.size, p.thetaSeries.size))
    usWithInput = np.argwhere((p.timeSeries >= externalCurrentStart) & (
        p.timeSeries <= externalCurrentStop))
    input1 = e.getU(
        e.getTuningCurve(theta_0=90)) * 0.25
    input2 = e.getU(e.getTuningCurve(theta_0=-90)) * 0.2

    additionalUInputAtTimeT[usWithInput] = np.add(input1, input2)
    if(p.initialCondition == 'noise'):
        firingActivityOfAllNeurones = p.randomGenerator.normal(
            loc=10, scale=3, size=p.numberOfUnits)

    elif(p.initialCondition == 'tuningCurve'):
        firingActivityOfAllNeurones = e.getTuningCurve(theta_0=0)

    elif(p.initialCondition == 'steadyState'):
        firingActivityOfAllNeurones = np.ones(
            p.numberOfUnits) * e.getF(-0.4635)

    elif(p.initialCondition == 'slightlyAwayFromSteadyState'):
        firingActivityOfAllNeurones = np.ones(p.numberOfUnits) * e.getF(-0.6)

    firingActivityOfAllNeurones = np.abs(firingActivityOfAllNeurones)

    uActivityOfAllNeurones = e.getU(firingActivityOfAllNeurones)
    neuronalWeights = neuronalPopulation.getAllWeights()
    t0 = p.timeSeries[0].astype('float64')
    tf = p.timeSeries[-1].astype('float64')

    sol = solve_ivp(e.getDuDtWithExternalInput, (t0, tf), uActivityOfAllNeurones,
                    args=[neuronalWeights, additionalUInputAtTimeT], dense_output=True, t_eval=p.timeSeries)
    usAtTimeT = np.transpose(sol.sol(p.timeSeries))
    fsAtTimeT = e.getF(usAtTimeT)

    # Solve for time
    fig = plt.figure()
    plt.suptitle(
        'Firing activity (f) of each neurone in the population over time')
    ax = fig.gca()

    # Labelling X-Axis
    ax.set_xlabel('Time')
    # Labelling Y-Axis
    ax.set_ylabel('Neurones firing rate (Hz)')
    # Labelling Z-Axis
    # ax.set_zlabel('Time')
    colors = [plt.cm.tab10(x) for x in np.linspace(0, 0.3, 4)]
    colors = [[1, 0, 0, 0.5], [0, 1, 0, 0.9], [0, 0, 1, 0.5]]
    for index, fAtTimeT in enumerate(fsAtTimeT.T):
        currentColor = int(index/len(fsAtTimeT.T)*3)
        plt.plot(p.timeSeries, fAtTimeT, color=colors[currentColor])

    fig = plt.figure()
    plt.suptitle('Population firing activity over time - ' "\n" r"N=%d, K=%d, A=%d, B=%d, " "$\lambda$" "=%f" %
                 (p.numberOfUnits, p.K, p.A, p.B, p.penaltyForMagnitude_0))
    ax = fig.gca(projection="3d")
    # Labelling X-Axis
    ax.set_xlabel('Firing rate $f$ (Hz)')

    # Labelling Y-Axis
    ax.set_zlabel('Time (ms)')
    ax.yaxis.set_major_formatter('{x:1.0f}°')

    # Labelling Z-Axis
    ax.set_ylabel('Population of HD Cells')
    #plt.xlim(0, 40)

    for tToSample in tsToSample:
        isATPeriodWithAdditionalInput = np.argwhere(
            usWithInput == tToSample).size > 0
        if (isATPeriodWithAdditionalInput):
            color = [1, 0, 0, 0.8]
        else:
            color = [0, 0, 0, 0.3]
        x, y, z,  = p.thetaSeries, fsAtTimeT[tToSample], p.timeSeries[tToSample]
        plt.plot(y, x, z, color=color)

    #yValues = p.thetaSeries[np.argmax(fAtTimeT, axis=1)]
    ax.invert_xaxis()
    ax.invert_zaxis()
    plt.savefig(p.outputDirectory +
                '/figures/dudt-over-time.svg', dpi=350)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    # Labelling X-Axis
    ax.set_xlabel('External current, $u$')

    # Labelling Y-Axis
    ax.set_zlabel('Time (ms)')
    ax.yaxis.set_major_formatter('{x:1.0f}°')

    # Labelling Z-Axis
    ax.set_ylabel('Population of HD Cells')
    plt.title('Input currents over time with addition from local view detector')

    for tToSample in tsToSample:
        x, y, z,  = p.thetaSeries, additionalUInputAtTimeT[tToSample], p.timeSeries[tToSample]
        y1 = usAtTimeT[tToSample]
        isATPeriodWithAdditionalInput = np.argwhere(
            usWithInput == tToSample).size > 0
        if (isATPeriodWithAdditionalInput):
            color = 'red'
        else:
            color = 'black'
        plt.plot(y, x, z, color='red')
        plt.plot(y1, x, z, color='black')

    ax.invert_xaxis()
    ax.invert_zaxis()

    return fig
