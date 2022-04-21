import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pl
import parameters as p
import mathEquations as e
import numpy as np
from scipy.integrate import solve_ivp
import weightHandler as wh

mpl.rcParams['savefig.dpi'] = 1000
mpl.rcParams['figure.figsize'] = (19, 10)

def plotTuningCurve():
  # To replicate Figure 2.
  # K = 8.08
  # A = 2.53 Hz
  # Be^k = 34.8 Hz
  theta_0 = 0
  fig = plt.figure("Paper - Figure 2")
  ax = fig.add_subplot()
  dataY = e.getTuningCurve(theta_0=theta_0)
  dataX = theta_0 - p.theta
  plt.plot(dataX, dataY)
  plt.suptitle("Activity of a HD cell in the anterior thalamus")
  plt.title(r"K=%d, A=%d, B=%d" % (p.K, p.A, p.B))
  plt.xlabel(r"Difference between head direction versus preferred direction , $\theta - \theta_0$")
  plt.ylabel("Firing rate, f (Hz)")
  plt.tight_layout()

  # Add padding to x and y axis.
  plt.ylim(0, dataY.max()+(dataY.max()*0.1))
  plt.xlim(p.theta.min(), p.theta.max())

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
    ax[rowId, columnId].plot(p.theta, neurone.getWeights())
    maxYIndex = np.argmax(neurone.getWeights())
    ax[rowId, columnId].axvline(p.theta[maxYIndex], color='red')
    ax[rowId, columnId].set_title(
        r'$\theta_0=%d°$' "\n" r'Strongest connection to: %d°' % (neurone.theta_0, p.theta[maxYIndex]))
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

def plotWeightDistribution(weights, hasNoise=False):
  # NOTE: This produces figures that are sensitive to theta_0 (i.e., the preferred head direction).
  # TODO: This function could do with some experimentation, where neuronal population weights are not rolled, but rather set according to varying theta_0.
  fig = plt.figure()
  ax = fig.add_subplot()
  im = ax.imshow(weights)
  ticks = [{
      'location': int(i),
      'label': str(np.ceil(p.theta[int(i)]))+'°'
  } for i in np.linspace(0, len(p.theta)-1, 9)]
  ax.invert_yaxis()

  ax.set_xticks([tick['location'] for tick in ticks],
                [tick['label'] for tick in ticks])
  ax.set_yticks([tick['location'] for tick in ticks],
                [tick['label'] for tick in ticks])
  plt.xlabel(r"HD cell's preferred direction, $\theta$ (degrees)")
  plt.ylabel(r"HD cell's preferred direction, $\theta$ (degrees)")
  plt.title(r"K=%d, A=%d, B=%d" % (p.K, p.A, p.B))
  fig.colorbar(im)
  if(hasNoise):
    plt.suptitle("Strength of connections (weights) between neurones (noisy)")
    plt.savefig(p.outputDirectory+'/figures/weights-heatmap-noisy.svg', dpi=350)
  else:
    plt.suptitle("Strength of connections (weights) between neurones (noiseless)")
    plt.savefig(p.outputDirectory + '/figures/weights-heatmap-noiseless.svg', dpi=350)
  
  return fig


def plotTest(neuronalPopulation):
    # Solve for time
  fig = plt.figure()
  ax = fig.gca(projection='3d')

    # Labelling X-Axis
  ax.set_xlabel('Theta')

  # Labelling Y-Axis
  ax.set_ylabel('Hz')

  # Labelling Z-Axis
  ax.set_zlabel('Time')
  
  initialF = p.randomGenerator.uniform(low=0,high=1,size=p.numberOfUnits)
  initialU = e.getU(initialF)
  w = neuronalPopulation.getAllWeights()
  sol = solve_ivp(e.getDuDt, (p.timeSeries[0], p.timeSeries[-1]), initialU, args=[w, initialF])
  uAtTimeT = sol.y.T
  fAtTimeT = e.getF(uAtTimeT)
  for fIndex, f in enumerate(fAtTimeT):
    plt.plot(p.theta, f, p.timeSeries[fIndex],  color='black')
