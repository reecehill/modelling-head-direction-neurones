import matplotlib.pyplot as plt
import parameters as p
import mathEquations as e
import numpy as np


def plotTuningCurve():
  # To replicate Figure 2.
  # K = 8.08
  # A = 2.53 Hz
  # Be^k = 34.8 Hz

  fig = plt.figure("Paper - Figure 2")
  ax = fig.add_subplot()
  dataY = e.getTuningCurve()
  dataX = p.theta_0 - p.theta
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


def plotWeightDistribution(weights):  
  fig = plt.figure()
  ax = fig.add_subplot()
  im = ax.imshow(weights)
  ax.xaxis.set_major_formatter('{x:1.0f}°')
  ax.yaxis.set_major_formatter('{x:1.0f}°')
  ax.invert_yaxis()
  plt.xlabel(r"HD cell's preferred direction, $\theta$ (degrees)")
  plt.ylabel(r"HD cell's preferred direction, $\theta$ (degrees)")
  plt.suptitle("Strength of connections (weights) between neurones")
  plt.title(r"K=%d, A=%d, B=%d, $\theta_0$=%d" % (p.K, p.A, p.B, p.theta_0))
  fig.colorbar(im)
  return fig
