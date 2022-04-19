import numpy as np
import parameters as p
import weightHandler as wh


def generatePopulation():
    # Population collection format is inspired by Eloquent, Laravel: https://laravel.com/docs/9.x/eloquent
    
    # It is possible to generate neurones in one of two ways. See parameters.py for more information.
    if(p.generateWeightsMethod == 'independentNeurone'):
      # Get each neurone's weight independently by looping through all theta values.
        population = {
            'neurones': [{
                'theta_0': theta,
                'weights': wh.generateWeightsForOneNeurone(theta_0=theta)
            } for theta in p.theta]
        }
        
    elif(p.generateWeightsMethod == 'templateNeurone'):
        # Instantiate a neurone with preferred direction 0, then roll its weights to offset them according to thetaIndex.
        population = {
            'neurones': [{
                'theta_0': theta,
                'weights': np.roll(wh.generateWeightsForOneNeurone(theta_0=0), thetaIndex)
            } for thetaIndex, theta in enumerate(p.theta)]
        }
    
    np.savetxt(p.outputDirectory+'/noiseless-weights.csv',
              [neurone['weights'] for neurone in population['neurones']], delimiter=',')
    return population


def getPopulationWeights(neuronalPopulation):
    return [neurone['weights'] for neurone in neuronalPopulation['neurones']]
