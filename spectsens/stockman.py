from __future__ import division
import numpy as np


def stockman(minLambda=390, maxLambda=770, LOG=False,
				RETURN_SPECTRUM=False, resolution=1):
    '''
    '''
    dsamp = resolution * 10.0
    sens = np.genfromtxt('base/spectsens/stockman/specSens.csv', 
                    delimiter=',')[::dsamp, :]

    spectrum = sens[:, 0]
    print maxLambda
    # find location of min and max Lambda
    ind1 = 0 #np.where(spectrum == minLambda)[0]
    ind2 = np.where(spectrum == maxLambda)[0]
    print ind1, ind2
    # take only between min and max lambda
    spectrum = spectrum[ind1:ind2 + 1]
    sens = sens[ind1:ind2 + 1, :]

    if not LOG:
        sens = 10.0 ** sens

    if RETURN_SPECTRUM:
        return sens, spectrum

    return sens

def stockmanfund(minLambda=390, maxLambda=770, LOG=False,
				RETURN_SPECTRUM=False, resolution=1):
    '''
    '''
    dsamp = resolution * 10.0
    sens = np.genfromtxt('base/spectsens/stockman/fundamentals2deg.csv', 
                    delimiter=',')[::dsamp, :]

    spectrum = sens[:, 0]
    # find location of min and max Lambda
    ind1 = 0 #np.where(spectrum == minLambda)[0]
    ind2 = np.where(spectrum == maxLambda)[0]

    # take only between min and max lambda
    spectrum = spectrum[ind1:ind2 + 1]
    sens = sens[ind1:ind2 + 1, :]

    if not LOG:
        sens = 10.0 ** sens

    if RETURN_SPECTRUM:
        return sens, spectrum
    return sens