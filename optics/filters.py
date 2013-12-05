from __future__ import division
import numpy as np

def stockman(minLambda=390, maxLambda=770, ONLY_LENS=False, 
            ONLY_MACULA=False, RETURN_SPECTRUM=False, resolution=1):
    '''
    '''
    dsamp = resolution * 10.0
    lens = np.genfromtxt('base/optics/stockman/lens.csv', 
                         delimiter=',')[::dsamp, :]
    macula = np.genfromtxt('base/optics/stockman/macular.csv', 
                            delimiter=',')[::dsamp, :]

    spectrum = lens[:, 0]
    # find location of maxLambda
    ind1 = np.where(spectrum == minLambda)[0]
    ind2 = np.where(spectrum == maxLambda)[0]

    spectrum = spectrum[ind1:ind2 + 1]
    filters = (10.0 ** (lens[ind1:ind2 + 1, 1] + 
        macula[ind1:ind2 + 1, 1]))

    if ONLY_LENS:
        return lens[ind1:ind2 + 1, 1]
    if ONLY_MACULA:
        return macula[ind1:ind2 + 1, 1]
    if RETURN_SPECTRUM:
        return filters, spectrum
    return filters