from __future__ import division
import numpy as np
import os


this_dir = os.path.dirname(os.path.realpath(__file__))

def stockman(minLambda=390, maxLambda=770, ONLY_LENS=False, 
            ONLY_MACULA=False, RETURN_SPECTRUM=False, resolution=1):
    '''
    '''
    dsamp = resolution * 10.0
    lens = np.genfromtxt(this_dir + '/stockman/lens.csv', 
                         delimiter=',')[::dsamp, :]
    macula = np.genfromtxt(this_dir+ '/stockman/macular.csv', 
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


def lens_age_correction(age, spectrum):
    '''
    '''
    #dat = np.genfromtxt(this_dir + '/pokorny/lens_age_1987.csv', 
    #    delimiter=',', names=True)
    sp_dat = {'wavelength': np.arange(400, 651, 10),
        'TL1': np.array([0.6, 0.51, 0.433, 0.377, 0.327, 
            0.295, 0.267, 0.233, 0.207, 0.187, 0.167, 
            0.147, 0.133, 0.12, 0.107, 0.093, 0.08, 0.067, 
            0.053, 0.04, 0.033, 0.027, 0.02, 0.013, 0.007, 0]),
        'TL2': np.array([1, 0.583, 0.3, 0.116, 0.033, 0.005, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}

    ind = np.where(spectrum == 650)[0] + 1

    # interpolate values from pokorny with linear interp.
    TL1 = np.interp(spectrum[:ind], sp_dat['wavelength'], sp_dat['TL1'])
    TL2 = np.interp(spectrum[:ind], sp_dat['wavelength'], sp_dat['TL2'])

    # From Pokorny, Smith and Lutze 1987:
    if age <= 60:
        lens = TL1 * (1 + 0.02 * (age - 32)) + TL2

    if age > 60:
        lens = TL1 * (1.56 + 0.0667 * (age - 60)) + TL2

    # zero out remainder of spectrum (lens has no filtering there)
    out = np.zeros(len(spectrum))
    out[:ind] = lens
    return out
