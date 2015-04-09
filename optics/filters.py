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


def lens_age_correction(age, spectrum, stiles=True):
    '''If stiles is true, use Wyszechki and Stiles (1982)
    measurements. If false, use van Norren and Vos (1974).
    '''
    if stiles:
        sp_dat = {'wavelength': np.arange(400, 651, 10),
                  'TL1': np.array([0.6, 0.51, 0.433, 0.377, 0.327, 
                            0.295, 0.267, 0.233, 0.207, 0.187, 0.167, 
                            0.147, 0.133, 0.12, 0.107, 0.093, 0.08, 0.067, 
                            0.053, 0.04, 0.033, 0.027, 0.02, 0.013, 0.007, 0]),
                  'TL2': np.array([1, 0.583, 0.3, 0.116, 0.033, 0.005, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
    else: # van Norren and Vos
        sp_dat = {'wavelength': np.arange(400, 651, 10),
                  'TL1': np.array([0.6, 0.51, 0.433, 0.377, 0.327, 
                            0.295, 0.267, 0.233, 0.207, 0.187, 0.167, 
                            0.147, 0.133, 0.12, 0.107, 0.093, 0.08, 0.067, 
                            0.053, 0.04, 0.033, 0.027, 0.02, 0.013, 0.007, 0]),
                  'TL2': np.array([1.33, 0.77, 0.354, 0.116, 0.033, 0.005, 0,
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


def vanNorren(age, spectrum, field_size='large'):
    '''
    '''

    if field_size.lower() == 'large':
        dRL = 0.225
    elif field_size.lower() == 'small':
        dRL = 0.446
    else:
        raise ValueError('field size must be small or large')

    media = ((dRL + 0.000031 * age ** 2) *
             (400 / spectrum) ** 4 +
             14.19 * 10.68 * np.exp(-((0.057 * (spectrum - 273)) ** 2)) +
             (0.998 - 0.000063 * age ** 2) * 2.13 *
             np.exp(-((0.029 * (spectrum - 370)) ** 2)) + 
             (0.059 + 0.000186 * age ** 2) * 11.95 *
             np.exp(-((0.021 * (spectrum - 325)) ** 2)) +
             (0.016 + 0.000132 * age ** 2) * 1.43 *
             np.exp(-((0.008 * (spectrum - 325)) ** 2)) + 0.111)

    return media
             
