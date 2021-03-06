# -*- coding: utf-8 *-*
import numpy as np
from base import optics as op

def neitz(LambdaMax=559, OpticalDensity=0.5, LOG=False,
                StartWavelength=390, EndWavelength=780, 
                resolution=1, EXTINCTION=False):
    """This function returns a photopigment spectral sensitivity curve
    as defined by Carroll, McMahon, Neitz, and Neitz.

    :param LambdaMax: Wavelength peak for photopigment
    :param OpticalDensity: optical density required
    :param OutputType:  log or anti-log.  \n
                        if log, maximum data ouput is 0. \n
                        if anti-log, data output is between 0 and 1. \n
    :param StartWavelength: beginning wavelength
    :param EndWavelength: end wavelength
    :param Resolution: Number of data points

    :returns: array of sensitivity values.
    :rtype: np.array

    .. note::
       Ported from Jim K's Matlab function.

    """

    A = 0.417050601
    B = 0.002072146
    C = 0.000163888
    D = -1.922880605
    E = -16.05774461
    F = 0.001575426
    G = 5.11376E-05
    H = 0.00157981
    I = 6.58428E-05
    J = 6.68402E-05
    K = 0.002310442
    L = 7.31313E-05
    M = 1.86269E-05
    N = 0.002008124
    O = 5.40717E-05
    P = 5.14736E-06
    Q = 0.001455413
    R = 4.217640000E-05
    S = 4.800000000E-06
    T = 0.001809022
    U = 3.86677000E-05
    V = 2.99000000E-05
    W = 0.001757315
    X = 1.47344000E-05
    Y = 1.51000000E-05
    Z = OpticalDensity #+ 0.00000001

    A2 = (np.log10(1.0 / LambdaMax) - np.log10(1.0 / 558.5))

    vector = np.log10(np.arange(StartWavelength,
                    EndWavelength + resolution, resolution) ** -1.0)

    const = 1.0 / np.sqrt(2.0 * np.pi)

    exTemp = (np.log10(-E + E * np.tanh(-(((10.0 ** (vector -
                A2))) - F) / G)) + D +
              A * np.tanh(-(((10.0 ** (vector - A2))) - B) / C) -
              (J / I * (const * np.exp(1.0) ** (-0.5 *
              (((10.0 ** (vector - A2)) - H) / I) ** 2.0))) -
              (M / L * (const * np.exp(1.0) ** (-0.5 *
               (((10.0 ** (vector - A2)) - K) / L) ** 2.0))) -
              (P / O * (const * np.exp(1.0) ** (-0.5 *
               (((10.0 ** (vector - A2)) - N) / O) ** 2.0))) +
              (S / R * (const * np.exp(1.0) ** (-0.5 *
              (((10.0 ** (vector - A2)) - Q) / R) ** 2.0))) +
              ((V / U * (const * np.exp(1.0) ** (-0.5 *
              (((10.0 ** (vector - A2)) - T) / U) ** 2.0))) / 10.0) +
              ((Y / X * (const * np.exp(1.0) ** (-0.5 *
              (((10.0 ** (vector - A2)) - W) / X) ** 2.0))) / 100.0))
    ODTemp = np.log10((1.0 - 10.0 ** -((10.0 ** exTemp) *
                        Z)) / (1.0 - 10 ** -Z))

    if LOG:
        extinction = exTemp
        withOD = ODTemp
    else:
        extinction = 10.0 ** exTemp
        withOD = 10.0 ** ODTemp

    if EXTINCTION:
        return extinction
    return withOD


def load_spect(speak=420, mpeak=530, lpeak=559,
    lambdamin=390, lambdamax=750, OD=[0.3, 0.4, 0.4],
    add_filters=True, LOG=False, resolution=1):
    ''' Load the spectral sensitivity functions with 
    pre-retinal filters accounted for. Returns a dictionary
    with 'l', 'm' and 's' as keys.
    '''
    # spectral sensitivity
    s_cone = neitz(LambdaMax=420, OpticalDensity=OD[0], 
                    LOG=LOG,
                    StartWavelength=lambdamin, 
                    EndWavelength=lambdamax, 
                    resolution=resolution, EXTINCTION=False)
    m_cone = neitz(LambdaMax=530, OpticalDensity=OD[1], 
                    LOG=LOG,
                    StartWavelength=lambdamin, 
                    EndWavelength=lambdamax, 
                    resolution=resolution, EXTINCTION=False)
    l_cone = neitz(LambdaMax=559, OpticalDensity=OD[2], 
                    LOG=LOG,
                    StartWavelength=lambdamin, 
                    EndWavelength=lambdamax, 
                    resolution=resolution, EXTINCTION=False)

    # filters
    if add_filters:
        filters = op.filters.stockman(minLambda=lambdamin, 
                maxLambda=lambdamax, resolution=resolution)

        # account for pre-retinal filters
        s_cone /= filters
        m_cone /= filters
        l_cone /= filters

    # package into dict
    sens = {'l': l_cone, 'm': m_cone, 's': s_cone}

    return sens
