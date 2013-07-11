from __future__ import division
import numpy as np


def MTF(spatial_frequency,eccentricity, paper='Williams1996_astig'):
    """    
    Compute the modulation frequency transfer as a function of eccentricity
    based on the equation derrived in Navarro, Artal, and Williams 1993. 
    
    .. math::
       F = 1-C * exp(-A*f) + C*exp(-B*f)
    
    :param spatial_frequency: array or integer of spatial frequencies
    :param eccentricity: eccentricity from 0 to 60 degrees at which to compute MTF.\
    Linear interpolation is used for values that fall inbetween those \
    reported in Navarro et al and are, therefore, less reliable though as expected. \
    Eccentricities reported (in degrees): [0, 5, 10, 20, 30, 40, 50, 60]
    :param paper: Choose the coefficients to use for generating the curve.  \n
                 Options are : \n
                 * 'Williams1996_astig' \n
                 * 'Williams1996_clc' \n
                 * 'Navarro1993' 
                 
    :returns: MTF for input frequencies at given eccentricity.

    
    """
    if paper == 'Navarro1993':    
        theta = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0] # in degrees
        A_theta = [0.172, 0.245, 0.245, 0.328, 0.606, 0.82, 0.93, 1.89] # in degrees
        B_theta = [0.037, 0.041, 0.041, 0.038, 0.064, 0.064, 0.059, 0.108] # in degrees
        C_theta = [0.22, 0.2, 0.2, 0.14, 0.12, 0.09, 0.067, 0.05] # unitless

    if paper == 'Williams1996_astig':
        theta = [0.0, 10.0, 20.0,  40.0]
        A_theta = [0.0129, 0.0140, 0.0082, 0.0059]
        B_theta = [0.0816, 0.1036, 0.1313, 0.1555]
        C_theta = [0.7921, 0.8299, 0.9120, 0.9178]
 
    if paper == 'Williams1996_clc':
        theta =   [0.0,    10.0,   20.0,   40.0]
        A_theta = [0.0122, 0.0154, 0.0000, 0.0000]
        B_theta = [0.0988, 0.1466, 0.2305, 0.4663]
        C_theta = [0.8172, 0.8266, 0.9378, 0.9515]       
    # linear interpolation:

    A = np.interp(eccentricity,theta,A_theta)
    B = np.interp(eccentricity,theta,B_theta)
    C = np.interp(eccentricity,theta,C_theta)

    f = (1.0 - C) * np.exp(-A * spatial_frequency) + C * np.exp(-B * spatial_frequency)
    
    return f
    

    
def MTF_Pupil(spatial_frequency, pupilDiameter, normalized = 0):
    """
    
    Compute the modulation frequency transfer as a function of pupil size
    as derrived in Artal and Navarro 1994. 
    F = 1-C * exp(-A*f) + C*exp(-B*f)

    :param spatial_frequency: array or integer of spatial frequencies
    :param eccentricity: eccentricity from 0 to 60 degrees at which to compute MTF.\
    Linear interpolation is used for values that fall inbetween those \
    reported in Navarro et al and are, therefore, less reliable though as expected. \
    Pupil size reported (in mm): [2.5, 3.0, 4.0, 6.0, 8.0]. The behavior of \
    this function in between reported sizes may be worse than MTF(). \
    :param normalized: Default = 0 (no). 1 (yes) will normalize the spatial \
    frequency.  
    
    .. note::
       * This is not fully tested.  Permits comparison of a given
         optical system (eye) to a perfect one.  Requires consideration of maximum 
         spatial frequency of the optical system (uLim).    
    

    :returns f: MTF for input frequencies at given eccentricity.
    
    .. warning:: 
       normalized option 1 is not fully tested.
    
    """

    pupil = [2.5, 3.0, 4.0, 6.0, 8.0] # in mm.

    if normalized == 0:

        A_pupil = [0.16, 0.16, 0.18, 0.31, 0.53] # in degrees
        B_pupil = [0.06, 0.05, 0.04, 0.06, 0.08] # in degrees
        C_pupil = [0.36, 0.28, 0.18, 0.2, 0.11] # unitless
        
        A = np.interp(pupilDiameter,pupil,A_pupil)
        B = np.interp(pupilDiameter,pupil,B_pupil)
        C = np.interp(pupilDiameter,pupil,C_pupil)

        f = (1 - C) * np.exp(-A * spatial_frequency) + C * np.exp(-B * spatial_frequency)

    elif normalized == 1:
        
        A_pupil = [10.57, 12.68, 19.04, 49.19, 112.15] # no dimension
        B_pupil = [3.96, 3.96, 4.23, 9.52, 16.92] # no dimension
        C_pupil = [0.36, 0.28, 0.18, 0.2, 0.11] # no dimension
        uLim = [66.1, 79.3, 105.8, 158.7, 211.6] # in cycles/degree;
        
        A = np.interp(pupilDiameter,pupil,A_pupil)
        B = np.interp(pupilDiameter,pupil,B_pupil)
        C = np.interp(pupilDiameter,pupil,C_pupil)
        u = np.interp(pupilDiameter,pupil,uLim)
    
        f = (1 - C) * np.exp(-A * (spatial_frequency / u)) + C * np.exp(-B * (spatial_frequency / u))
    
    return f

def StilesCrawford1stKind(x, xmax, n, rho):
    """
    
    Stiles-Crawford effect of the 1st order.
    
    :param x: array of x values
    :param xmax: maximum value
    :param n: 
    :param rho: angle
    
    :returns: styles-crawford effect
    
    .. warning::
       This funciton is not finished or tested.
    """
    
    return np.log(n) - rho*(x - xmax)**2

    
    
def SterhlRatio(sigma, lam):
    """
    Find the Sterhl ratio.
    
    .. warning::
       This funciton is not finished or tested.
       
    """
    
    return np.exp(-2.0 * np.pi * sigma / lam)**2.0

    

def NumericalAperature(n, theta):
    """
    Find the numerical aperature of a system
    
    .. warning::
       Not really working. Untested.
       
    """
    return n * np.sin(theta)

    
def DiffractionLimit(lam, n,theta):
    """
    
    Find the diffraction limit of a system.
    
    :param lam: wavelength
    :param n: 
    :param theta:
    
    .. warning::
       This funciton is not finished or tested.
        
    """
    return lam / 2.0 * ( NumericalAperature(n, theta) )
    

