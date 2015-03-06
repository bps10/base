from __future__ import division
import os
import numpy as np


this_dir = os.path.dirname(os.path.realpath(__file__))


def smithpokorny(minLambda=390, maxLambda=770, log=False, 
                 return_spect=False, resolution=1, quanta=True):
    '''
    '''
    if quanta: 
        data_name = '/smithpokorny/spq.csv'
    else:
        data_name = '/smithpokorny/sp.csv'


    dat = np.genfromtxt(this_dir + data_name, 
                    delimiter=',')
    spectrum = dat[:, 0]

    # make sure min and max are within measured range
    if minLambda < spectrum[0]:
        raise InputError('minLambda cannot be less than {0} nm'.format(
            str(spectrum[0])))
    if maxLambda > spectrum[-1]:
        raise InputError('maxLambda cannot be greater than {0} nm'.format(
            str(spectrum[-1])))
                        
    # interpolate
    npoints = (maxLambda - minLambda) / resolution
    newspect = np.linspace(minLambda, maxLambda, npoints)
    
    sens = np.zeros((len(newspect), 3))
    sens[:, 0] = np.interp(newspect, spectrum, dat[:, 1]) # L cones
    sens[:, 1] = np.interp(newspect, spectrum, dat[:, 2]) # M cones
    sens[:, 2] = np.interp(newspect, spectrum, dat[:, 3]) # S cones

    if not log:
        sens = 10.0 ** sens

    if return_spect:
        return sens, newspect
    return sens



def JuddVosCIE_to_sp():
    '''
    '''    
    M = np.array([[0.15514, 0.54312, -0.03286],
                  [-0.15514, 0.45684, 0.03286],
                  [0.0, 0.0, 0.00801]])
    return M


def sp_to_JuddVosCIE():
    '''
    '''
    M = JuddVosCIE_to_sp()
    return np.linalg.inv(M)

