import numpy as np

from base import data as dm
        

def RemoveMean(image):
    """
    RemoveMean(image)
    
    subtract the mean of an image.
    
    :param image: input image.
    
    :returns: image with mean subtracted

    """

    x, y = image.shape
    mu = np.sum(image) / (x * y)
    out = image - mu
    
    return out


def decibels(x):
    '''
    '''
    return 10.0 * np.log10(x)
    

def Density(dat):
    """Compute the density of a power spectrum.
    
    :param dat: data to normalize.
    :type dat: np.array
    
    :returns: normalized data.
    
    .. note::
       Currently very simply data / sum(data)
    """

    return dat / np.sum(dat)
    
    
def PowerSpectrum2(im, win=2, n1=1, n2=0):
    """2D spectrum estimation using the modified periodogram.
    This one includes a window function to decrease variance in \
    the estimate.
    
    :param x: input sequence
    :type x: numpy.array
    :param n1: starting index, x(n1)
    :type n1: int
    :param n2: ending index, x(n2)
    :type n2: int
    :param win: The window type \n
                1 = Rectangular \n
                2 = Hamming \n
                3 = Hanning \n
                4 = Bartlett \n
                5 = Blackman \n
    :type win: int
    
    :returns: spectrum estimate.
    :rtype: numpy.array
            
    .. note:: 
       If n1 and n2 are not specified the periodogram of the entire 
       sequence is computed.
    
    """
    
    if n2 == 0:
        n2 = len(im[:,1])
    
    N  = n2 - n1 + 1
    w  = np.ones((N))
    
    if (win == 2):
        w = np.hamming(N)
    elif (win == 3):
        w = np.hanning(N)
    elif (win == 4):
        w = np.bartlett(N)
    elif (win == 5): 
        w = np.blackman(N);
    
    
    
    xs, ys = im.shape
    if xs/ys != 1:
        raise ValueError('Dimensions must be equal')
    
    
    m = w[:] * w[:][np.newaxis,:]
    
    U  = np.linalg.norm(w)**2.0 / N**2.0
    
    fftim = np.abs(np.fft.fftshift(np.fft.fft2(((im) * m)))) / ( (N**2.0) * U)
    
    return fftim
    
    
    
def welch2d(x,L = None, over = 0.5, win = 2.0):
    """2D spectrum estimation using Welch's method.
    The spectrum of an image x is estimated using Welch's \
    method of averaging modified periodograms.
    
    :param x: image
    :param L: section length
    :param over: amount of overlap, where 0<over<1,
    :param win: The window type \n
                    1 = Rectangular \n
                    2 = Hamming \n
                    3 = Hanning \n
                    4 = Bartlett \n
                    5 = Blackman \n
    
    
    :returns: Welch's estimate of the power spectrum, returned in decibels. 
    
    .. note:: 
       Modified from: M.H. Hayes. \
       "Statistical Digital Signal Processing and Modeling" \
       (John Wiley & Sons, 1996).
        
    """
    
    xs, ys = x.shape
    x = RemoveMean(x)
    
    if L == None:
        L = xs
        
    if xs / ys != 1.0:
        raise ValueError('This is a stupid program. Dimensions need to be \
        equal (len(x)=len(y))')
        
    if L < len(x[:,0]) / 2.0:
        raise ValueError('Length must be longer than 1/2 length of x')
        
    if (over >= 1) or (over < 0):
        raise ValueError('Overlap is invalid')
       
    n0 = (1.0 - over) * L
    n1 = np.array([1.0, 1.0]) - n0
    n2 = np.array([L, L]) - n0
    nsect = int(1.0 + np.floor((len(x) - L) /( n0)))
    
    Px = 0
    for ix in range(0,nsect):
        n1[0] = n1[0] + n0
        n2[0] = n2[0] + n0
        for iy in range(0,nsect):
            n1[1] = n1[1] + n0
            n2[1] = n2[1] + n0
            Px += PowerSpectrum2(x[n1[0]:n2[0],n1[1]:n2[1]], win) / (nsect**2)
    
    xs, ys = Px.shape
    
    f2 = np.arange(-xs / 2.0, xs / 2.0)
    f1 = np.arange(-ys / 2.0, ys / 2.0)
    XX, YY = np.meshgrid(f1,f2)
    foo, r = dm.cart2pol(XX,YY)
    if np.mod(xs,2)==1 or np.mod(ys,2)==1:
        r = np.around(r)-1
    else:
        r = np.around(r)
    
    r = np.array(r,dtype=int)
    ### need to possibly use a for loop.
    avg = dm.accum(r.flatten() + 1, Px.flatten()
                    )[1:] / dm.accum(r.flatten() + 1)[1:]
    avg = avg[:(xs / 2 + 1)]
            
    return avg


    
def rgb2gray(rgb):
    """Convert an image from rbg into gray scale.
    
    :param rgb: RGB image to be converted.
    
    :returns: Grayscale image of the same size as RGB input image.
    
    *Stole this one from stackoverflow.*
    
    Formula used in conversion:
    
    :math:`0.2999*R_{channel} + 0.587*G_{channel} + 0.114*B_{channel}`
        
    """
    
    r, g, b = np.rollaxis(rgb[...,:3], axis = -1)
    
    return 0.299 * r + 0.587 * g + 0.114 * b

        
def FindPowerExponents(spectrum):
    """Find the exponent,m, that provides a best fit for to data using 1/f^m for
    a 2D spectrum.
    
    :param spectrum: power spectrum.
    
    
    :returns:  * M_X: values that are best fits to data in X columns
                 * M_Y: values that are best fits to data in Y columns
    
    .. note:: 
         not working in python yet. See Cone Activity
 
    """

    xs, ys, = spectrum.shape

    M_Y = np.zeros((ys))
    #M_X = np.zeros((xs))
    """
    for i in range(0,xs):
        M_X[i] = FitPowerLaw(spectrum[i,:],2)


    for i in range(0,yx):
        M_Y[i] = FitPowerLaw(np.flipud(spectrum[:,i]),2)
    """
    return M_Y