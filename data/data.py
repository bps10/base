from __future__ import division
import scipy.io as spio
import numpy as np
try:
    import Image
except ImportError:
    from PIL import Image
import cPickle

### Data manipulation tools ###
###############################
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects. 
    stolen from: https://stackoverflow.com/questions/7008608/
    scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def find_local_max_min(data, only_max=False, only_min=False):
	'''Not used
	'''
	# find reversals
	data = np.array(data)
	minima = (np.r_[True, data[1:] < data[:-1]] & 
		np.r_[data[:-1] < data[1:], True])
	if only_min:
		return minima
	maxima = (np.r_[True, data[1:] > data[:-1]] & 
		np.r_[data[:-1] > data[1:], True])
	if only_max:
		return maxima
	return minima, maxima


def rad2deg(radians):
    """convert radians to degrees
    """
    return radians * 180.0 / np.pi


def deg2rad(degrees):
    """convert degrees to radians
    """
    return degrees * np.pi / 180.0

	
def accum(ii, a = None):
	"""
	
	A wraper for np.bincount that mimicks some of the functionality of Matlab's 'accumarray' function.
	
	:param ii:
	:param a:
	

	:returns: array of accumulated data
	
      .. note::
         Currently working much slower than Matlab's accumarray. Need to work \
         on this function or porting Octaves accum
         
      .. todo::
         Improve speed of accum
         
	"""
	if a == None:
	
		out = np.array([np.bincount(ii)])
		out = out.flatten()
		
	elif len(a.shape) == 1:
	
		out = np.array([np.bincount(ii, a)])
		out = out.flatten()
		
	else :
		out = np.array([np.bincount(ii, a) for q in a]).T
		out = np.mean(out[:,min(ii):],axis = 1)
		
	return out

def MatlabSort(data):
	"""
 
	sort with output index
 
	"""
	foo = list(enumerate(data))
	dtype = [('index',int),('theta',float)]
	foo = np.array(foo,dtype=dtype)
	sorted = np.sort(foo,order='theta')
	index = np.zeros((len(sorted)))
	dat = np.zeros((len(sorted)))
	for i in range(0,len(sorted)):
		index[i]= sorted[i][0]
		dat[i] = sorted[i][1]
		
	return index,dat
	

# Below from PsychoPy library. Copyright (C) 2009 Jonathan Peirce
# Distributed under the terms of the GNU General Public License (GPL).

def image2array(im):
    """
    
    Takes an image object (PIL) and returns an array
    
     fredrik lundh, october 1998
    
     fredrik@pythonware.com
     http://www.pythonware.com
    
    """
    if im.mode not in ("L", "F"):
            raise ValueError, "can only convert single-layer images"
    if im.mode == "L":
            a = np.fromstring(im.tostring(), np.uint8)
    else:
            a = np.fromstring(im.tostring(), np.float32)
    a.shape = im.size[1], im.size[0]
    return a

def array2image(a):
    """
    
    Takes an array and returns an image object (PIL)
    
    """ 
    # fredrik lundh, october 1998
    #
    # fredrik@pythonware.com
    # http://www.pythonware.com
    #
    if a.dtype.kind in ['u','I', 'B']:
            mode = "L"
    elif a.dtype.kind == np.float32:
            mode = "F"
    else:
            raise ValueError, "unsupported image mode"
    return Image.fromstring(mode, (a.shape[1], a.shape[0]), a.tostring())


def sph2cart(*args):
    """
    
    convert from spherical coordinates (elevation, azimuth, radius)
    to cartesian (x,y,z)
        
        
    """    
    if len(args)==1:	#received an Nx3 array
        elev = args[0][0,:]
        azim = args[0][1,:]
        radius = args[0][2,:]
        returnAsArray = True
    elif len(args)==3:
        elev = args[0]
        azim = args[1]
        radius = args[2]
        returnAsArray = False
        
    z = radius * np.sin(radians(elev))
    x = radius * np.cos(radians(elev))*np.cos(radians(azim))
    y = radius * np.cos(radians(elev))*np.sin(radians(azim))
    if returnAsArray:
        return np.asarray([x, y, z])
    else:
        return x, y, z
    


#---color conversions---#000000#FFFFFF------------------------------------------ 
def dkl2rgb(dkl_Nx3, conversionMatrix=None):
    """
    Convert from DKL color space (cone-opponent space from Derrington,
    Krauskopf & Lennie) to RGB. 

    Requires a conversion matrix, which will be generated from generic
    Sony Trinitron phosphors if not supplied (note that this will not be
    an accurate representation of the color space unless you supply a 
    conversion matrix
    

    """
    dkl_3xN = np.transpose(dkl_Nx3)#its easier to use in the other orientation!
    if np.size(dkl_3xN)==3:
        RG, BY, LUM = sph2cart(dkl_3xN[0],dkl_3xN[1],dkl_3xN[2])
    else:
        RG, BY, LUM = sph2cart(dkl_3xN[0,:],dkl_3xN[1,:],dkl_3xN[2,:])
    dkl_cartesian = np.asarray([LUM, RG, BY])

    if conversionMatrix==None:
        conversionMatrix = np.asarray([ \
            #LUMIN	%L-M	%L+M-S  (note that dkl has to be in cartesian coords first!)
            [1.0000, 1.0000, -0.1462],	#R
            [1.0000, -0.3900, 0.2094],	#G
            [1.0000, 0.0180, -1.0000]])	#B
        #log.warning('This monitor has not been color-calibrated. Using default DKL conversion matrix.')
        
    rgb = np.dot(conversionMatrix, dkl_cartesian)
    
    return np.transpose(rgb)#return in the shape we received it
    
def lms2rgb(lms_Nx3, conversionMatrix=None):
    """
    Convert from cone space (Long, Medium, Short) to RGB. 
    
    Requires a conversion matrix, which will be generated from generic
    Sony Trinitron phosphors if not supplied (note that you will not get
    an accurate representation of the color space unless you supply a 
    conversion matrix)

    """
    
    lms_3xN = np.transpose(lms_Nx3)#its easier to use in the other orientation!
        
    if conversionMatrix==None:
        cones_to_rgb = np.asarray([ \
            #L		M		S
            [ 4.97068857, -4.14354132, 0.17285275],#R
            [-0.90913894, 2.15671326, -0.24757432],#G
            [-0.03976551, -0.14253782, 1.18230333]#B
            ])
        #log.warning('This monitor has not been color-calibrated. Using default LMS conversion matrix.')
    else: cones_to_rgb=conversionMatrix
    
    rgb_to_cones = np.linalg.pinv(cones_to_rgb)#get inverse
    rgb = np.dot(cones_to_rgb, lms_3xN)
    return np.transpose(rgb)#return in the shape we received it

def pol2cart(theta, radius, units='deg'):
    """
    
    Convert from polar to cartesian coordinates
    


    """
    if units in ['deg', 'degs']:
        theta = theta*np.pi/180.0
    xx = radius*np.cos(theta)
    yy = radius*np.sin(theta)
    
    return xx,yy


def  cart2pol(x,y, units='deg'):
    """
    Convert from cartesian to polar coordinates
        
    units refers to the units (rad or deg) for theta that should be returned
    
    """
    radius= np.hypot(x,y)
    theta= np.arctan2(y,x)
    if units in ['deg', 'degs']:
        theta=theta*180/np.pi
    return theta, radius


