import glob as glob
import os


def getAllFiles(dirName, suffix = None, subdirectories = 1):
    """
    Get a list of path names of all files in a directory.
    
    :param Directory: a directory.
    :type Directory: str
    :param suffix: find only files with a specific ending.
    :type suffix: str
    :param subdirectories: indicate how deep (# of directories) you would \
    like to search: 0 = working directory.
    :type subdirectories: int
    
    :returns: a list of path names.
    :rtype: list

    e.g. subdirectories = 1: Find all files within a directory and its 
    first layer of subdirectories.
    
    """
    if suffix is None:
        suffix = ''

    depth = '/*'
    for i in range(subdirectories):
        depth += depth

    f = dirName + depth + suffix

    files = []
    for name in glob.glob(f):
        files.append(name)
    
    return files		


def make_dir(directory):
    ''' Check if a directory exists and make one if it does not'''
    directory = os.path.dirname(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)


# Below from PsychoPy library. Copyright (C) 2009 Jonathan Peirce
# Distributed under the terms of the GNU General Public License (GPL).
def toFile(filename, data):
    """
    
    save data (of any sort) as a pickle file
    
    simple wrapper of the cPickle module in core python
    """
    f = open(filename, 'w')
    cPickle.dump(data,f)
    f.close()

def fromFile(filename):
    """
    
    load data (of any sort) from a pickle file
    
    simple wrapper of the cPickle module in core python
    
    """
    f = open(filename)
    contents = cPickle.load(f)
    f.close()
    return contents
