import numpy as np
import collections


IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])

def irc2xyz(irc, origin_xyz,vx_size_xyz):
    #the arguments are all tuple
    #IRC to CRI
    cri = irc[::-1] #3,3 @ 3, -> 3,
    return  cri * np.array(vx_size_xyz) + origin_xyz
 
def xyz2irc(xyz, origin_xyz,vx_size_xyz):
    #the arguments are all tuple
    #xyz to irc
    cri = (np.array(xyz, dtype = np.float32) - np.array(origin_xyz, dtype = np.float32 ) )/np.array(vx_size_xyz)
    print(cri)
    irc  = cri[::-1]
    return IrcTuple(*np.round(irc).astype(int))


def importer(name, root_package=False, relative_globals=None, level=0):
    """ We only import modules, functions can be looked up on the module.
    Usage: 

    from foo.bar import baz
    >>> baz = importer('foo.bar.baz')

    import foo.bar.baz
    >>> foo = importer('foo.bar.baz', root_package=True)
    >>> foo.bar.baz

    from .. import baz (level = number of dots)
    >>> baz = importer('baz', relative_globals=globals(), level=2)

    The fromlist should be:
    -  a list of names to emulate from name import ...'', 
    -  an empty list to emulate import name

    """
    return __import__(name, locals=None, # locals has no use
                      globals=relative_globals, 
                      fromlist=[] if root_package else [None],
                      level=level)

    

    