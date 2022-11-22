#!/usr/bin/python3

""" A module for writing and reading tensor files based on the parameters used
to create them.
"""

import pickle
import os
from pathfinder import PathFinder


def update_pars_and_pather(pars, pather, filename, **kwargs):
    if kwargs:
        pars = pars.copy()
        pars.update(kwargs)
        if pather is not None:
            warnings.warn("tensorstorer was given both a pather and kwargs. "
                          "Generating a new pather.")
    if kwargs or pather is None:
        # If kwargs, we need to make a pather to make sure that path
        # matches contents.
        pather = PathFinder(filename, pars) 
    return pars, pather


def read_tensor_file(prefix="", pars={}, pather=None, filename=None, **kwargs):
    pars, pather = update_pars_and_pather(pars, pather, filename, **kwargs)
    path = pather.generate_path(prefix, extension=".p")
    if not os.path.isfile(path):
        # This is meant to be caught by the caller.
        raise RuntimeError
    print("Reading %s from %s"%(prefix, path))
    f = open(path, 'rb')
    res = pickle.load(f)
    f.close()
    return res


def write_tensor_file(data=None, prefix="", pars={}, pather=None,
                      filename=None, **kwargs):
    pars, pather = update_pars_and_pather(pars, pather, filename, **kwargs)
    path = pather.generate_path(prefix, extension=".p")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Writing %s to %s"%(prefix, path))
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()
    write_pars_file(prefix=prefix, pars=pars, pather=pather)


def write_pars_file(prefix="", pars={}, pather=None, filename=None, **kwargs):
    pars, pather = update_pars_and_pather(pars, pather, filename, **kwargs)
    path = pather.generate_path(prefix, extension=".pars")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, 'w')
    for k,v in sorted(pars.items()):
        print("%s = %s"%(k, v), file=f)
    f.close()

