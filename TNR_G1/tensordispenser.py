#!/usr/bin/python3

""" A module that generates and stores various results of different
coarse-graining algorithms for different lattice models. The point is
that when a tensor is requested, the module checks whether it already is
stored on the hard drive, and returns it if it is. If not it generates it,
stores it on the hard drive and returns it.
"""

import numpy as np
import toolbox
import initialtensors
import os
import argparse
from tensorstorer import write_tensor_file, read_tensor_file
from timer import Timer
from matplotlib import pyplot
from TNR import tnr_step
from scon import scon
from pathfinder import PathFinder
from custom_parser import parse_argv

filename = os.path.basename(__file__)
global_timer = Timer()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions for getting different tensors. These are the user interface.

def get_general(prefix, generator, pars, **kwargs):
    """ A general getter function that either gets the asked-for data
    from a file or generates it with the given generator function.  """
    pars = get_pars(pars, **kwargs)
    id_pars, pars = get_id_pars_and_set_default_pars(pars)
    try:
        result = read_tensor_file(prefix=prefix, pars=id_pars,
                                  filename=filename)
    except RuntimeError:
        result = generator(pars, id_pars)
    return result

def get_tensor(pars=None, infotime=True, **kwargs):
    generator = lambda p, i: generate_tensor(p, i, infotime=infotime)[0:2]
    T, log_fact = get_general("tensor", generator, pars, **kwargs)
    return T, log_fact

def get_normalized_tensor(pars=None, infotime=True, **kwargs):
    generator = generate_normalized_tensor  
    T = get_general("tensor_normalized", generator, pars, **kwargs)
    return T

def get_gauges(pars=None, infotime=True, **kwargs):
    kwargs["return_gauges"] = True
    generator = lambda p, i: generate_tensor(p, i, infotime=infotime)[-1]
    gauges = get_general("gauges", generator, pars, **kwargs)
    return gauges

def get_pieces(pars=None, infotime=True, **kwargs):
    kwargs["return_pieces"] = True
    generator = lambda p, i: generate_tensor(p, i, infotime=infotime)[2]
    pieces = get_general("pieces", generator, pars, **kwargs)
    return pieces


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions for modifying the given parameters to the needed form and
# sorting extra parameters from the ones that are important for
# identifying the tensors.

def get_pars(pars, **kwargs):
    if pars is None:
        return kwargs
    else:
        new_pars = pars.copy()
        new_pars.update(kwargs)
        return new_pars


# Parameters that need to be always given and will be used for
# identifying files.
global_mandatory_id_pars = {"dtype", "iter_count",
                            "initial2x2", "initial4x4", "symmetry_tensors",
                            "model"}

# Parameters that need to be always given, depend on the model and will
# be used for identifying files.
model_id_pars = {}
model_id_pars["ising"] = {"J", "H", "beta"}
model_id_pars["potts3"] = {"J", "beta"}

# Parameters that need to be always given, depend on the algorithm and
# will be used for identifying files.
algorithm_mandatory_id_pars = {}
algorithm_mandatory_id_pars["tnr"] = {"chis_tnr", "chis_trg", "opt_eps_conv",
                                      "horz_refl", "opt_max_iter",
                                      "opt_iters_tens"}
algorithm_mandatory_id_pars["trg"] = {"chis", "J", "H"}

# Parameters that may be given, depend on the algorithm and will be used
# for identifying files. If not given, the default (the second element
# in the tuple) will be used.
algorithm_optional_id_pars = {}
algorithm_optional_id_pars["tnr"] = {("A_chis", None),
                                     ("A_eps", 0),
                                     ("opt_eps_chi", 0),
                                     ("fix_gauges", False),
                                     ("reuse_initial", False)}
algorithm_optional_id_pars["trg"] = {("eps", 0)}

# Parameters that may be given and will NOT be used for identifying
# files. If not given, the default (the second element in the tuple)
# will be used.
optional_other_pars = {("save_errors", False),
                       ("print_errors", 0),
                       ("return_gauges", False),
                       ("return_pieces", False),
                       ("save_fit_plot", False)}

def get_id_pars_and_set_default_pars(pars):
    """ Make a copy of pars and populate with defaults as needed. Also
    copy from pars to id_pars the parameters by which different tensors
    should be identified, also using defaults for some of the values as
    needed.
    """
    new_pars = pars.copy()
    id_pars = {}
    
    mandatory_id_pars = set()
    optional_id_pars = set()
    # The following are necessary regardless of algorithm and model.
    model_name = pars["model"].strip().lower()
    mandatory_id_pars |= global_mandatory_id_pars.copy()
    mandatory_id_pars |= model_id_pars[model_name]
    if pars["iter_count"] > 0:
        algorithm_name = pars["algorithm"].strip().lower()
        mandatory_id_pars.add("algorithm")
        mandatory_id_pars |= algorithm_mandatory_id_pars[algorithm_name]
        optional_id_pars |= algorithm_optional_id_pars[algorithm_name]

    for k in mandatory_id_pars:
        if k in pars:
            id_pars[k] = pars[k]
        else:
            raise RuntimeError("The required parameter %s was not given."%k)
    for t in optional_id_pars:
        k = t[0]
        d = t[1]
        id_pars[k] = pars.get(k, d)
    for t in optional_other_pars:
        new_pars.setdefault(*t)
    return id_pars, new_pars


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions for generating tensors.

def generate_first_tensor(pars):
    T = initialtensors.get_initial_tensor(pars)
    log_fact = 0
    gauges = {}
    pieces = {}
    if pars["initial4x4"]:
        # Construct G_vh
        dim = T.shape[0]
        try:
            qim = T.qhape[0]
        except TypeError:
            qim = None
        eye = type(T).eye(dim=dim, qim=qim)
        u = scon((eye, eye, eye, eye), ([-1,-5], [-2,-6], [-3,-7], [-4,-9]))
        swap = u.transpose((0,1,2,3,7,6,5,4))
        swap = swap.join_indices([0,1,2,3], [4,5,6,7], dirs=[1,1])
        gauges["G_vh"] = swap

        # Construct G_hv
        dim = T.shape[1]
        try:
            qim = T.qhape[1]
        except TypeError:
            qim = None
        eye = type(T).eye(dim=dim, qim=qim)
        u = scon((eye, eye, eye, eye), ([-1,-5], [-2,-6], [-3,-7], [-4,-9]))
        swap = u.transpose((0,1,2,3,7,6,5,4))
        swap = swap.join_indices([0,1,2,3], [4,5,6,7], dirs=[1,1])
        gauges["G_hv"] = swap

        # Contract T
        T = toolbox.contract2x2(T)
        T = toolbox.contract2x2(T)
    elif pars["initial2x2"]:
        # Construct G_vh
        dim = T.shape[0]
        try:
            qim = T.qhape[0]
        except TypeError:
            qim = None
        eye = type(T).eye(dim=dim, qim=qim)
        u = scon((eye, eye), ([-1,-3], [-2,-4]))
        swap = u.transpose((0,1,3,2))
        swap = swap.join_indices([0,1], [2,3], dirs=[1,1])
        gauges["G_vh"] = swap
        
        # Construct G_hv
        dim = T.shape[1]
        try:
            qim = T.qhape[1]
        except TypeError:
            qim = None
        eye = type(T).eye(dim=dim, qim=qim)
        u = scon((eye, eye), ([-1,-3], [-2,-4]))
        swap = u.transpose((0,1,3,2))
        swap = swap.join_indices([0,1], [2,3], dirs=[1,1])
        gauges["G_hv"] = swap

        # Contract T
        T = toolbox.contract2x2(T)
    return T, log_fact, pieces, gauges


def generate_next_tensor(pars):
    algo_name = pars["algorithm"].strip().lower()
    # Get the tensor from the previous step.
    T, log_fact = get_tensor(pars, iter_count=pars["iter_count"]-1,
                             infotime=False)
    print('\n / Coarse-graining, iter_count = #%i: / '%(pars["iter_count"]))
    if algo_name == "tnr":
        gauges = {}
        pieces = {}
        if pars["horz_refl"]:
            gauges = get_gauges(pars, iter_count=pars["iter_count"]-1,
                                infotime=False)
        if pars["reuse_initial"] or pars["fix_gauges"]:
            pieces = get_pieces(pars, iter_count=pars["iter_count"]-1,
                                infotime=False)
        tnr_result = tnr_step(T, pars=pars, gauges=gauges, pieces=pieces,
                              log_fact=log_fact)
        T, log_fact = tnr_result[0:2]
        if pars["return_pieces"]:
            pieces = tnr_result[2]
        if pars["return_gauges"]:
            gauges = tnr_result[-1]
    elif algo_name == "trg":
        pieces = None
        gauges = None
        T, log_fact = trg_step(T, pars=pars, log_fact=log_fact)
    return T, log_fact, pieces, gauges


def generate_tensor(pars, id_pars, infotime=True):
    if infotime:
        # - Infoprint and start timer -
        print("\n" + ("="*70) + "\n")
        print("Generating coarse-grained tensor with the following "
              "parameters:")
        for k,v in sorted(pars.items()):
            print("%s = %s"%(k, v))
        global_timer.start()

    if pars["iter_count"] == 0:
        T, log_fact, pieces, gauges = generate_first_tensor(pars)
    else:
        algo_name = pars["algorithm"].strip().lower()
        T, log_fact, pieces, gauges = generate_next_tensor(pars)
        # Save to file(s)
        pather = PathFinder(filename, id_pars) 
        write_tensor_file(data=(T, log_fact), prefix="tensor", pars=id_pars,
                          pather=pather)

        if algo_name == "tnr" and pars["return_pieces"]:
            write_tensor_file(data=pieces, prefix="pieces", pars=id_pars,
                              pather=pather)
        
        if algo_name == "tnr" and pars["return_gauges"]:
            write_tensor_file(data=gauges, prefix="gauges", pars=id_pars,
                              pather=pather)

    if infotime:
        print("\nDone generating the coarse-grained tensor.")
        global_timer.print_elapsed()
        global_timer.stop()
        print()

    return_value = (T, log_fact)
    if "algorithm" in pars and pars["algorithm"].strip().lower() == "tnr":
        if pars["return_pieces"]:
            return_value += (pieces,)
        if pars["return_gauges"]:
            return_value += (gauges,)
    return return_value


def generate_normalized_tensor(pars, id_pars):
    # - Infoprint and start timer -
    print("\n" + ("="*70) + "\n")
    print("Generating the normalized, coarse-grained tensor with the "
          "following parameters:")
    for k,v in sorted(pars.items()):
        print("%s = %s"%(k, v))
    global_timer.start()

    algo_name = pars["algorithm"].strip().lower()
    # Number of tensors to use to fix the normalization
    n = max(8, pars["iter_count"] + 4)
    # Number of tensors from the beginning to discard
    n_discard = max(min(pars["iter_count"]-3, 3), 0)
    tensors_and_log_facts = []
    for i in range(n+1):
        T, log_fact = get_tensor(pars=pars, iter_count=i, infotime=False)
        tensors_and_log_facts.append((T, log_fact))
    tensors, log_facts = zip(*tensors_and_log_facts)
    Zs = np.array([scon(T, [1,2,1,2]).norm() for T in tensors])
    log_Zs = np.log(Zs)
    log_Zs += np.array(log_facts)

    if algo_name == "tnr":
        Ns = np.array([2*4**i for i in range(n+1)])
    elif algo_name == "trg":
        Ns = np.array([2*2**i for i in range(n+1)])
    if pars["initial4x4"]:
        Ns *= 16
    elif pars["initial2x2"]:
        Ns *= 4
    A, B = np.polyfit(Ns[pars["n_discard"]:], log_Zs[pars["n_discard"]:], 1)
    tensors = [T / np.exp(N*A - log_fact)
               for T, N, log_fact in zip(tensors, Ns, log_facts)]

    if pars["print_errors"]:
        print("Fit when normalizing Ts: %.3e * N + %.3e"%(A,B))

    if pars["save_fit_plot"]:
        pyplot.plot(Ns, log_Zs, marker='*', linestyle='')
        pyplot.plot(Ns, A*Ns+B)
        pather = PathFinder(filename, id_pars, ignore_pars=['iter_count'])
        path = pather.generate_path("Normalization_fit", extension='.pdf')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pyplot.savefig(path)
        pyplot.clf()

    for i, T in enumerate(tensors):
        write_tensor_file(data=T, prefix="tensor_normalized", pars=id_pars,
                          filename=filename, iter_count=i)

    print("Returning normalized tensor.")
    global_timer.print_elapsed()
    global_timer.stop()
    return tensors[pars["iter_count"]]

