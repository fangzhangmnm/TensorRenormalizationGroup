#!/usr/bin/python3

import numpy as np
import os
import sys
import modeldata
import tensordispenser
import scaldim_plot
import toolbox
from ed_scaldimer import separate_vector_by_qnum, qnums_from_eigenvectors
from tensors.tensor import Tensor
from tensors.symmetrytensors import TensorZ2
from tensorstorer import write_tensor_file, read_tensor_file
from scon_sparseeig import scon_sparseeig
from timer import Timer
from pathfinder import PathFinder
from custom_parser import parse_argv
from scon import scon

np.set_printoptions(precision=7)
np.set_printoptions(linewidth=100)


def get_id_pars(pars):
    id_pars = dict()
    mandatory_id_pars = {"model", "symmetry_tensors", "dtype",
                         "initial2x2", "initial4x4",
                         "n_normalization", "n_discard", "g",
                         "block_width", "do_eigenvectors", "n_dims_do"}
    modelname = pars["model"].lower().strip()
    if modelname == "ising":
        mandatory_id_pars |= {"J", "H", "beta"}
    elif modelname == "potts3":
        mandatory_id_pars |= {"J", "beta"}
    if pars["symmetry_tensors"]:
        mandatory_id_pars |= {"qnums_do"}
    if not pars["symmetry_tensors"]:
        mandatory_id_pars |= {"sep_qnums"}

    for k in mandatory_id_pars:
        if k in pars:
            id_pars[k] = pars[k]
        else:
            raise RuntimeError("The required parameter %s was not given."%k)
    return id_pars


def parse():
    pars = parse_argv(sys.argv,
                      # Format is: (name_of_argument, type, default)
                      ("model", "str", ""),
                      ("dtype", "dtype", np.complex_),
                      ("J", "float", 1),
                      ("H", "float", 0),
                      ("initial2x2", "bool", False),
                      ("initial4x4", "bool", False),
                      ("n_dims_do", "int", 17),
                      ("qnums_do", "int_list", []),  #[] denotes all
                      ("n_normalization", "int", 3),
                      ("n_discard", "int", 0),
                      ("block_width", "int", 8),
                      ("do_eigenvectors", "bool", False),
                      ("symmetry_tensors", "bool", False),
                      ("sep_qnums", "bool", False),
                      # Sixvertex parameters
                      ("g", "float", 0.),
                      ("gs", "float_list", [0,1,21]),
                      # IO parameters.
                      ("n_dims_plot", "int", 17),
                      ("max_dim_plot", "float", 20000),
                      ("qnums_plot", "int_list", []),  #[] denotes all
                      ("xtick_rotation", "int", 0),
                      ("show_plots", "bool", True),
                      ("save_plots", "bool", False),
                      ("draw_exact_lines", "bool", True),
                      ("draw_exact_circles", "bool", True),
                      ("plot_by_qnum", "bool", True),
                      ("save_scaldim_file", "bool", True))
    pars = vars(pars)

    # - Format parameters -
    pars["plot_by_momenta"] = False
    pars["beta"] = modeldata.get_critical_beta(pars)
    pars["J"] = pars["dtype"](pars["J"])
    pars["H"] = pars["dtype"](pars["H"])
    if (not pars["symmetry_tensors"] and pars["sep_qnums"]
            and not pars["do_eigenvectors"]):
        raise ValueError("sep_qnums requires do_eigenvectors.")
    if pars["model"].strip().lower() != "ising":
        raise NotImplementedError("model != 'ising' unimplemented.")
    if pars["n_dims_plot"] > pars["n_dims_do"]:
        raise ValueError("n_dims_plot > n_dims_do")
    if not set(pars["qnums_plot"]).issubset(set(pars["qnums_do"])):
        raise ValueError("qnums_plot is not included in qnums_do")

    return pars


def get_T(pars):
    # We always get the invariant tensor here, and cast it to the
    # non-invariant if needed. This gets around the silly fact that the
    # basis of the original tensor depends on symmetry_tensors,
    # something that should be fixed.
    T = tensordispenser.get_tensor(
        pars, iter_count=0, symmetry_tensors=True)[0]
    if not pars["symmetry_tensors"]:
        T = Tensor.from_ndarray(T.to_ndarray())
    log_fact = 0
    Fs = []
    Ns = []
    cum = T
    for i in range(1, pars["n_normalization"]):
        cum = toolbox.contract2x2(cum)
        log_fact *= 4
        m = cum.abs().max()
        if m != 0:
            cum /= m
            log_fact += np.log(m)
        N = 4**i
        F = np.log(scon(cum, [1,2,1,2]).value()) + log_fact
        Fs.append(F)
        Ns.append(N)
    A, B = np.polyfit(Ns[pars["n_discard"]:], Fs[pars["n_discard"]:], 1)
    T /= np.exp(A)
    return T


def get_D(t, pars):
    ham = (- pars["J"]*np.array([[ 1,-1],
                                 [-1, 1]],
                                dtype=pars["dtype"])
           + pars["H"]*np.array([[-1, 0],
                                 [ 0, 1]],
                                dtype=pars["dtype"]))
    boltz = np.exp(-pars["beta"]*ham)
    ham_g = - pars["g"]* pars["J"]*np.array([[ 1,-1],
                                             [-1, 1]],
                                            dtype=pars["dtype"])
    boltz_g = np.exp(-pars["beta"]*ham_g)
    ones = np.ones((2,2),  dtype=pars["dtype"])
    D = np.einsum('ab,bc,cd,da->abcd', boltz, boltz_g, boltz_g, boltz)
    u = np.array([[1, 1],
                  [1,-1]]) / np.sqrt(2)
    u_dg = u.T.conjugate()
    D = scon((D, u, u, u_dg, u_dg),
             ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))
    if pars["symmetry_tensors"]:
        D = TensorZ2.from_ndarray(D, shape=[[1,1]]*4, qhape=[[0,1]]*4,
                                  dirs=[1,1,-1,-1])
    else:
        D = Tensor.from_ndarray(D)
    return D


def get_eigs(pars):
    print("Building the matrix to diagonalize.")
    T = get_T(pars)
    D = get_D(type(T), pars)

    # Get the eigenvalues and their logarithms.
    block_width = pars["block_width"]
    n_dims_do = pars["n_dims_do"]

    translation = range(block_width)
    scon_list = [D] + [T]*(block_width-1)
    index_list = [[block_width*2, -101, 2, -1]]
    for i in range(2, block_width+1):
        index_list += [[2*i-2, -100-i, 2*i, -(2*i-1)]]

    res = scon_sparseeig(scon_list, index_list, translation,
                         range(block_width), hermitian=True,
                         return_eigenvectors=pars["do_eigenvectors"],
                         qnums_do=pars["qnums_do"],
                         maxiter=500, tol=1e-8, k=pars["n_dims_do"])
    if pars["do_eigenvectors"]:
        es, evects = res
    else:
        es = res

    # Convert es to complex for taking the log.
    es = es.astype(np.complex_, copy=False)
    if (not pars["symmetry_tensors"]) and pars["sep_qnums"]:
        qnums = qnums_from_eigenvectors(evects, pars)
        es = separate_vector_by_qnum(es, qnums, pars)

    ret_val = (es,)
    if pars["do_eigenvectors"]:
        ret_val += (evects,)
    if len(ret_val) > 1:
        return ret_val
    else:
        return ret_val[0]


def load_eigs(pars, **kwargs):
    if kwargs:
        pars = pars.copy()
        pars.update(kwargs)
    id_pars = get_id_pars(pars)
    filename = os.path.basename(__file__)
    try:
        res = read_tensor_file("cdf_ed_eigs", pars=id_pars, filename=filename)
    except RuntimeError:
        print("Constructing eigs.")
        timer = Timer()
        timer.start()
        res = get_eigs(pars)
        print("Done constructing eigs.\n")
        timer.print_elapsed()
        if pars["save_scaldim_file"]:
            write_tensor_file(data=res, prefix="cdf_ed_eigs", pars=id_pars,
                              filename=filename)
    return res


#=============================================================================#


if __name__ == "__main__":
    pars = parse()
    id_pars = get_id_pars(pars)
    filename = os.path.basename(__file__)
    pather = PathFinder(filename, id_pars)

    # - Infoprint -
    print("\n" + ("="*70) + "\n")
    print("Running %s with the following parameters:"%filename)
    for k,v in sorted(pars.items()):
        print("%s = %s"%(k, v))

    maxi = 5
    leigs_by_theta = {}
    for g in np.linspace(*pars["gs"]):
        res = load_eigs(pars, g=g)
        if pars["do_eigenvectors"]:
            eigs = res[0]
        else:
            eigs = res


        theta = np.arctan((1-g)/(1+g))

        leigs = eigs.log()
        leigs *= -pars["block_width"]/(2*np.pi)
        if hasattr(leigs, "sects"):
            leigs -= leigs[(0,)][0]
            for qnum, sect in leigs.sects.items():
                leigs[qnum] = sect[sect<maxi]
        else:
            leigs -= leigs[0]
            leigs = leigs[leigs<maxi]
        leigs_by_theta[theta] = leigs

    exacts_by_theta = {}
    exact_degs_by_theta = {}
    for g in {0,1}:
        theta = np.arctan((1-g)/(1+g))
        prim_data = modeldata.get_primary_data(maxi*2, pars, 0, g=g)
        exacts_by_theta[theta] = np.array(prim_data[0]) * (0.5 if g == 0
                                                           else 1)
        exact_degs_by_theta[theta] = prim_data[2]

    scaldim_plot.plot_dict(leigs_by_theta, pars, x_label=r"$\theta$",
                           exacts_dict=exacts_by_theta,
                           exact_degs_dict=exact_degs_by_theta,
                           y_label=r"$\Delta_\alpha(\theta)$",
                           id_pars=id_pars)

