#!/usr/bin/python3

import numpy as np
import os
import sys
import modeldata
import tensordispenser
import initialtensors
import operator
import itertools
import warnings
import scaldim_plot
import toolbox
from tensors.tensor import Tensor
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
                         "n_normalization", "n_discard",
                         "block_width", "defect_angles", "KW",
                         "do_momenta", "do_eigenvectors", "n_dims_do"}
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
                      ("defect_angles", "float_list", [0]),
                      ("KW", "bool", False),
                      ("do_eigenvectors", "bool", False),
                      ("do_momenta", "bool", False),
                      ("symmetry_tensors", "bool", False),
                      ("sep_qnums", "bool", False),
                      # IO parameters.
                      ("n_dims_plot", "int", 17),
                      ("max_dim_plot", "float", 20000),
                      ("qnums_plot", "int_list", []),  #[] denotes all
                      ("xtick_rotation", "int", 0),
                      ("show_plots", "bool", True),
                      ("save_plots", "bool", False),
                      ("draw_exact_lines", "bool", True),
                      ("draw_exact_circles", "bool", True),
                      ("draw_defect_angle", "bool", True),
                      ("plot_by_qnum", "bool", True),
                      ("plot_by_momenta", "bool", False),
                      ("plot_by_alpha", "bool", False),
                      ("save_scaldim_file", "bool", True))
    pars = vars(pars)

    # - Format parameters -
    pars["beta"] = modeldata.get_critical_beta(pars)
    pars["J"] = pars["dtype"](pars["J"])
    pars["H"] = pars["dtype"](pars["H"])
    pars["defect_angles"] = sorted(pars["defect_angles"], key=abs)
    if (not pars["symmetry_tensors"] and pars["sep_qnums"]
            and not pars["do_eigenvectors"]):
        raise ValueError("sep_qnums requires do_eigenvectors.")
    if pars["defect_angles"] != [0] and pars["KW"]:
        raise ValueError("Non-trivial defect_angles and KW.")
    if pars["n_dims_plot"] > pars["n_dims_do"]:
        raise ValueError("n_dims_plot > n_dims_do")
    if not set(pars["qnums_plot"]).issubset(set(pars["qnums_do"])):
        raise ValueError("qnums_plot is not included in qnums_do")


    return pars

def get_defect(alpha, T, index):
    # Build a group-like defect.
    dim = T.shape[index]
    try:
        qim = T.qhape[index]
    except TypeError:
        qim = None
    defect = type(T).eye(dim, qim=qim, dtype=T.dtype)
    if alpha != 0:
        for k,v in defect.sects.items():
            phase = np.exp(1j*alpha*k[0])
            defect[k] = v*phase
    return defect


def get_T(pars):
    # We always get the invariant tensor here, and cast it to the
    # non-invariant one later if we so wish.
    # This makes constructing the defects easier.
    T = tensordispenser.get_tensor(pars, iter_count=0,
                                   symmetry_tensors=True)[0]
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


def get_T_first(T, pars, alpha=0):
    if pars["KW"]:
        T_first = initialtensors.get_KW_tensor(pars)
    elif pars["do_momenta"]:
        defect_horz = get_defect(alpha, T, 0)
        defect_vert = get_defect(alpha, T, 1).conjugate().transpose()
        T_first = scon((T, defect_horz, defect_vert),
                      ([1,-2,-3,4], [-1,1], [4,-4]))
    else:
        defect_horz = get_defect(alpha, T, 0)
        T_first = scon((T, defect_horz), ([1,-2,-3,-4], [-1,1]))
    return T_first


def qnums_from_eigenvectors(evects, pars):
    sites = len(evects.shape) - 1
    if pars["model"].strip().lower() == "ising":
        symop = np.array([[1,0], [0,-1]], dtype=np.float_)
        symop = type(evects).from_ndarray(symop)
    else:
        # TODO generalize symop to models other than Ising.
        NotImplementedError("Symmetry operators for models other than "
                            "Ising have not been implemented.")
    ncon_list = (evects, evects.conjugate()) + (symop,)*sites
    index_list = ((list(range(1,sites+1)) + [-1],
                   list(range(sites+1, 2*sites+1)) + [-2])
                  + tuple([i+sites,i] for i in range(1,sites+1)))
    qnums = scon(ncon_list, index_list)
    qnums = qnums.diag()
    # Round to the closest possible qnum for the given model.
    pos_qnums = initialtensors.symmetry_classes_dims_qims[pars["model"]][2]
    max_qnum = max(pos_qnums)
    qnums = qnums.astype(np.complex_).log()*(max_qnum+1)/(2j*np.pi)
    qnums = [min(pos_qnums, key=lambda x: abs(x-q)) for q in qnums]
    return qnums


def separate_vector_by_qnum(v, qnums, pars):
    # Get the right tensor type and the possible qnums for this model.
    symdata = initialtensors.symmetry_classes_dims_qims[pars["model"]]
    T, pos_qnums = symdata[0], symdata[2]
    vals = {}
    for q in pos_qnums:
        vals[q] = []
    for s, q in zip(v, qnums):
        vals[q].append(s)
    dim = [len(vals[q]) for q in pos_qnums]
    tensor = T.empty(shape=(dim,), qhape=(pos_qnums,), dirs=[1], invar=False)
    for q in pos_qnums:
        block = np.array(vals[q])
        tensor[(q,)] = block
    print(tensor)
    return tensor


def get_cft_data(pars):
    T = get_T(pars)

    scaldims_by_alpha = {}
    if pars["do_momenta"]:
        momenta_by_alpha = {}
    if pars["do_eigenvectors"]:
        evects_by_alpha = {}
    for alpha in pars["defect_angles"]:
        print("Building the matrix to diagonalize.")
        T_first = get_T_first(T, pars, alpha=alpha)

        # Get the eigenvalues and their logarithms.
        block_width = pars["block_width"]
        n_dims_do = pars["n_dims_do"]

        if pars["do_momenta"]:
            translation = list(range(1, block_width)) + [0]
        else:
            translation = range(block_width)
        scon_list = [T_first] + [T]*(block_width-1)
        index_list = [[block_width*2, -101, 2, -1]]
        for i in range(2, block_width+1):
            index_list += [[2*i-2, -100-i, 2*i, -(2*i-1)]]
        if pars["KW"] and pars["do_momenta"]:
            U = initialtensors.get_KW_unitary(pars)
            scon_list += [U]
            for l in index_list:
                l[0] += 2
                l[2] += 2
                l[3] += -2
            index_list[0][3] *= -1
            index_list[1][3] *= -1
            U_indices = [3,5,-1,-2]
            index_list.append(U_indices)

        if not pars["symmetry_tensors"]:
            # Cast to non-invariant tensors.
            scon_list = [Tensor.from_ndarray(T.to_ndarray())
                         for T in scon_list]

        hermitian = not pars["do_momenta"]
        res = scon_sparseeig(scon_list, index_list, translation,
                             range(block_width), hermitian=hermitian,
                             return_eigenvectors=pars["do_eigenvectors"],
                             qnums_do=pars["qnums_do"],
                             maxiter=500, tol=1e-8, k=pars["n_dims_do"])
        if pars["do_eigenvectors"]:
            es, evects = res
        else:
            es = res

        # Convert es to complex for taking the log.
        es = es.astype(np.complex_, copy=False)
        # Log and scale the eigenvalues.
        block_width = pars["block_width"]
        if pars["KW"]:
            block_width -= 0.5
        log_es = es.log() * block_width / (2*np.pi)

        # Extract the central charge.
        if alpha == 0:
            if pars["KW"]:
                c = (log_es.real().max() + 0.0625) * 12
            else:
                c = log_es.real().max() * 12
        try:
            log_es -= c/12
        except NameError:
            raise ValueError("Need to provide 0 in defect_angles to be able "
                             "to obtain the central charge.")
        log_es *= -1
        scaldims = log_es.real()
        if (not pars["symmetry_tensors"]) and pars["sep_qnums"]:
            qnums = qnums_from_eigenvectors(evects, pars)
            scaldims = separate_vector_by_qnum(scaldims, qnums, pars)
        scaldims_by_alpha[alpha] = scaldims
        if pars["do_momenta"]:
            momenta = log_es.imag()
            if (not pars["symmetry_tensors"]) and pars["sep_qnums"]:
                momenta = separate_vector_by_qnum(momenta, qnums, pars)
            momenta_by_alpha[alpha] = momenta
        if pars["do_eigenvectors"]:
            evects_by_alpha[alpha] = evects
    ret_val = (scaldims_by_alpha, c)
    if pars["do_momenta"]:
        ret_val += (momenta_by_alpha,)
    if pars["do_eigenvectors"]:
        ret_val += (evects_by_alpha,)
    return ret_val


def load_cft_data(pars, **kwargs):
    if kwargs:
        pars = pars.copy()
        pars.update(kwargs)
    id_pars = get_id_pars(pars)
    filename = os.path.basename(__file__)
    try:
        res = read_tensor_file("scals_by_alpha", pars=id_pars,
                               filename=filename)
    except RuntimeError:
        print("Constructing scaling dimensions.")
        timer = Timer()
        timer.start()
        res = get_cft_data(pars)
        print("Done constructing scaling dimensions.\n")
        timer.print_elapsed()
        if pars["save_scaldim_file"]:
            write_tensor_file(data=res, prefix="scals_by_alpha",
                              pars=id_pars, filename=filename)
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

    res = load_cft_data(pars)
    scaldims_by_alpha, c = res[:2]
    if pars["do_momenta"]:
        momenta_by_alpha = res[2]
    else:
        momenta_by_alpha = None

    scaldim_plot.plot_and_print_dict(scaldims_by_alpha, c, pars, pather,
                                     momenta_dict=momenta_by_alpha,
                                     id_pars=id_pars)

