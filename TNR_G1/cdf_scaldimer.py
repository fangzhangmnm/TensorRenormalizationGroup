# Written in python3.

import numpy as np
import sys
import os
import tensordispenser
import initialtensors
import modeldata
import scipy.sparse.linalg as spsla
import scaldim_plot
from tensorstorer import write_tensor_file, read_tensor_file
from tensors.symmetrytensors import TensorZ2
from tensors.tensor import Tensor
from functools import partial
from TNR import tnr_step_vertstr
from scon import scon
from pathfinder import PathFinder
from custom_parser import parse_argv

np.set_printoptions(precision=7)
np.set_printoptions(linewidth=100)

# Constants
filename = os.path.basename(__file__)

def get_tensor_id_pars(pars):
    id_pars = dict()
    mandatory_id_pars = {"symmetry_tensors", "dtype", "iter_count"}
    mandatory_id_pars |= {"chis_tnr", "chis_trg", "opt_eps_conv",
                          "opt_eps_chi", "opt_iters_tens", "opt_max_iter",
                          "A_eps", "A_chis", "return_pieces", "g",
                          "reuse_initial", "fix_gauges"}
    mandatory_id_pars |= {"J", "H", "beta"}
    for k in mandatory_id_pars:
        if k in pars:
            id_pars[k] = pars[k]
        else:
            raise RuntimeError("The required parameter %s was not given."%k)
    return id_pars


def get_scaldim_id_pars(pars):
    id_pars = dict()
    mandatory_id_pars = {"symmetry_tensors", "dtype", "iter_count",
                         "n_discard", "block_width", "n_dims_do"}
    mandatory_id_pars |= {"chis_tnr", "chis_trg", "opt_eps_conv",
                          "opt_eps_chi", "opt_iters_tens", "opt_max_iter",
                          "A_eps", "A_chis", "return_pieces",
                          "g", "reuse_initial", "fix_gauges"}
    mandatory_id_pars |= {"J", "H", "beta"}
    for k in mandatory_id_pars:
        if k in pars:
            id_pars[k] = pars[k]
        else:
            raise RuntimeError("The required parameter %s was not given."%k)
    return id_pars


def parse():
    pars = parse_argv(sys.argv,
                      # Format is: (name_of_argument, type, default)
                      # Parameters for TNR
                      ("chis_tnr", "int_list", [4]),
                      ("chis_trg", "int_list", [6]),
                      ("opt_eps_chi", "float", 1e-8),
                      ("opt_eps_conv", "float", 1e-11),
                      ("opt_iters_tens", "int", 1),
                      ("opt_max_iter", "int", 10000),
                      ("A_eps", "float", 1e-11),
                      ("A_chis", "int_list", None),
                      ("print_errors", "int", 2),
                      ("symmetry_tensors", "bool", True),
                      ("return_pieces", "bool", True),
                      ("reuse_initial", "bool", True),
                      # Parameters about what to run TNR on
                      ("iter_count", "int", 6),  # Number of iterations
                      ("block_width", "int", 2),
                      ("dtype", "dtype", np.complex_),
                      ("J", "float", 1),  # Coupling constant
                      ("H", "float", 0),  # External magnetic field
                      ("g", "float", 0),
                      ("gs", "float_list", [0,1,21]),
                      # Flags about which quantities to calculate and
                      # plot and how.
                      ("n_dims_do", "int", 15),
                      ("n_dims_plot", "int", 15),
                      ("xtick_rotation", "int", 45),
                      ("max_dim_plot", "float", 20000),
                      ("n_discard", "int", 0),
                      ("plot_by_qnum", "bool", True),
                      ("draw_exact_lines", "bool", True),
                      ("draw_exact_circles", "bool", True),
                      ("show_plots", "bool", True),
                      ("save_plots", "bool", False))
    pars = vars(pars)


    # - Format parameters and set constants -
    pars["horz_refl"] = False
    pars["initial2x2"] = False
    pars["initial4x4"] = False
    pars["fix_gauges"] = False
    pars["qnums_plot"] = []
    pars["model"] = "Ising"
    pars["algorithm"] = "TNR"
    pars["beta"] = modeldata.get_critical_beta(pars)
    pars["J"] = pars["dtype"](pars["J"])
    pars["H"] = pars["dtype"](pars["H"])
    pars["return_gauges"] = True
    pars["plot_by_momenta"] = False
    return pars


def generate_initial_tensors(pars):
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
    if pars["symmetry_tensors"]:
        u = np.array([[1, 1],
                      [1,-1]]) / np.sqrt(2)
        u_dg = u.T.conjugate()
        D = scon((D, u, u, u_dg, u_dg),
                 ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))
        D = TensorZ2.from_ndarray(D, shape=[[1,1]]*4, qhape=[[0,1]]*4,
                                  dirs=[1,1,-1,-1])
    else:
        D = Tensor.from_ndarray(D)
    T = initialtensors.get_initial_tensor(pars, iter_count=0)
    A_list = [D, T]
    log_fact_list = [0,0]
    return A_list, log_fact_list


def ascend_tensors(A_list, log_fact_list, pars):
    print('\n / Coarse-graining D, iter_count = #%i: / '%(pars["iter_count"]))
    i = pars["iter_count"]
    pieces = tensordispenser.get_pieces(pars, iter_count=i)
    if i > 1:
        gauges = tensordispenser.get_gauges(pars, iter_count=i-1)
    else:
        gauges = {}
    T, T_log_fact = tensordispenser.get_tensor(pars, iter_count=i-1)
    A_list, log_fact_list, pieces_vertstr =\
            tnr_step_vertstr(A_list, T_log_fact=T_log_fact, pieces=pieces,
                             gauges=gauges, pars=pars,
                             log_fact_list=log_fact_list)
    return A_list, log_fact_list


def get_tensors(pars, **kwargs):
    pars = pars.copy()
    pars.update(kwargs)
    id_pars= get_tensor_id_pars(pars)
    try:
        result = read_tensor_file(prefix="cdf_tensors", pars=id_pars,
                                  filename=filename)
    except RuntimeError:
        result = generate_tensors(pars)
    return result


def generate_tensors(pars, **kwargs):
    pars = pars.copy()
    pars.update(kwargs)
    if pars["iter_count"] == 0:
        A_list, log_fact_list = generate_initial_tensors(pars)
    else:
        # Get the tensor from the previous step
        A_list, log_fact_list = get_tensors(pars,
                                            iter_count=pars["iter_count"]-1)
        # and ascend it.
        A_list, log_fact_list = ascend_tensors(A_list, log_fact_list, pars)
        # Save to file.
        id_pars = get_tensor_id_pars(pars)
        write_tensor_file((A_list, log_fact_list), prefix="cdf_tensors",
                          pars=id_pars, filename=filename)
    return A_list, log_fact_list


def get_normalized_tensors(pars, **kwargs):
    pars = pars.copy()
    pars.update(kwargs)
    id_pars= get_tensor_id_pars(pars)
    try:
        result = read_tensor_file(prefix="cdf_tensors_normalized",
                                  pars=id_pars, filename=filename)
    except RuntimeError:
        result = generate_normalized_tensors(pars)
    return result


def generate_normalized_tensors(pars):
    # Number of tensors to use to fix the normalization
    n = max(8, pars["iter_count"] + 4)
    # Number of tensors from the beginning to discard
    tensors_and_log_facts = []
    for i in reversed(list(range(n+1))):
        tensors_and_log_facts.append(get_tensors(pars=pars, iter_count=i))
    tensors_and_log_facts = tuple(reversed(tensors_and_log_facts))
    A_lists = tuple(map(lambda t: t[0], tensors_and_log_facts))
    log_fact_lists = np.array(tuple(map(lambda t: t[1],
                                        tensors_and_log_facts)))

    Zs = np.array(tuple(scon(A_list, ([1,2,3,2], [3,4,1,4])).norm()
                        for A_list in A_lists))
    log_Zs = np.log(Zs)
    log_Zs = np.array(tuple(log_Z + log_fact_list[0] + log_fact_list[1]
                            for log_Z, log_fact_list
                            in zip(log_Zs, log_fact_lists)))
    Ns = np.array([2*4**i for i in range(n+1)])
    A, B = np.polyfit(Ns[pars["n_discard"]:], log_Zs[pars["n_discard"]:], 1)

    if pars["print_errors"]:
        print("Fit when normalizing Ts: %.3e * N + %.3e"%(A,B))

    A_lists = [[A_list[0]/np.exp(N*A/2 + B/2 - log_fact_list[0]),
                A_list[1]/np.exp(N*A/2 + B/2 - log_fact_list[1])]
               for A_list, N, log_fact_list
               in zip(A_lists, Ns, log_fact_lists)]

    id_pars = get_tensor_id_pars(pars)
    for i, A_list in enumerate(A_lists):
        write_tensor_file(data=A_list, prefix="cdf_tensors_normalized",
                          pars=id_pars, iter_count=i, filename=filename)

    A_list = A_lists[pars["iter_count"]]
    return A_list


def get_eigs(pars, **kwargs):
    pars = pars.copy()
    pars.update(kwargs)
    id_pars= get_scaldim_id_pars(pars)
    try:
        result = read_tensor_file(prefix="cdf_eigs", pars=id_pars,
                                  filename=filename)
    except RuntimeError:
        T = tensordispenser.get_normalized_tensor(pars)
        A_list = get_normalized_tensors(pars)
        A1, A2 = A_list
        result = generate_eigs(T, A1, A2, pars)
        write_tensor_file(data=result, prefix="cdf_eigs", pars=id_pars,
                          filename=filename)
    return result


def generate_eigs(T, A1, A2, pars):
    print("Building the matrix to diagonalize.")
    # Get the eigenvalues and their logarithms.
    # Use iterative diagonalization.
    block_width = pars["block_width"]
    n_dims_do = pars["n_dims_do"]

    dimT = T.shape[3]
    try:
        qimT = T.qhape[3]
    except TypeError:
        qimT = None
    dim1 = A1.shape[3]
    try:
        qim1 = A1.qhape[3]
    except TypeError:
        qim1 = None
    dim2 = A2.shape[3]
    try:
        qim2 = A2.qhape[3]
    except TypeError:
        qim2 = None
    flatdimT = type(T).flatten_dim(dimT)
    flatdim1 = type(T).flatten_dim(dim1)
    flatdim2 = type(T).flatten_dim(dim2)
    matrix_flatdim = flatdimT**(block_width-2)*flatdim1*flatdim2

    def caster(v, charge=0):
        v = type(T).from_ndarray(v, shape=[dim1,dim2] + [dimT]*(block_width-2),
                                 qhape=[qim1,qim2] + [qimT]*(block_width-2),
                                 charge=charge, dirs=[1]*block_width)
        return v

    scon_list_end = [A1, A2] + [T]*(block_width-2)
    index_list = [[i*2 + 1 for i in range(block_width)],
                  [block_width*2, -1, 2, 1], [2, -2, 4, 3]]
    for i in range(3, block_width+1):
        index_list += [[2*i-2, -i, 2*i, 2*i-1]]

    def transfer_op(v, charge=0):
        v = np.reshape(v, (flatdim1, flatdim2) + (flatdimT,)*(block_width-2))
        v = caster(v, charge=charge)
        print(".", end='', flush=True)
        scon_list = [v] + scon_list_end
        Av = scon(scon_list, index_list)
        Av = Av.to_ndarray()
        Av = np.reshape(Av, (matrix_flatdim,))
        return Av

    print("Diagonalizing...", end="")
    if hasattr(T, "sects"):
        qnums = [0,1]
        es = type(T).empty(shape=[[n_dims_do]*len(qnums)],
                           qhape=[qnums], invar=False,
                           dirs=[1], dtype=np.float_)
        for q in qnums:
            transfer_op_lo = spsla.LinearOperator(
                (matrix_flatdim, matrix_flatdim),
                partial(transfer_op, charge=q),
                dtype=T.dtype
            )
            # Use the algorithm for Hermitian operators.
            es_block = spsla.eigsh(transfer_op_lo, k=n_dims_do, maxiter=500,
                                   tol=1e-8, return_eigenvectors=False)
            order = np.argsort(-np.real(np.log(es_block)))
            es_block = es_block[order]
            es[(q,)] = es_block

    else:
        transfer_op_lo = spsla.LinearOperator((matrix_flatdim, matrix_flatdim),
                                              partial(transfer_op, charge=0),
                                              dtype=T.dtype)
        # Use the algorithm for Hermitian operators.
        es = spsla.eigsh(transfer_op_lo, k=n_dims_do, maxiter=500,
                         tol=1e-8, return_eigenvectors=False)
        es = type(T).from_ndarray(es)
    print()

    # Convert es to complex for taking the log.
    es = es.astype(np.complex_, copy=False)
    return es



#=============================================================================#


def main():
    pars = parse()
    id_pars = get_tensor_id_pars(pars)
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
        theta = np.arctan((1-g)/(1+g))

        eigs = get_eigs(pars, g=g)
        leigs = eigs.log()
        leigs *= -pars["block_width"]/(2*np.pi)
        if pars["symmetry_tensors"]:
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


if __name__ == "__main__":
    main()

