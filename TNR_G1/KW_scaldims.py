# Written in python3.

import numpy as np
import sys
import os
import tensordispenser
import initialtensors
import scipy.sparse.linalg as spsla
import operator
import itertools
import scaldim_plot
from tensorstorer import write_tensor_file, read_tensor_file
from functools import partial
from timer import Timer
from TNR import tnr_step_vertstr
from scon import scon
from pathfinder import PathFinder
from custom_parser import parse_argv

np.set_printoptions(precision=7)
np.set_printoptions(linewidth=100)


# - Constants - 
beta_c = np.log(1 + np.sqrt(2)) / 2
T_c = 1/beta_c
filename = os.path.basename(__file__)


def get_tensor_id_pars(pars):
    id_pars = dict()
    mandatory_id_pars = {"symmetry_tensors", "dtype", "iter_count"}
    mandatory_id_pars |= {"chis_tnr", "chis_trg", "opt_eps_conv",
                          "opt_eps_chi", "opt_iters_tens", "opt_max_iter",
                          "A_eps", "A_chis", "return_pieces",
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
    mandatory_id_pars = {"symmetry_tensors", "dtype", "iter_count", "n_discard",
                         "block_width", "do_momenta", "n_dims_do",
                         "do_coarse_momenta"}
    mandatory_id_pars |= {"chis_tnr", "chis_trg", "opt_eps_conv",
                          "opt_eps_chi", "opt_iters_tens", "opt_max_iter",
                          "A_eps", "A_chis", "return_pieces",
                          "reuse_initial", "fix_gauges"}
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
                      ("dtype", "dtype", np.complex_),  # numpy data type to be used throughout,
                      ("J", "float", 1),  # Coupling constant
                      ("H", "float", 0),  # External magnetic field
                      # Flags about which quantities to calculate and plot and how.
                      ("do_coarse_momenta", "bool", False),
                      ("do_momenta", "bool", False),
                      ("n_dims_do", "int", 15),
                      ("n_dims_plot", "int", 15),
                      ("xtick_rotation", "int", 45),
                      ("max_dim_plot", "float", 20000),
                      ("n_discard", "int", 0),
                      ("plot_by_qnum", "bool", True),
                      ("plot_by_momenta", "bool", False),
                      ("draw_exact_lines", "bool", True),
                      ("draw_exact_circles", "bool", True),
                      ("draw_defect_angle", "bool", True),
                      ("show_plots", "bool", True),
                      ("save_plots", "bool", False))
    pars = vars(pars)


    # - Format parameters and set constants -
    pars["horz_refl"] = False
    pars["initial2x2"] = False
    pars["initial4x4"] = False
    pars["fix_gauges"] = False
    pars["KW"] = True
    pars["qnums_plot"] = []
    pars["model"] = "Ising"
    pars["algorithm"] = "TNR"
    pars["beta"] = beta_c
    pars["J"] = pars["dtype"](pars["J"])
    pars["H"] = pars["dtype"](pars["H"])
    pars["return_gauges"] = True
    if pars["plot_by_momenta"] and not pars["do_momenta"]:
        raise ValueError("plot_by_momenta but not do_momenta")
    if pars["do_momenta"] and pars["block_width"] < 2:
        raise ValueError("do_momenta but block_width < 2")
    return pars


def ascend_U(U, u, u1, z, z1, z2, pars):
    # This is O(chi^10).
    if pars["print_errors"]:
        print("Ascending U.")
    U1 = U.flip_dir(1)
    U2 = U.flip_dir(2)
    U2 = U2.flip_dir(3)
    U2 = U2.flip_dir(5)
    u_dg = u.conjugate().transpose((2,3,0,1))
    u1_dg = u1.conjugate().transpose((2,3,0,1))
    z_dg = z.conjugate().transpose((1,2,0))
    z1_dg = z1.conjugate().transpose((1,2,0))
    z2_dg = z2.conjugate().transpose((1,2,0))
    asc_U = scon((z1, z2, z,
                  u1, u,
                  U1,
                  U2,
                  u_dg, u1_dg,
                  z_dg, z1_dg, z2_dg),
                 ([-1,15,16], [-2,5,6], [-3,19,18],
                  [16,5,1,2], [6,19,7,14],
                  [1,2,7,11,12,13],
                  [12,13,14,9,3,4],
                  [11,9,17,10], [3,4,8,20],
                  [15,17,-4], [10,8,-5], [20,18,-6]))
    return asc_U


def generate_initial_tensors(pars):
    T = initialtensors.get_initial_tensor(pars)
    D_sigma = initialtensors.get_KW_tensor(pars)
    U = initialtensors.get_KW_unitary(pars)
    eye = np.eye(2, dtype=np.complex_)
    eye = type(U).from_ndarray(eye, shape=[[1,1]]*2, qhape=[[0,1]]*2,
                               dirs=[1,-1])
    U = scon((U, eye), ([-1,-2,-4,-5], [-3,-6]))
    A_list = [D_sigma, T]
    log_fact_list = [0,0]
    return A_list, log_fact_list, U


def ascend_tensors(A_list, log_fact_list, U, pars):
    print('\n / Coarse-graining D_sigma, iter_count = #%i: / '
          %(pars["iter_count"]))
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
    U = ascend_U(U, pieces['u'], pieces_vertstr['u'], pieces['z'],
                 pieces_vertstr['z1'], pieces_vertstr['z2'], pars)
    return A_list, log_fact_list, U


def get_tensors(pars, **kwargs):
    pars = pars.copy()
    pars.update(kwargs)
    id_pars= get_tensor_id_pars(pars)
    try:
        result = read_tensor_file(prefix="KW_tensors", pars=id_pars,
                                  filename=filename)
    except RuntimeError:
        result = generate_tensors(pars)
    return result


def generate_tensors(pars, **kwargs):
    pars = pars.copy()
    pars.update(kwargs)
    if pars["iter_count"] == 0:
        A_list, log_fact_list, U = generate_initial_tensors(pars)
    else:
        # Get the tensor from the previous step
        A_list, log_fact_list, U = get_tensors(pars,
                                               iter_count=pars["iter_count"]-1)
        # and ascend it.
        A_list, log_fact_list, U = ascend_tensors(A_list, log_fact_list, U, pars)
        # Save to file.
        id_pars = get_tensor_id_pars(pars)
        write_tensor_file((A_list, log_fact_list, U), prefix="KW_tensors",
                          pars=id_pars, filename=filename)
    return A_list, log_fact_list, U


def get_normalized_tensors(pars, **kwargs):
    pars = pars.copy()
    pars.update(kwargs)
    id_pars= get_tensor_id_pars(pars)
    try:
        result = read_tensor_file(prefix="KW_tensors_normalized", pars=id_pars,
                                  filename=filename)
    except RuntimeError:
        result = generate_normalized_tensors(pars)
    return result


def generate_normalized_tensors(pars):
    # Number of tensors to use to fix the normalization
    n = max(8, pars["iter_count"] + 4)
    # Number of tensors from the beginning to discard
    n_discard = max(min(pars["iter_count"]-3, 3), 0)
    tensors_and_log_facts = []
    for i in reversed(list(range(n+1))):
        tensors_and_log_facts.append(get_tensors(pars=pars, iter_count=i))
    tensors_and_log_facts = tuple(reversed(tensors_and_log_facts))
    A_lists = tuple(map(lambda t: t[0], tensors_and_log_facts))
    log_fact_lists = np.array(tuple(map(lambda t: t[1], tensors_and_log_facts)))
    Us = np.array(tuple(map(lambda t: t[2], tensors_and_log_facts)))

    Zs = np.array(tuple(scon(A_list, ([1,2,3,2], [3,4,1,4])).norm() for A_list in A_lists))
    log_Zs = np.log(Zs)
    log_Zs = np.array(tuple(log_Z + log_fact_list[0] + log_fact_list[1]
                            for log_Z, log_fact_list in zip(log_Zs, log_fact_lists)))
    Ns = np.array([4*4**i-2**i for i in range(n+1)])
    A, B = np.polyfit(Ns[pars["n_discard"]:], log_Zs[pars["n_discard"]:], 1)

    if pars["print_errors"]:
        print("Fit when normalizing Ts: %.3e * N + %.3e"%(A,B))

    A_lists = [[A_list[0]/np.exp(N*A/2 - log_fact_list[0]),
                A_list[1]/np.exp(N*A/2 - log_fact_list[1])]
               for A_list, N, log_fact_list in zip(A_lists, Ns, log_fact_lists)]

    id_pars = get_tensor_id_pars(pars)
    for i, (A_list, U) in enumerate(zip(A_lists, Us)):
        write_tensor_file(data=(A_list, U), prefix="KW_tensors_normalized",
                          pars=id_pars, iter_count=i, filename=filename)

    A_list = A_lists[pars["iter_count"]]
    U = Us[pars["iter_count"]]
    return A_list, U


def combine_coarse_momenta(res_fine, res_coarse):
    scaldims, c = res_fine[0:2]
    momenta = res_coarse[2]
    return scaldims, c, momenta


def get_scaldims(pars, **kwargs):
    pars = pars.copy()
    pars.update(kwargs)
    id_pars= get_scaldim_id_pars(pars)
    try:
        result = read_tensor_file(prefix="KW_scaldims", pars=id_pars,
                                  filename=filename)
    except RuntimeError:
        T = tensordispenser.get_normalized_tensor(pars)
        A_list, U = get_normalized_tensors(pars)
        A1, A2 = A_list
        if pars["do_coarse_momenta"]:
            temp_pars = pars.copy()
            temp_pars["do_coarse_momenta"] = False
            temp_pars["do_momenta"] = False
            res_fine = get_scaldims(temp_pars)
            res_coarse = generate_scaldims(T, A1, A2, U, pars)
            result = combine_coarse_momenta(res_fine, res_coarse)
        else:
            result = generate_scaldims(T, A1, A2, U, pars)
        write_tensor_file(data=result, prefix="KW_scaldims", pars=id_pars,
                          filename=filename)
    return result


def build_T_T_first(T, A1, A2, U, pars):
    # TODO Should this bond dimensions be an independent parameter?
    chis = [chi for chi in pars["chis_trg"]]
    T_orig = T
    if pars["print_errors"]:
        print("Building the coarse-grained transfer matrix times "
              "translation.")
    T_dg = T.conjugate().transpose((0,3,2,1)) 
    y_env = scon((T, T, T_dg, T_dg),
                 ([1,-1,5,2], [5,-2,3,4], [1,2,6,-3], [6,4,3,-4]))
    u = y_env.eig((0,1), (2,3), hermitian=True, chis=chis, print_errors=pars["print_errors"])[1]
    y = u.conjugate().transpose((2,0,1))
    y_dg = y.conjugate().transpose((1,2,0))
    SW, NE = T.split((0,3), (1,2), chis=chis, print_errors=pars["print_errors"])
    SW = SW.transpose((0,2,1))
    NE = NE.transpose((1,2,0))
    T = scon((NE, y, T, SW, y_dg),
             ([1,3,-1], [-2,1,4], [3,4,6,5], [6,-3,2], [5,2,-4]))
    # TODO quantify the error in this coarse-graining and print it.

    if pars["print_errors"] > 1:
        print("Optimizing y1 & y2.")
    cost = np.inf
    cost_change = np.inf
    counter = 0
    
    # Initial y1 & y2
    init_env1 = scon((T_orig, A1), ([-1,-2,1,-5], [1,-3,-4,-6]))
    y1_dg = init_env1.svd((1,2), (0,4,5,3), chis=chis)[0]
    init_env2 = scon((A2, T_orig), ([-1,-2,1,-5], [1,-3,-4,-6]))
    y2_dg = init_env2.svd((1,2), (0,4,5,3), chis=chis)[0]
    y1 = y1_dg.conjugate().transpose((2,0,1))
    y2 = y2_dg.conjugate().transpose((2,0,1))

    while cost_change > 1e-11 and counter < 10000:
        # The optimization step
        # This is O(chi^8).
        env1_half = scon((y2,
                          NE, A1, A2, T_orig, SW,
                          U,
                          y1_dg, y2_dg),
                         ([-3,4,3],
                          [-1,14,-7], [14,-2,13,12], [13,4,6,8], [6,3,10,9], [10,-4,7],
                          [12,8,9,1,2,11],
                          [1,2,-5], [11,7,-6]))
        env1 = scon((env1_half, env1_half.conjugate()), ([-1,-2,3,4,5,6,7], [-11,-12,3,4,5,6,7]))
        u = env1.eig((0,1), (2,3), hermitian=True, chis=chis)[1]
        y1_dg = u
        y1 = y1_dg.conjugate().transpose((2,0,1))

        env2_half = scon((y1, y2,
                          NE, A1, A2, T_orig, SW,
                          U,
                          y1_dg),
                         ([-1,1,3], [-2,6,8],
                          [1,2,-7], [2,3,12,11], [12,6,7,9], [7,8,13,10], [13,-3,-6],
                          [11,9,10,4,5,-5],
                          [4,5,-4]))
        env2 = scon((env2_half, env2_half.conjugate()), ([1,2,3,4,-5,-6,7], [1,2,3,4,-15,-16,7]))
        u = env2.eig((0,1), (2,3), hermitian=True, chis=chis)[1]
        y2 = u.transpose((2,0,1)).flip_dir(0)
        y2_dg = y2.conjugate().transpose((1,2,0))

        old_cost = cost
        cost = scon((env2, y2, y2_dg), ([1,2,3,4], [5,3,4], [1,2,5])).value()
        if np.imag(cost) > 1e-13:
            warnings.warn("optimize y1 & y2 cost is complex: " + str(cost))
        else:
            cost = np.real(cost)
        cost_change = np.abs((old_cost - cost)/cost)
        counter += 1
    T_first = scon((env2_half, y2_dg),
                   ([-2,-3,-4,-5,1,2,-1], [1,2,-6]))
    #TODO This is too computationally expensive, O(chi^10)
    #     Even the smarter way of |C| - |cross term| is O(chi^9) at
    #     least.
    #if pars["print_errors"] > 1:
    #    orig_T_first = scon((NE, A1, A2, T_orig, SW,
    #                         U),
    #                        ([-2,6,-1], [6,-3,3,2], [3,-4,4,1], [4,-5,7,5], [7,-6,-10],
    #                         [2,1,5,-7,-8,-9]))
    #    coarsed_T_first = scon((y1_dg, y2_dg,
    #                            T_first,
    #                            y1, y2),
    #                           ([-2,-3,1], [-4,-5,2],
    #                            [-1,1,2,-6,3,4],
    #                            [3,-7,-8], [4,-9,-10]))
    #    err = (orig_T_first - coarsed_T_first).norm() / orig_T_first.norm()
    #    print("After %i iterations, error in optimize y1 & y2 is %.3e."
    #          %(counter, err))
    return T, T_first


def generate_scaldims(T, A1, A2, U, pars):
    print("Building the matrix to diagonalize.")
    # Get the eigenvalues and their logarithms.
    # Use iterative diagonalization.
    block_width = pars["block_width"]
    n_dims_do = pars["n_dims_do"]
    do_coarse_momenta = pars["do_coarse_momenta"]

    if do_coarse_momenta:
        T, T_first = build_T_T_first(T, A1, A2, U, pars)

    dimT = T.shape[3]
    try:
        qimT = T.qhape[3]
    except TypeError:
        qimT = None
    if do_coarse_momenta:
        dim1 = T_first.shape[4]
        try:
            qim1 = T_first.qhape[4]
        except TypeError:
            qim1 = None
        dim2 = T_first.shape[5]
        try:
            qim2 = T_first.qhape[5]
        except TypeError:
            qim2 = None
    else:
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
        v = type(T).from_ndarray(v, shape=[dim1, dim2] + [dimT]*(block_width-2),
                                 qhape=[qim1, qim2] + [qimT]*(block_width-2),
                                 charge=charge, dirs=[1]*block_width)
        return v

    if do_coarse_momenta:
        scon_list_end = [T_first] + [T]*(block_width-2) 
        index_list = [[i*2 + 1 for i in range(block_width)],
                      [block_width*2, -1, -2, 4, 1, 3]]
    else:
        scon_list_end = [A1, A2] + [T]*(block_width-2)
        index_list = [[i*2 + 1 for i in range(block_width)],
                      [block_width*2, -1, 2, 1], [2, -2, 4, 3]]
    for i in range(3, block_width+1):
        index_list += [[2*i-2, -i, 2*i, 2*i-1]]

    if pars["do_momenta"] and not pars["do_coarse_momenta"]:
        translation = [block_width-1] + list(range(block_width-1))

    def transfer_op(v, charge=0):
        v = np.reshape(v, (flatdim1, flatdim2) + (flatdimT,)*(block_width-2))
        v = caster(v, charge=charge)
        if pars["do_momenta"] and not pars["do_coarse_momenta"]:
            v = np.transpose(v, translation)
            v = scon((U, v),
                     ([-1,-2,-3,1,2,3],
                      [1,2,3] + [-i for i in range(4, block_width+1)]))
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
            transfer_op_lo = spsla.LinearOperator((matrix_flatdim, matrix_flatdim),
                                                  partial(transfer_op, charge=q),
                                                  dtype=T.dtype)
            if pars["do_momenta"]:
                es_block = spsla.eigs(transfer_op_lo, k=n_dims_do, maxiter=500,
                                      tol=1e-8, return_eigenvectors=False)
            else:
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
        if pars["do_momenta"]:
            es = spsla.eigs(transfer_op_lo, k=n_dims_do, maxiter=500,
                            tol=1e-8, return_eigenvectors=False)
        else:
            # Use the algorithm for Hermitian operators.
            es = spsla.eigsh(transfer_op_lo, k=n_dims_do, maxiter=500,
                             tol=1e-8, return_eigenvectors=False)
        es = type(T).from_ndarray(es)
    print()

    log_es = es.log()
    log_es /= 2*np.pi
    if do_coarse_momenta:
        log_es *= pars["block_width"]*2 - 1/(2**(pars["iter_count"]+1))
    else:
        log_es *= pars["block_width"] - 1/(2**(pars["iter_count"]+1))

    c = 1/2
    log_es -= c/12
    log_es *= -1
    scaldims = log_es.real()
    if pars["do_momenta"]:
        momenta = log_es.imag()
    else:
        momenta = None
    return scaldims, c, momenta


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

    scaldims, c, momenta = get_scaldims(pars)
    scaldim_plot.plot_and_print(scaldims, c, pars, pather, momenta=momenta,
                                id_pars=id_pars)

if __name__ == "__main__":
    main()

