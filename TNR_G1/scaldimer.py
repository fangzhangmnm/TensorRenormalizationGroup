#!/usr/bin/python3

import numpy as np
import os
import sys
import tensordispenser
import modeldata
import operator
import itertools
import warnings
import scaldim_plot
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
    mandatory_id_pars = {"model", "algorithm", "symmetry_tensors", "dtype",
                         "iter_count", "initial2x2", "initial4x4", "n_discard",
                         "block_width", "do_dual", "defect_angles",
                         "do_momenta", "do_coarse_momenta", "do_eigenvectors",
                         "n_dims_do"}
    algoname = pars["algorithm"].lower().strip()
    if algoname == "tnr":
        mandatory_id_pars |= {"chis_tnr", "chis_trg", "opt_eps_conv",
                              "opt_eps_chi", "opt_iters_tens", "opt_max_iter",
                              "A_eps", "A_chis", "return_pieces",
                              "reuse_initial", "fix_gauges", "horz_refl"}
    modelname = pars["model"].lower().strip()
    if modelname == "ising":
        mandatory_id_pars |= {"J", "H", "beta"}
    elif modelname == "potts3":
        mandatory_id_pars |= {"J", "beta"}
    elif modelname == "sixvertex":
        mandatory_id_pars |= {"sixvertex_a", "sixvertex_b", "sixvertex_c"}
    if pars["symmetry_tensors"]:
        mandatory_id_pars |= {"qnums_do"}

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
                      ("algorithm", "str", ""),
                      # Parameters for TNR
                      ("chis_tnr", "int_list", [4]),
                      ("chis_trg", "int_list", [1,2,3,4,5,6]),
                      ("opt_eps_conv", "float", 1e-11),
                      ("opt_eps_chi", "float", 1e-8),
                      ("opt_iters_tens", "int", 1),
                      ("opt_max_iter", "int", 10000),
                      ("A_eps", "float", 1e-11),
                      ("A_chis", "int_list", None),
                      ("print_errors", "int", 2),
                      ("return_pieces", "bool", True),
                      ("reuse_initial", "bool", True),
                      ("fix_gauges", "bool", True),
                      ("symmetry_tensors", "bool", False),
                      ("horz_refl", "bool", True),
                      ("return_gauges", "bool", True),
                      # Parameters for both TRG and TNR
                      ("iter_count", "int", 6),  # Number of iterations
                      ("dtype", "dtype", np.complex_),  # numpy data type
                      ("J", "float", 1),  # Coupling constant
                      ("H", "float", 0),  # Coupling constant
                      ("initial2x2", "bool", False),
                      ("initial4x4", "bool", False),
                      # Spectrum parameters.
                      ("n_dims_do", "int", 15),
                      ("qnums_do", "int_list", []),  #[] denotes all
                      ("n_discard", "int", 0),
                      ("block_width", "int", 0),
                      ("do_coarse_momenta", "bool", False),
                      ("do_dual", "bool", False),
                      ("defect_angles", "float_list", [0]),
                      ("do_momenta", "bool", False),
                      ("do_eigenvectors", "bool", False),
                      # Sixvertex parameters
                      ("sixvertex_a", "float", 1),
                      ("sixvertex_b", "float", 1),
                      ("sixvertex_c", "float", np.sqrt(2)),
                      # IO parameters.
                      ("n_dims_plot", "int", 15),
                      ("max_dim_plot", "float", 20000),
                      ("qnums_plot", "int_list", []),  #[] denotes all
                      ("xtick_rotation", "int_list", 0),
                      ("show_plots", "bool", True),
                      ("save_plots", "bool", False),
                      ("plot_by_qnum", "bool", True),
                      ("plot_by_momenta", "bool", False),
                      ("plot_by_alpha", "bool", False),
                      ("draw_exact_lines", "bool", True),
                      ("draw_exact_circles", "bool", True),
                      ("draw_defect_angle", "bool", True),
                      ("save_fit_plot", "bool", False),
                      ("save_scaldim_file", "bool", True))
    pars = vars(pars)

    # - Format parameters -
    pars["beta"] = modeldata.get_critical_beta(pars)
    pars["J"] = pars["dtype"](pars["J"])
    pars["H"] = pars["dtype"](pars["H"])
    pars["defect_angles"] = sorted(pars["defect_angles"], key=abs)
    if pars["defect_angles"] != [0] and pars["do_dual"]:
        raise ValueError("Non-trivial defect_angles but do_dual.")
    if pars["defect_angles"] != [0] and not pars["symmetry_tensors"]:
        raise ValueError("Non-trivial defect_angles and not symmetry_tensors.")
    if pars["plot_by_momenta"] and not pars["do_momenta"]:
        raise ValueError("plot_by_momenta but not do_momenta")
    if pars["do_momenta"] and pars["block_width"] < 2:
        raise ValueError("do_momenta but block_width < 2")
    if pars["n_dims_plot"] > pars["n_dims_do"]:
        raise ValueError("n_dims_plot > n_dims_do")
    if not set(pars["qnums_plot"]).issubset(set(pars["qnums_do"])):
        raise ValueError("qnums_plot is not included in qnums_do")

    return pars


def get_defect(alpha, T, index):
    # Build the defect.
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
    T = tensordispenser.get_normalized_tensor(pars)
    parts = ()
    if pars["do_coarse_momenta"]:
        # TODO Should this bond dimension be an independent parameter?
        chis = [chi for chi in pars["chis_trg"]]
        T_orig = T
        if pars["print_errors"]:
            print("Building the coarse-grained transfer matrix times "
                  "translation.")
        T_dg = T.conjugate().transpose((0,3,2,1)) 
        y_env = scon((T, T, T_dg, T_dg),
                     ([1,-1,5,2], [5,-2,3,4], [1,2,6,-3], [6,4,3,-4]))
        U = y_env.eig((0,1), (2,3), hermitian=True, chis=chis)[1]
        y = U.conjugate().transpose((2,0,1))
        y_dg = y.conjugate().transpose((1,2,0))
        SW, NE = T.split((0,3), (1,2), chis=chis)
        SW = SW.transpose((0,2,1))
        NE = NE.transpose((1,2,0))
        T = scon((NE, y, T, SW, y_dg),
                 ([1,3,-1], [-2,1,4], [3,4,6,5], [6,-3,2], [5,2,-4]))
        parts = (y, y_dg, NE, SW, T_orig)
        # TODO quantify the error in this coarse-graining and print it.
    return T, parts


def get_T_last(T, pars, alpha=0, parts=None):
    if pars["do_coarse_momenta"]:
        if pars["print_errors"] > 1:
            print("Optimizing y_last, alpha =", alpha)
        y, y_dg, NE, SW, T_orig = parts
        defect_horz = get_defect(alpha, T_orig, 0)
        defect_vert = get_defect(alpha, T_orig, 1).conjugate().transpose()
        cost = np.inf
        cost_change = np.inf
        counter = 0
        y_last = y
        y_last_dg = y_dg
        # TODO Should this bond dimension be an independent parameter?
        chis = [chi for chi in pars["chis_trg"]]
        while cost_change > 1e-11 and counter < 10000:
            # The optimization step
            # This is O(chi^6). Could use a pre-environment too.
            env_part1 = scon((NE, defect_horz, T_orig, SW, defect_vert,
                              y_last_dg),
                             ([-2,1,-1], [1,6], [6,-3,5,2], [5,-4,3], [2,4],
                              [4,3,-5]))
            env = scon((env_part1, env_part1.conjugate()),
                       ([1,-1,-2,2,3], [1,-3,-4,2,3]))
            U = env.eig((0,1), (2,3), hermitian=True, chis=chis)[1]
            y_last_dg = U
            y_last = y_last_dg.conjugate().transpose((2,0,1))
            old_cost = cost
            cost = scon((env, y_last, y_last_dg),
                        ([1,2,3,4], [5,1,2], [3,4,5])).value()
            if np.imag(cost) > 1e-13:
                warnings.warn("optimize y_last cost is complex: " + str(cost))
            else:
                cost = np.real(cost)
            cost_change = np.abs((old_cost - cost)/cost)
            counter += 1
        T_last = scon((y_last,
                       NE, defect_horz, T_orig, SW, defect_vert,
                       y_last_dg),
                      ([-2,7,8],
                       [7,1,-1], [1,6], [6,8,5,2], [5,-3,3], [2,4],
                       [4,3,-4]))
        if pars["print_errors"] > 1:
            orig_T_last = scon((NE, defect_horz,
                                T_orig, SW, defect_vert),
                               ([-2,1,-1], [1,6],
                                [6,-3,5,2], [5,-4,-6], [2,-5]))
            coarsed_T_last = scon((y_last_dg, T_last, y_last),
                                  ([-2,-3,1], [-1,1,-4,2], [2,-5,-6]))
            err = (orig_T_last - coarsed_T_last).norm() / orig_T_last.norm()
            print("After %i iterations, error in optimize y_last is %.3e."
                  %(counter, err))
    elif pars["do_momenta"]:
        defect_horz = get_defect(alpha, T, 0)
        defect_vert = get_defect(alpha, T, 1).conjugate().transpose()
        T_last = scon((T, defect_horz, defect_vert),
                      ([1,-2,-3,4], [-1,1], [4,-4]))
    else:
        defect_horz = get_defect(alpha, T, 0)
        T_last = scon((T, defect_horz), ([1,-2,-3,-4], [-1,1]))
    return T_last


def get_cft_data(pars):
    T, parts = get_T(pars)

    scaldims_by_alpha = {}
    if pars["do_momenta"]:
        momenta_by_alpha = {}
    if pars["do_eigenvectors"]:
        evects_by_alpha = {}
    for alpha in pars["defect_angles"]:
        print("Building the matrix to diagonalize.")
        T_last = get_T_last(T, pars, alpha=alpha, parts=parts)

        # Get the eigenvalues and their logarithms.
        if pars["block_width"] > 1:
            # Use iterative diagonalization.
            block_width = pars["block_width"]
            n_dims_do = pars["n_dims_do"]

            if pars["do_momenta"] and not pars["do_coarse_momenta"]:
                translation = list(range(1, block_width)) + [0]
            else:
                translation = range(block_width)
            scon_list = [T]*(block_width-1) + [T_last]
            index_list = [[block_width*2, -101, 2, -1]]
            for i in range(2, block_width+1):
                index_list += [[2*i-2, -100-i, 2*i, -(2*i-1)]]

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
        else:
            # Use full diagonalization.
            if pars["do_dual"]:
                M = T_last.join_indices((0,1), (2,3), dirs=[1,-1])
            else:
                M = scon(T_last, [1,-1,1,-2])
            print("Diagonalizing.")
            es, evects = M.eig(0,1)

        # Convert es to complex for taking the log.
        es = es.astype(np.complex_, copy=False)

        # Log and scale the eigenvalues.
        if pars["block_width"]:
            log_es = es.log() * pars["block_width"] / (2*np.pi)
            if pars["do_coarse_momenta"]:
                log_es *= 2
        elif pars["do_dual"]:
            log_es = es.log() / np.pi
        else:
            log_es = es.log() / (2*np.pi)

        # Extract the central charge.
        if alpha == 0:
            c = log_es.real().max() * 12
        try:
            log_es -= c/12
        except NameError:
            raise ValueError("Need to provide 0 in defect_angles to be able "
                             "to obtain the central charge.")
        log_es *= -1
        scaldims = log_es.real()
        scaldims_by_alpha[alpha] = scaldims
        if pars["do_momenta"]:
            momenta = log_es.imag()
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
        # TODO Remove the following once all important files have been
        # renamed.
        #old_id_pars = id_pars.copy()
        #del(old_id_pars["do_eigenvectors"])
        #try:
        #    res = read_tensor_file(prefix="scals_by_alpha", pars=old_id_pars,
        #                           filename=filename)
        #    write_tensor_file(data=res, prefix="scals_by_alpha", pars=id_pars,
        #                      filename=filename)
        #    print("Renamed old style scaldim file.")
        #except RuntimeError:
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


def combine_momenta_nd(scaldims_a, scaldims_b, momenta_a, momenta_b, pars):
    # The goal is to take pairs of (scaldim, momentum) from a and b,
    # match them and take the scaldim from a and momentum from b.
    scaldims_b = list(scaldims_b)
    momenta_b = list(momenta_b)
    scaldims = np.array(scaldims_a)
    momenta = []
    for sa, ma in zip(scaldims_a, momenta_a):
        scaldims_diff = np.abs(scaldims_b - sa)
        momenta_diff = np.abs(momenta_b - ma) % pars["block_width"]
        momenta_diff = [min(md, np.abs(md - pars["block_width"]))
                        for md in momenta_diff]
        sum_diff = scaldims_diff + momenta_diff
        idx = np.argmin(sum_diff)
        if sum_diff[idx] > 0.1:
            warnings.warn("Combining scaling dimensions %.3e and %.3e with "
                          "momenta %.3e and %.3e, even though the sum_diff is "
                          "%.3e."%(sa, scaldims_b[idx],
                                   ma, momenta_b[idx],
                                   sum_diff[idx]))
        momenta.append(momenta_b[idx])
        del(scaldims_b[idx])
        del(momenta_b[idx])
    momenta = np.array(momenta)
    return scaldims, momenta


def combine_coarse_momenta(res_fine_by_alpha, res_coarse_by_alpha, pars):
    c = res_fine_by_alpha[1]
    scaldims_by_alpha = {}
    momenta_by_alpha = {}
    for alpha in res_fine_by_alpha[0].keys():
        scaldims_a = res_fine_by_alpha[0][alpha]
        scaldims_b = res_coarse_by_alpha[0][alpha]
        momenta_a = res_fine_by_alpha[2][alpha]
        momenta_b = res_coarse_by_alpha[2][alpha]
        if pars["symmetry_tensors"]:
            scaldims = scaldims_a.empty_like()
            momenta = momenta_a.empty_like()
            for q in scaldims_a.sects.keys():
                scaldims_q_a = scaldims_a[q]
                scaldims_q_b = scaldims_b[q]
                momenta_q_a = momenta_a[q]
                momenta_q_b = momenta_b[q]
                scaldims_q, momenta_q =\
                        combine_momenta_nd(scaldims_q_a, scaldims_q_b,
                                           momenta_q_a, momenta_q_b,
                                           pars)
                scaldims[q] = scaldims_q
                momenta[q] = momenta_q
        else:
            scaldims, momenta = combine_momenta_nd(scaldims_a, scaldims_b,
                                                   momenta_a, momenta_b,
                                                   pars)
        scaldims_by_alpha[alpha] = scaldims
        momenta_by_alpha[alpha] = momenta
    return scaldims_by_alpha, c, momenta_by_alpha


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

    if pars["do_coarse_momenta"]:
        temp_pars = pars.copy()
        temp_pars["do_coarse_momenta"] = False
        res_fine = load_cft_data(temp_pars)
        temp_pars["do_coarse_momenta"] = True
        res_coarse = load_cft_data(temp_pars)
        scaldims_by_alpha, c, momenta_by_alpha =\
                combine_coarse_momenta(res_fine, res_coarse, pars)
    else:
        res = load_cft_data(pars)
        scaldims_by_alpha, c = res[:2]
        if pars["do_momenta"]:
            momenta_by_alpha = res[2]
        else:
            momenta_by_alpha = None

    scaldim_plot.plot_and_print_dict(scaldims_by_alpha, c, pars, pather,
                                     momenta_dict=momenta_by_alpha,
                                     id_pars=id_pars)

