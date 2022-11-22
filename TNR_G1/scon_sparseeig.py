import numpy as np
import operator as opr
import itertools as itt
import functools as fct
import scipy.sparse.linalg as spsla
from scon import scon

def scon_sparseeig(tensor_list, index_list, in_inds, out_inds,
                   hermitian=False, print_progress=True, qnums_do=(),
                   return_eigenvectors=True, scon_func=None, **kwargs):
    # We assume all tensors in tensor_list are of the same type type_.
    # T0 represents this type.
    T0 = tensor_list[0]
    type_ = type(T0)
    n_eigs = kwargs.setdefault("k", 6)

    # Figure out the numbers, dims, qims and dirs of all the free
    # indices of the network that we want the eigenvalues of.
    tensor_list = list(tensor_list)
    index_list = list(index_list)
    in_inds = tuple(in_inds)
    out_inds = tuple(out_inds)
    free_inds = []
    free_dims = []
    free_qims = []
    free_dirs = []
    for t, l in zip(tensor_list, index_list):
        for j, k in enumerate(l):
            if k < 0:
                free_inds.append(k)
                free_dims.append(t.shape[j])
                if t.qhape is not None:
                    free_dirs.append(t.dirs[j])
                    free_qims.append(t.qhape[j])
                else:
                    free_dirs.append(None)
                    free_qims.append(None)

    # Divide the free stuff into incoming and outgoing stuff based on
    # in_inds and out_inds.
    free_inds, free_dims, free_qims, free_dirs =\
        zip(*sorted(zip(free_inds, free_dims, free_qims, free_dirs),
                    reverse=True))
    in_dims = list(map(free_dims.__getitem__, in_inds))
    in_qims = list(map(free_qims.__getitem__, in_inds))
    if in_qims[0] is None:
        in_qims = None
    in_dirs = map(free_dirs.__getitem__, in_inds)
    try:
        in_dirs = list(map(opr.neg, in_dirs))
    except TypeError:
        in_dirs = None
    in_flatdims = list(map(type_.flatten_dim, in_dims))
    matrix_flatdim = fct.reduce(opr.mul, in_flatdims)

    # Contraction indices for the vector.
    c_inds = tuple(map(free_inds.__getitem__, in_inds))
    c_inds_set = set(c_inds)
    # Change the signs of the corresponding indices in index_list.
    index_list = [[-i if i in c_inds_set else i
                   for i in l]
                  for l in index_list]
    c_inds = list(map(opr.neg, c_inds))
    index_list.append(c_inds)

    # The permutation on the final legs.
    perm = list(np.argsort(out_inds))

    def scon_op(v, charge=0):
        v = np.reshape(v, in_flatdims)
        v = type_.from_ndarray(v, shape=in_dims, qhape=in_qims,
                               charge=charge, dirs=in_dirs)
        if scon_func is None:
            scon_list = tensor_list + [v]
            Av = scon(scon_list, index_list)
        else:
            Av = scon_func(v)
        Av = Av.to_ndarray()
        Av = np.transpose(Av, perm)
        Av = np.reshape(Av, (matrix_flatdim,))
        if print_progress:
            print(".", end='', flush=True)
        return Av
     
    if print_progress:
        print("Diagonalizing...", end="")

    if hasattr(T0, "sects"):
        # Figure out the list of charges for eigenvectors.
        all_qnums = map(sum, itt.product(*in_qims))
        if T0.qodulus is not None:
            all_qnums = set(q % T0.qodulus for q in all_qnums)
        else:
            all_qnums = set(all_qnums)
        if qnums_do:
            qnums = sorted(all_qnums & set(qnums_do))
        else:
            qnums = sorted(all_qnums)

        # Initialize S and U.
        S_dtype = np.float_ if hermitian else np.complex_
        U_dtype = T0.dtype if hermitian else np.complex_
        S = type_.empty(shape=[[n_eigs]*len(qnums)],
                        qhape=[qnums], invar=False,
                        dirs=[1], dtype=S_dtype)
        if return_eigenvectors:
            U = type_.empty(shape=in_dims+[[n_eigs]*len(qnums)],
                            qhape=in_qims+[qnums],
                            dirs=in_dirs+[-1], dtype=U_dtype)

        for q in qnums:
            scon_op_lo = spsla.LinearOperator((matrix_flatdim, matrix_flatdim),
                                              fct.partial(scon_op, charge=q),
                                              dtype=T0.dtype)
            if hermitian:
                res_block = spsla.eigsh(
                    scon_op_lo, return_eigenvectors=return_eigenvectors,
                    **kwargs
                )
            else:
                res_block = spsla.eigs(
                    scon_op_lo, return_eigenvectors=return_eigenvectors,
                    **kwargs
                )
            if return_eigenvectors:
                S_block, U_block = res_block
            else:
                S_block = res_block

            order = np.argsort(-np.abs(S_block))
            S_block = S_block[order]
            if return_eigenvectors:
                U_block = U_block[:,order]
                U_block = np.reshape(U_block, in_flatdims+[n_eigs])
                U_block = type_.from_ndarray(U_block, shape=in_dims+[[n_eigs]],
                                             qhape=in_qims+[[q]],
                                             dirs=in_dirs+[-1])
                for k, v in U_block.sects.items():
                    U[k] = v
            S[(q,)] = S_block

    else:
        scon_op_lo = spsla.LinearOperator((matrix_flatdim, matrix_flatdim),
                                          scon_op, dtype=T0.dtype)
        if hermitian:
            res = spsla.eigsh(scon_op_lo,
                              return_eigenvectors=return_eigenvectors,
                              **kwargs)
        else:
            res = spsla.eigs(scon_op_lo,
                             return_eigenvectors=return_eigenvectors,
                             **kwargs)
        if return_eigenvectors:
            S, U = res
            U = type_.from_ndarray(U)
            U = U.reshape(in_dims+[n_eigs])
        else:
            S = res
        order = np.argsort(-np.abs(S))
        S = S[order]
        S = type_.from_ndarray(S)
        if return_eigenvectors:
            U = U[...,order]
            U = type_.from_ndarray(U)
    print()

    if return_eigenvectors:
        return S, U
    else:
        return S

