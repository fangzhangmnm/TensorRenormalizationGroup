import numpy as np
import collections
from tensors import tensor, symmetrytensors, tensorcommon
from functools import reduce
from scon import scon


def contract2x2(T_list, vert_flip=False):
    """ Takes an iterable of rank 4 tensors and contracts a square made
    of them to a single rank 4 tensor. If only a single tensor is given
    4 copies of the same tensor are used. If vert_flip=True the lower
    two are vertically flipped and complex conjugated.
    """
    if isinstance(T_list, (np.ndarray, tensorcommon.TensorCommon)):
        T = T_list
        T_list = [T]*4
    else:
        T_list = list(T_list)
    if type(T_list[0]) is np.ndarray:
        return contract2x2_ndarray(T_list, vert_flip=vert_flip)
    else:
        return contract2x2_Tensor(T_list, vert_flip=vert_flip)

def contract2x2_Tensor(T_list, vert_flip=False):
    if vert_flip:
        def flip(T):
            T.transpose((0,3,2,1))
        flip(T_list[2])
        flip(T_list[3])
    T4 = scon((T_list[0], T_list[1], T_list[2], T_list[3]),
              ([-2,-3,1,3], [1,-4,-6,4], [-1,3,2,-7], [2,4,-5,-8]))
    T4 = T4.join_indices((0,1), (2,3), (4,5), (6,7), dirs=[1,1,-1,-1])
    return T4

def contract2x2_ndarray(T_list, vert_flip=False):
    if vert_flip:
        def flip(T):
            return np.transpose(T.conjugate(), (0,3,2,1))
        T_list[2] = flip(T_list[2])
        T_list[3] = flip(T_list[3])
    T4 = scon((T_list[0], T_list[1], T_list[2], T_list[3]),
              ([-2,-3,1,3], [1,-4,-6,4], [-1,3,2,-7], [2,4,-5,-8]))
    sh = T4.shape
    S = np.reshape(T4, (sh[0]*sh[1], sh[2]*sh[3], sh[4]*sh[5], sh[6]*sh[7]))
    return S

