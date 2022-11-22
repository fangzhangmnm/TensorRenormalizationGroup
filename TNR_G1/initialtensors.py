import numpy as np
from scon import scon
from tensors.tensor import Tensor
from tensors.symmetrytensors import TensorZ2, TensorZ3, TensorU1

""" Module for getting the initial tensors for different models. """

def ising_hamiltonian(pars):
    ham = (- pars["J"]*np.array([[ 1,-1],
                                 [-1, 1]],
                                dtype=pars["dtype"])
           + pars["H"]*np.array([[-1, 0],
                                 [ 0, 1]],
                                dtype=pars["dtype"]))
    return ham

def potts3_hamiltonian(pars):
    ham = -pars["J"]*np.eye(3, dtype=pars["dtype"])
    return ham

hamiltonians = {}
hamiltonians["ising"] = ising_hamiltonian
hamiltonians["potts3"] = potts3_hamiltonian

symmetry_classes_dims_qims = {}
symmetry_classes_dims_qims["ising"] = (TensorZ2, [1,1], [0,1])
symmetry_classes_dims_qims["potts3"] = (TensorZ3, [1,1,1], [0,1,2])

# Transformation matrices to the bases where the symmetry is explicit.
symmetry_bases = {}
symmetry_bases["ising"] = np.array([[1, 1],
                                    [1,-1]]) / np.sqrt(2)
phase = np.exp(2j*np.pi/3)
symmetry_bases["potts3"] = np.array([[1,       1,         1],
                                     [1,    phase, phase**2],
                                     [1, phase**2,    phase]],
                                    dtype=np.complex_) / np.sqrt(3)

def get_initial_tensor(pars, **kwargs):
    if kwargs:
        pars = pars.copy()
        pars.update(kwargs)
    model_name = pars["model"].strip().lower()
    ham = hamiltonians[model_name](pars)
    boltz = np.exp(-pars["beta"]*ham)
    T_0 = np.einsum('ab,bc,cd,da->abcd', boltz, boltz, boltz, boltz)
    if pars["symmetry_tensors"]:
        u = symmetry_bases[model_name]
        u_dg = u.T.conjugate()
        T_0 = scon((T_0, u, u, u_dg, u_dg),
                   ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))
        cls, dim, qim = symmetry_classes_dims_qims[model_name]
        T_0 = cls.from_ndarray(T_0, shape=[dim]*4, qhape=[qim]*4,
                               dirs=[1,1,-1,-1])
    else:
        T_0 = Tensor.from_ndarray(T_0)
    return T_0


def get_KW_tensor(pars):
    eye = np.eye(2, dtype=np.complex_)
    ham = hamiltonians["ising"](pars)
    B = np.exp(-pars["beta"] * ham)
    H = np.array([[1,1], [1,-1]], dtype=np.complex_)/np.sqrt(2)
    y_trigged = np.ndarray((2,2,2), dtype=np.complex_)
    y_trigged[:,:,0] = eye
    y_trigged[:,:,1] = sigma_np('y')
    D_sigma = np.sqrt(2) * np.einsum('ab,abi,ic,ad,adk,kc->abcd',
                                     B, y_trigged, H,
                                     B, y_trigged.conjugate(), H)

    u = symmetry_bases["ising"]
    u_dg = u.T.conjugate()
    D_sigma = scon((D_sigma, u, u, u_dg, u_dg),
                   ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))
    if pars["symmetry_tensors"]:
        D_sigma = TensorZ2.from_ndarray(D_sigma, shape=[[1,1]]*4,
                                        qhape=[[0,1]]*4, dirs=[1,1,-1,-1])
    else:
        D_sigma = Tensor.from_ndarray(D_sigma, dirs=[1,1,-1,-1])
    return D_sigma


def get_KW_unitary(pars):
    eye = np.eye(2, dtype=np.complex_)
    CZ = Csigma_np("z")
    U = scon((CZ,
              R_np(np.pi/4, 'z'), R_np(np.pi/4, 'x'),
              R_np(np.pi/4, 'y')),
             ([-1,-2,5,6],
              [-3,5], [3,6],
              [-4,3]))
    u = symmetry_bases["ising"]
    u_dg = u.T.conjugate()
    U = scon((U, u, u_dg, u_dg, u),
             ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))
    U *= -1j
    if pars["symmetry_tensors"]:
        U = TensorZ2.from_ndarray(U, shape=[[1,1]]*4, qhape=[[0,1]]*4,
                                  dirs=[1,1,-1,-1])
    else:
        U = Tensor.from_ndarray(U, dirs=[1,1,1,-1,-1,-1])
    return U


def Csigma_np(sigma_str):
    eye = np.eye(2, dtype=np.complex_)
    CNOT = np.zeros((2,2,2,2), dtype=np.complex_)
    CNOT[:,0,:,0] = eye
    CNOT[:,1,:,1] = sigma_np(sigma_str)
    return CNOT


def dim2_projector_np(i,j):
    xi, xj = [np.array([[1,0]]) if n == 0 else np.array([[0,1]])
              for n in (i, j)]
    P = np.kron(xi.T, xj)
    return P


def sigma_np(c):
    if c=="x":
        res = np.array([[ 0, 1],
                        [ 1, 0]], dtype=np.complex_)
    elif c=="y":
        res = np.array([[ 0j,-1j],
                        [ 1j, 0j]], dtype=np.complex_)
    elif c=="z":
        res = np.array([[ 1, 0],
                        [ 0,-1]], dtype=np.complex_)
    return res


def R_np(alpha, c):
    s = sigma_np(c)
    eye = np.eye(2, dtype=np.complex_)
    res = np.cos(alpha)*eye + 1j*np.sin(alpha)*s
    return res


