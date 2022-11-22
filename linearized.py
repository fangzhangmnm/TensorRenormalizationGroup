import numpy as np
from tqdm.auto import tqdm
import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator,aslinearoperator
from functorch import jvp,vjp
from math import prod
import torch
from opt_einsum import contract


def mysvd(M,k=10,tol=0,maxiter=500):
    M=aslinearoperator(M)
    dim=M.shape[1]
    k=min(k,min(M.shape))
    eigvecs,eigvals=[],[]
    tols=tol if isinstance(tol,list) else [tol]*k
    maxiters=maxiter if isinstance(maxiter,list) else [maxiter]*k
    pbar1=tqdm(range(k),leave=False)
    for j in pbar1:
        v=np.random.randn(dim);v=v/np.linalg.norm(v)
        with tqdm(range(maxiters[j]),leave=False) as pbar:
            for i in pbar:
                vn=M.rmatvec(M.matvec(v))
                for u in eigvecs:
                    vn=vn-u*(u.conj()@vn)
                if np.linalg.norm(vn)==0:
                    raise ValueError
                eig=np.linalg.norm(vn)/np.linalg.norm(v)
                vn=vn/np.linalg.norm(vn)
                err=np.linalg.norm(vn-v)
                v=vn
                pbar1.set_postfix(eig=eig,err=err)
                if err<=tols[j]:
                    pbar.close()
                    break
        
        if err>tols[j]:
            print('Not Converged! err=',err)
        print('eig=',eig)
        eigvecs.append(v)
        eigvals.append(eig)
    u=np.array([v/np.linalg.norm(v) for v in [M*v for v in eigvecs]]).T
    s=np.array(np.abs(eigvals))**.5
    vh=np.array(eigvecs).conj()
    return u,s,vh


def myeigh(M,k=10,tol=0,maxiter=500,impose_hermitian=True):
    M=aslinearoperator(M)
    dim=M.shape[1]
    k=min(k,min(M.shape))
    eigvecs,eigvals=[],[]
    tols=tol if isinstance(tol,list) else [tol]*k
    maxiters=maxiter if isinstance(maxiter,list) else [maxiter]*k
    pbar1=tqdm(range(k),leave=False)
    for j in pbar1:
        v=np.random.randn(dim);v=v/np.linalg.norm(v)
        with tqdm(range(maxiters[j]),leave=False) as pbar:
            for i in pbar:
                if impose_hermitian:
                    vn=(M.rmatvec(v)+M.matvec(v))/2
                else:
                    vn=M.matvec(v)
                for u in eigvecs:
                    vn=vn-u*(u.conj()@vn)
                if np.linalg.norm(vn)==0:
                    raise ValueError
                eig=np.linalg.norm(vn)/np.linalg.norm(v)
                vn=vn/np.linalg.norm(vn)
                err=np.linalg.norm(vn-v)
                v=vn
                pbar1.set_postfix(eig=eig,err=err)
                if err<=tols[j]:
                    pbar.close()
                    break
        
        if err>tols[j]:
            print('Not Converged! err=',err)
        print('eig=',eig)
        eigvecs.append(v)
        eigvals.append(eig)
    eigvecs=np.array(eigvecs)
    eigvals=np.array(eigvals)
    return eigvals,eigvecs.T


#M=np.random.randn(5,4)
#u,s,vh=mysvd(scipy.sparse.linalg.aslinearoperator(M))
#print(np.linalg.norm(u@np.diag(s)@vh-M))


#======== HOTRG ============

from HOTRGZ2 import forward_layer, HOTRG_layer


def get_linearized_cylinder(T0):
    dimT=prod(T0.shape)
    pbar=tqdm()
    @wrap_pbar(pbar)
    @wrap_for_LinearOperator
    def matvec(v):
        return contract('iIab,jJbc,kKcd,lLda,IJKL->ijkl',T0,T0,T0,T0,v.reshape(T0.shape)).reshape(-1)
    @wrap_pbar(pbar)
    @wrap_for_LinearOperator
    def rmatvec(u):
        return contract('iIab,jJbc,kKcd,lLda,ijkl->IJKL',T0,T0,T0,T0,u.conj().reshape(T0.shape)).conj().reshape(-1)
    return LinearOperator(shape=(dimT,dimT),matvec=matvec,rmatvec=rmatvec)

def get_linearized_HOTRG_autodiff(T0,layers):
    dimT=prod(T0.shape)
    pbar=tqdm()
    print(f'dimension: {dimT}x{dimT}')
    def forward_layers(v):
        v=v.reshape(T0.shape)
        for layer in layers:
            v=forward_layer(v,v,layer)
        return v.reshape(-1)
    v0=T0.reshape(-1)
    @wrap_pbar(pbar)
    @wrap_for_LinearOperator
    def matvec(v):
        # https://pytorch.org/functorch/nightly/generated/functorch.jvp.html
        _,u=jvp(forward_layers,primals=(v0,),tangents=(v,))
        return u

    # https://pytorch.org/functorch/stable/generated/functorch.vjp.html
    _,vjpfunc=vjp(forward_layers,v0)
    @wrap_pbar(pbar)
    @wrap_for_LinearOperator
    def rmatvec(u):
        v=vjpfunc(u)[0]
        return v
    return LinearOperator(shape=(dimT,dimT),matvec=matvec,rmatvec=rmatvec)

def get_linearized_HOTRG_full_autodiff(T0,options):
    dimT=prod(T0.shape)
    pbar=tqdm()
    print(f'dimension: {dimT}x{dimT}')
    def forward_layers(v):
        v=v.reshape(T0.shape)
        for i in range(len(T0.shape)//2):
            v,_=HOTRG_layer(v,v,max_dim=T0.shape[0],options=options,Tref=v)
        return v.reshape(-1)
    v0=T0.reshape(-1)
    @wrap_pbar(pbar)
    @wrap_for_LinearOperator
    def matvec(v):
        # https://pytorch.org/functorch/nightly/generated/functorch.jvp.html
        _,u=jvp(forward_layers,primals=(v0,),tangents=(v,))
        return u
    # https://pytorch.org/functorch/stable/generated/functorch.vjp.html
    _,vjpfunc=vjp(forward_layers,v0)
    @wrap_pbar(pbar)
    @wrap_for_LinearOperator
    def rmatvec(u):
        v=vjpfunc(u)[0]
        return v
    return LinearOperator(shape=(dimT,dimT),matvec=matvec,rmatvec=rmatvec)

def verify_linear_operator(M,tol=1e-9,nTests=20):
    print('checking linearity of M')
    for i in range(nTests):
        v=np.random.randn(M.shape[1])
        Mv=M._matvec(v)
        M2v=M._matvec(2*v)
        assert np.linalg.norm(2*Mv-M2v)<max(tol,tol*np.linalg.norm(M2v))

    print('checking linearity of M^H')
    for i in range(nTests):
        u=np.random.randn(M.shape[0])
        MHu=M._rmatvec(u)
        MH2u=M._rmatvec(2*u)
        assert np.linalg.norm(2*MHu-MH2u)<max(tol,tol*np.linalg.norm(MH2u))

    print('checking if M^H is the hermitian conjugate of M')
    for i in range(nTests):
        u=np.random.randn(M.shape[0])
        v=np.random.randn(M.shape[1])
        uMv=u.conj()@M._matvec(v)
        vHMHuH=v.conj()@M._rmatvec(u)
        assert (np.abs(uMv-vHMHuH.conj())<max(tol,tol*np.abs(uMv)))

    print('checking symmetric of M^H M')
    for i in range(nTests):
        u=np.random.randn(M.shape[0])
        v=np.random.randn(M.shape[0])
        uHMHMv=u.conj()@M._rmatvec(M._matvec(v))
        vHMHMu=v.conj()@M._rmatvec(M._matvec(u))
        assert (np.abs(uHMHMv-vHMHMu)<max(tol,tol*np.abs(vHMHMu)))

    print('checking symmetric of M M^H')
    for i in range(nTests):
        u=np.random.randn(M.shape[1])
        v=np.random.randn(M.shape[1])
        uHMMHv=u.conj()@M._matvec(M._rmatvec(v))
        vHMMHu=v.conj()@M._matvec(M._rmatvec(u))
        assert (np.abs(uHMMHv-vHMMHu)<max(tol,tol*np.abs(uHMMHv)))

    print('verification success')
    return True

def check_hermicity(M,tol=1e-9,nTests=20):
    assert M.shape[0]==M.shape[1]
    print('checking hermicity')
    for i in range(nTests):
        u=np.random.randn(M.shape[0])
        v=np.random.randn(M.shape[1])
        uMv=u@M._matvec(v)
        uMHv=u@M._rmatvec(v)
        error=np.abs(uMv-uMHv)
        if (np.abs(uMv-uMHv)>=max(tol,tol*np.abs(uMv))):
            print('hermicity is False')
            print('error is ',error,' / ',np.abs(uMv))
            return False
    print('hermicity is True')
    return True


def wrap_for_LinearOperator(func):
    def _wrapper(v_numpy):
        assert len(v_numpy.shape)==1 or (len(v_numpy.shape)==2 and v_numpy.shape[1]==1)
        v_torch=torch.tensor(v_numpy).reshape(-1)
        u_torch=func(v_torch)
        u_numpy=u_torch.detach().cpu().numpy()
        if len(v_torch.shape)>1:
            u_numpy=u_numpy.reshape(-1,1)
        else:
            u_numpy=u_numpy.reshape(-1)
        return u_numpy
    return _wrapper

def wrap_pbar(pbar):
    def _decorator(func):
        def _wrapper(*args1,**args2):
            rtvals=func(*args1,**args2)
            pbar.update(1)
            return rtvals
        return _wrapper
    return _decorator