import torch
from opt_einsum import contract
from safe_svd import svd,sqrt,split_matrix
from dataclasses import dataclass


@dataclass
class TNRLayer:
    tensor_shape:'tuple(int)'
    u:'torch.Tensor'
    w:'torch.Tensor'
    v:'torch.Tensor'
    
@dataclass
class TNR_options:
    max_dim:int
    nIter:int
        
def TNR_layer(T,options):
    max_dim=options.max_dim
    dimT=T.shape[0]
    dimW=min(max_dim,dimT**2)
    layer=TNRLayer()
    layer.tensor_shape=T.shape
    layer.u=generate_random_isometry(dimT**2,dimT**2).reshape(dimT,dimT,dimT,dimT)
    layer.w=generate_random_isometry(dimW,dimT**2).reshape(dimW,dimT,dimT)
    layer.v=generate_random_isometry(dimW,dimT**2).reshape(dimW,dimT,dimT)
    with tqdm(range(options.nIter)) as pbar:
        for _iter in pbar:
            Eu=build_env_u(T,layer)
            u,s,vh=svd(Eu.reshape(dimT**2,dimT**2))
            layer.u=(u@vh).reshape(dimT,dimT,dimT,dimT)
            Ew=build_env_w(T,layer)
            u,s,vh=svd(Ew.reshape(dimW,dimT**2))
            layer.w=(u@vh).reshape(dimW,dimT,dimT)
            Ev=build_env_v(T,layer)
            u,s,vh=svd(Ev.reshape(dimW,dimT**2))
            layer.v=(u@vh).reshape(dimW,dimT,dimT)
    T=forward_TNR(T,layer)
    return T,layer
    
def build_env_u(T,layer):
    return contract('xyim,axk,byo,aXK,bYO,ijkl,mnlo,IjKL,MnLO->XYIM',
        layer.u.conj(),
        layer.w.conj(),layer.v.conj(),layer.w,layer.v,
        T.conj(),T.conj(),T,T)
        
def build_env_w(T,layer):
    return contract('xyim,XYIM,axk,byo,bYO,ijkl,mnlo,IjKL,MnLO->aXK',
        layer.u.conj(),layer.u,
        layer.w.conj(),layer.v.conj(),layer.v,
        T.conj(),T.conj(),T,T)

def build_env_v(T,layer):
    return contract('xyim,XYIM,axk,byo,aXK,ijkl,mnlo,IjKL,MnLO->bYO',
        layer.u.conj(),layer.u,
        layer.w.conj(),layer.v.conj(),layer.w,
        T.conj(),T.conj(),T,T)
        
def build_B(T,layer):
    upb=contract('Axk,Byo,ijkl,mnlo->ABjn',layer.w,layer.v,layer.u,T,T)
    return contract('ABjn,CDjn->ABCD',upb,upb.conj())

def build_C(T,layer):
    return contract('Axz,Bxw,Cyz,Dyw->ABCD',layer.w,layer.v,layer.v.conj(),layer.w.conj())

def forward_TNR(T,layer:TNRLayer):
    chi=layer.w.shape[0]
    B=build_B(T,layer)
    C=build_C(T,layer)
    BL,BR=split_matrix(contract('ABCD->ACBD',B).reshape(chi**2,-1))
    CU,CD=split_matrix(C.reshape(chi**2,-1))
    BL=BL.reshape(chi,chi,-1)
    BR=BR.reshape(-1,chi,chi)
    CU=CU.reshape(chi,chi,-1)
    CD=CD.reshape(-1,chi,chi)
    A=contract('iAB,CDj,kAC,BDj->ijkl',BR,BL,CD,CU)
    return A
    
def generate_random_isometry(dim0,dim1):
    dim=max(dim0,dim1)
    A=torch.randn(dim,dim)
    U=torch.matrix_exp(A-A.t())
    U=U[:dim0,:dim1]
    return U