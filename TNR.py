# https://arxiv.org/pdf/1412.0732.pdf
# https://arxiv.org/pdf/1509.07484.pdf



import torch
from opt_einsum import contract_path
from opt_einsum import contract
#from safe_svd import svd,sqrt
from torch.linalg import svdvals,svd,qr
from torch import sqrt,lobpcg 
from dataclasses import dataclass
from tqdm.auto import tqdm
from math import prod
import numpy as np
from ScalingDimensions import get_entanglement_entropy
from fix_gauge import fix_gauge,MCF_options
from HOTRGZ2 import gauge_invariant_norm

def _toN(t):
    return t.detach().cpu().tolist() if isinstance(t,torch.Tensor) else t

# def contract(eq,*Ts):
#     #print([T.shape for T in Ts])
#     #print(contract_path(eq,*Ts))
#     return _contract(eq,*Ts)

@dataclass
class TNRLayer:
    tensor_shape:'tuple(int)'
    u:'torch.Tensor'
    vL:'torch.Tensor'
    vR:'torch.Tensor'
    z:'torch.Tensor'
    
@dataclass
class TNR_options:
    max_nIter:int=200
    max_dim_TNR:int=8
    max_dim_TRG:int=16
    threshold_TTdiff:float=1e-7
    disentangling_method:str='relaxing'
        
        

def TNR_layer(T,options:TNR_options):
    dimT0,dimT1=T.shape[0],T.shape[2]
    dimW=min(options.max_dim_TNR,dimT0*dimT1)
    dimB=min(options.max_dim_TRG,dimW**2)

    getuv=disentangling_methods[options.disentangling_method]
    vL,vR,u=getuv(T,options)

    B=build_B(T=T,vL=vL,vR=vR,u=u)
    BL,BR=split_tensor(B,(0,2),(3,1),max_dim=dimB)
    env_z=build_env_z(BL=BL,BR=BR,vL=vL,vR=vR) # we must use the environment to determine the split of C
    z=get_isometry_from_transfer_tensor(env_z,(0,1),(2,3),max_dim=dimB,hermitian=True)
    Tn=build_A(BL=BL,BR=BR,vL=vL,vR=vR,z=z)

    return Tn,TNRLayer(tensor_shape=T.shape,vL=vL,vR=vR,u=u,z=z)
    
def TNR_layers(T0,nLayers,options,return_tensors=False):
    print('Generating TNR layers')
    tnr_options=TNR_options(**{k[4:]:v for k,v in options.items() if k[:4]=='tnr_'})
    mcf_options=MCF_options(**{k[4:]:v for k,v in options.items() if k[:4]=='mcf_'})
    
    T,logTotal=T0,0
    Ts,logTotals=[T],[0]
    layers=[]
    for iLayer in tqdm(list(range(nLayers)),leave=False):
        print('TNR layer',iLayer)
        norm=gauge_invariant_norm(T)
        T=T/norm
        logTotal=2*(logTotal+norm.log())
        
        T,layer=TNR_layer(T,options=tnr_options)
        if options.get('mcf_enabled',False):
            T,hh=fix_gauge(T,Ts[-1],options=mcf_options)

        layers.append(layer)
        Ts.append(T);logTotals.append(logTotal)
    print('TNR layers generated')
    return (layers,Ts,logTotals) if return_tensors else layers
    
    
#========== get disentangler ==========
disentangling_methods={}
def _register_disentangling_method(name):
    def _decorator(func):
        disentangling_methods[name]=func
        return func
    return _decorator
    
@_register_disentangling_method('relaxing')    
def getuv_relaxing(T,options):
    dimT0,dimT1=T.shape[0],T.shape[2]
    dimW=min(options.max_dim_TNR,dimT0*dimT1)
    
    vLRef=get_isometry_from_transfer_tensor(T,(0,2),(1,3),max_dim=dimW)
    vRRef=get_isometry_from_transfer_tensor(T,(0,3),(1,2),max_dim=dimW)
    u=torch.eye(dimT0**2).reshape(dimT0,dimT0,dimT0,dimT0)
    if dimW<dimT0*dimT1:
        vL,vR=None,None
        for cur_dimW in tqdm(range(1,dimW+1),leave=False):
            vL=extend_isometry_tensor(vL,vLRef,cur_dimW)
            vR=extend_isometry_tensor(vR,vRRef,cur_dimW)
            vL,vR,u=optimize_uv(T=T,vL=vL,vR=vR,u=u,options=options)
            benchmark_disentangler(U=u,A=build_TT_A(T))
    else:
        vL,vR=vLRef,vRRef
    return vL,vR,u
    
    
def optimize_uv(T,vL,vR,u,options):
    TTerr,TTerrdiff=build_TTerr_rel(T=T,vL=vL,vR=vR,u=u),0
    Bee=get_entanglement_entropy(svd_tensor(build_B(T=T,vL=vL,vR=vR,u=u),(0,2),(3,1))[1])
    pbar=tqdm(range(options.max_nIter))
    print('optimizing u, initial entanglement= ',_toN(Bee))
    for _iter in pbar:
        def update(oldValue,newValue,absolute=False):
            return newValue,(newValue-oldValue).norm()*(1 if absolute else oldValue.norm())

        env_vL=build_env_vL(T=T,vL=vL,vR=vR,u=u)
        vL,vLdiff=update(vL,get_isometry_from_environment(env_vL,(0,),(1,2)))

        env_vR=build_env_vR(T=T,vL=vL,vR=vR,u=u)
        vR,vRdiff=update(vR,get_isometry_from_environment(env_vR,(0,),(1,2)))

        env_u=build_env_u(T=T,vL=vL,vR=vR,u=u)
        u,udiff=update(u,get_isometry_from_environment(env_u,(0,1),(2,3)))

        Bee,Beediff=update(Bee,get_entanglement_entropy(svd_tensor(build_B(T=T,vL=vL,vR=vR,u=u),(0,2),(3,1))[1]))

        TTerr,TTerrdiff=update(TTerr,build_TTerr_rel(T=T,vL=vL,vR=vR,u=u),absolute=True)

        pbar.set_postfix({'TTerr':_toN(TTerr),'Bee':_toN(Bee)})
        if TTerrdiff<options.threshold_TTdiff:
            break
    if TTerrdiff>=options.threshold_TTdiff:
        print('optimizing not converged, err=',TTerrdiff)
    print('final entanglement=',_toN(Bee))
    return vL,vR,u


@_register_disentangling_method('fast')
def getuv_fast(T,options):
    dimT0,dimT1=T.shape[0],T.shape[2]
    dimW=min(options.max_dim_TNR,dimT0*dimT1)
    A=build_TT_A(T)
    u=get_disentangler_fast(A)
    benchmark_disentangler(U=u,A=build_TT_A(T))
    vL=get_isometry_from_transfer_tensor(T,(0,2),(1,3),max_dim=dimW)
    vR=get_isometry_from_transfer_tensor(T,(0,3),(1,2),max_dim=dimW)
    vL,vR,u=optimize_uv(T,vL,vR,u,options)
    benchmark_disentangler(U=u,A=build_TT_A(T))
    return vL,vR,u




# https://arxiv.org/pdf/2104.08283.pdf
def get_disentangler_fast(A:torch.Tensor):
    # i   j
    # uuuuu
    # I   J   0  1
    # AAAAA   AAAA
    # a   b   2  3
    print('getting fast entangler')
    dimi,dimj,dima,dimb=A.shape
    assert dimi<=dima and dimj<=dimb
    r=torch.randn(dimi,dimj)
    rA=contract('IJ,IJab->ab',r,A)
    s,u=lobpcg(rA.conj()@rA.T,k=1)
    aa=u[:,0]
    s,u=lobpcg(rA.T.conj()@rA,k=1)
    ab=u[:,0]
    AIJa=contract('IJab,b->IJa',A,ab).reshape(dimi*dimj,dima)
    AIJb=contract('IJab,a->IJb',A,aa).reshape(dimi*dimj,dimb)
    Via=svd(AIJa,full_matrices=False)[2][:dimi,:]
    Vjb=svd(AIJb,full_matrices=False)[2][:dimj,:]
    BijIJ=contract('ia,jb,IJab->ijIJ',Via,Vjb,A.conj())
    UijIJ=svd_tensor_to_isometry(BijIJ,(0,1),(2,3))
    print('fast entangler retrieved')
    return UijIJ

def get_disentangler_relaxing(A,dimVL,dimVR,nIter=100):
    dimi,dimj,dima,dimb=A.shape
    vLRef=get_isometry_from_transfer_tensor(A,(0,2),(1,3),max_dim=dimVL)
    vRRef=get_isometry_from_transfer_tensor(A,(0,3),(1,2),max_dim=dimVR)
    u=torch.eye(dimi*dimj).reshape(dimi,dimj,dimi,dimj)
    vL,vR=None,None
    for cur_dimW in tqdm(range(1,max(dimVL,dimVR)+1),leave=False):
        vL=extend_isometry_tensor(vL,vLRef,min(cur_dimW,dimVL))
        vR=extend_isometry_tensor(vR,vRRef,min(cur_dimW,dimVR))
        for _iter in tqdm(range(nIter)):
            vL,vR,u=optimize_disentangler_relaxing(A=A,vL=vL,vR=vR,u=u)
    return u

def refine_disentangler_relaxing(A,u,dimVL,dimVR,nIter=100):
    vL=get_isometry_from_transfer_tensor(A,(0,2),(1,3),max_dim=dimVL)
    vR=get_isometry_from_transfer_tensor(A,(0,3),(1,2),max_dim=dimVR)
    for _iter in tqdm(range(nIter)):
        vL,vR,u=optimize_disentangler_relaxing(A=A,vL=vL,vR=vR,u=u)
    return u


def optimize_disentangler_relaxing(A,vL,vR,u):
    #  \     /  0 3
    #  vL┐ ┌vR   B   #note that the leg ordering is different in TNR
    #   |uuu|   2 1 
    #   |AAA|   
    #   └┘ └┘
    vuA=contract('xia,yjb,ijIJ,IJab->xy',vL,vR,u,A)
    env_vL=contract('yjb,ijIJ,IJab,xy->xia',vR,u,A,vuA.conj())
    vL=get_isometry_from_environment(env_vL,(0,),(1,2))
    env_vR=contract('xia,ijIJ,IJab,xy->yjb',vL,u,A,vuA.conj())
    vR=get_isometry_from_environment(env_vR,(0,),(1,2))
    env_u=contract('xia,yjb,IJab,xy->ijIJ',vL,vR,A,vuA.conj())
    u=get_isometry_from_environment(env_u,(0,1),(2,3))
    return vL,vR,u


    
def benchmark_disentangler(U,A):
    UA=contract('ijIJ,IJab->ijab',U,A)

    ee1=get_entanglement_entropy(svdvals_tensor(A,(0,2),(1,3)))
    ee2=get_entanglement_entropy(svdvals_tensor(UA,(0,2),(1,3)))
    print('entanglement removed by disentangler: ',_toN(ee1),' -> ',_toN(ee2))
    


#========== TNR tensors ===========
    

def build_A(BL,BR,vL,vR,z):
    #     z  
    #   vRd-vLd    0     0 
    #-BR     BL-   z    2A3
    #   vR--vL    1 2    1 
    #     zd  
    upa=build_upa(BL=BL,BR=BR,vL=vL,vR=vR)
    return contract('axy,xycdzw,bzw->abcd',z,upa,z.conj())

def build_upa(BL,BR,vL,vR):
    #   |  |
    #  vRd-vLd   0      0   0 1
    #-BR    BL-   w2  2v   2upa3
    #  vR--vL     1    1    4 5
    #   |  |
    return contract('jxi,kyi,ajl,kmb,lzn,mwn->xyabzw',vR.conj(),vL.conj(),BR,BL,vR,vL)

def build_env_z(BL,BR,vL,vR):
    upb_z=build_upa(BL,BR,vL,vR)
    M=contract('xyijkl,XYijkl->xyXY',upb_z,upb_z.conj())
    M=(M+contract('xyXY->XYxy',M.conj()))/2
    return M

def unbuild_B(B,vL,vR,u):
    #  i m 
    # kT-To
    #  | | 
    # qT-Tt
    #  p s
    TTTT=contract('ADCB,Aik,Bmo,Cpq,Dst,imIM,psPS->IMPSkqot',B,vL.conj(),vR.conj(),vL,vR,u.conj(),u)
    return TTTT

def build_TTTT(T,flip_second_row=False):
    Td=T.permute(1,0,2,3).conj() if flip_second_row else T
    return contract('ijkl,mnlo,jpqr,nsrt->impskqot',T,T,Td,Td)

def build_B(T,vL,vR,u,flip_second_row=True):
    #  \     /  0 3
    # vL'┐ ┌vR'  B 
    #   | u |   2 1 
    #   └T-T┘   
    #   ┌T-T┐    
    #   | ud|
    #vL'd┘ └vR'd
    #  /     \
    Td=T.conj() if flip_second_row else T.permute(1,0,2,3)
    upb=build_upb(T=T,vL=vL,vR=vR,u=u)
    lwb=build_upb(T=Td,vL=vL.conj(),vR=vR.conj(),u=u.conj())
    B=contract('ABjn,CDjn->ADCB',upb,lwb)
    B=(B+contract('ADCB->CBAD',B.conj()))/2
    return B

def build_upb(T,vL,vR,u):
    #  \     /    0       0   0 1    A       B      
    # vL'┐ ┌vR'  vL'1   1vR'  upb    vLx   yvR       
    #   |uuu|      2+   +2    2 3     k uuu o          
    #   || ||        0 1       0      | i m |      
    #   └T-T┘         u       2T3     └kTlTo┘      
    #    | |         2 3       1        j n        
    # warning ! vL,vR index order is different here!
    return contract('Axk,Byo,xyim,ijkl,mnlo->ABjn',vL,vR,u,T,T)

def build_TTerr_rel(T,vL,vR,u):
    # |TT-wd w TT|^2=|TT|^2-|w TT'|^2
    sqrnorm1=contract('ijkl,mnlo,ijkL,mnLo->',T,T,T.conj(),T.conj())
    sqrnorm2=build_upb(T=T,vL=vL,vR=vR,u=u).norm()**2

    return (1-sqrnorm2/sqrnorm1)**.5


def unbild_upb(upb,vL,vR,u):
    #  i m 
    # kT-To
    #  j n 
    return contract('ABjn,Aik,Bmo,imIM->IMjnko',upb,vL.conj(),vR.conj(),u.conj())

def build_TT(T):
    return contract('ijkl,mnlo->imjnko',T,T)
    
def build_TT_A(T):
    dimT0,dimT1=T.shape[0],T.shape[2]
    return contract('ijkl,mnlo->imjkno',T,T).reshape(dimT0,dimT0,dimT0*dimT1,dimT0*dimT1)
    

def build_env_vL(T,vL,vR,u):
    # upb=contract('Axk,xyim,ijkl,mnlo->Ayojn',vL,u,T,T).conj()
    # return contract('Ayojn,xyim,ijkl,mnlo->Axk',upb,u,T,T)
    upb=build_upb(T=T,vL=vL,vR=vR,u=u)
    return contract('ABjn,Byo,xyim,ijkl,mnlo->Axk',upb,vR,u,T,T)


def build_env_vR(T,vL,vR,u):
    # upb=contract('Byo,xyim,ijkl,mnlo->xkBjn',vR,u,T,T).conj()
    # return contract('xkBjn,xyim,ijkl,mnlo->Byo',upb,u,T,T)
    upb=build_upb(T=T,vL=vL,vR=vR,u=u)
    return contract('ABjn,Axk,xyim,ijkl,mnlo->Byo',upb,vL,u,T,T)

def build_env_u(T,vL,vR,u):
    upb=contract('Axk,Byo,xyim,ijkl,mnlo->ABjn',vL,vR,u,T,T).conj()
    return contract('ABjn,Axk,Byo,ijkl,mnlo->xyim',upb,vL,vR,T,T)

    


#========== utilities ==========
    
def generate_random_isometry(dim0,dim1):
    dim=max(dim0,dim1)
    A=torch.randn(dim,dim)
    U=torch.matrix_exp(A-A.t())
    U=U[:dim0,:dim1]
    return U

def generate_random_isometry_tensor(shape0,shape1):
    dim0,dim1=np.prod(shape0),np.prod(shape1)
    return generate_random_isometry(dim0,dim1).reshape(shape0+shape1)


def svd_tensor(M,idx1,idx2):
    # returns u,s,vh, M=u*s@vh
    shape1=tuple(M.shape[i] for i in idx1)
    shape2=tuple(M.shape[i] for i in idx2)
    M=M.permute(idx1+idx2).reshape(prod(shape1),prod(shape2))
    u,s,vh=svd(M,full_matrices=False)
    u=u.reshape(shape1+(-1,))
    vh=vh.reshape((-1,)+shape2)
    return u,s,vh
    
def svdvals_tensor(M,idx1,idx2,k=None):
    shape1=tuple(M.shape[i] for i in idx1)
    shape2=tuple(M.shape[i] for i in idx2)
    M=M.permute(idx1+idx2).reshape(prod(shape1),prod(shape2))
    return svdvals(M,driver='gesvd')
    # MMh=M@M.T.conj()
    # if MMh.shape[1]<64:
    #     s=svdvals(MMh)
    # else:
    #     k=k or MMh.shape[1]//3
    #     s,u=lobpcg(MMh,k=k)
    # return s**.5
    
def qr_tensor(M,idx1,idx2):
    # return q,r, M=q@r
    shape1=tuple(M.shape[i] for i in idx1)
    shape2=tuple(M.shape[i] for i in idx2)
    M=M.permute(idx1+idx2).reshape(prod(shape1),prod(shape2))
    q,r=qr(M)
    q=q.reshape(shape1+(-1,))
    r=r.reshape((-1,)+shape2)
    return q,r
    

def split_tensor(M,idx1,idx2,max_dim=None):
    shape1=tuple(M.shape[i] for i in idx1)
    shape2=tuple(M.shape[i] for i in idx2)
    M=M.permute(idx1+idx2).reshape(prod(shape1),prod(shape2))
    u,s,vh=svd(M,full_matrices=False)
    s=sqrt(s)
    u,vh=(u*s)[:,:max_dim],((vh.T*s).T)[:max_dim,:]
    u=u.reshape(shape1+(-1,))
    vh=vh.reshape((-1,)+shape2)
    return u,vh

def svd_tensor_to_isometry(M,idx1,idx2):
    shape1=tuple(M.shape[i] for i in idx1)
    shape2=tuple(M.shape[i] for i in idx2)
    M=M.permute(idx1+idx2).reshape(prod(shape1),prod(shape2))
    u,_,vh=svd(M,full_matrices=False)
    uvh=(u@vh).reshape(shape1+shape2).permute(invert_permutation(idx1+idx2))
    return uvh

def get_isometry_from_environment(M,idx1,idx2):
    return svd_tensor_to_isometry(M,idx1,idx2).conj()

def get_isometry_from_transfer_tensor(E,idx1,idx2,max_dim,hermitian=False):
    #wd w E ~= E
    n=len(E.shape)//2
    if hermitian:
        E=(E.permute(idx1+idx2)+E.permute(idx2+idx1).conj()).permute(invert_permutation(idx1+idx2))/2
    u,_,_=svd_tensor(E,idx1,idx2)
    w=u[...,:max_dim].permute((n,)+tuple(range(n))).conj()
    return w

def extend_isometry_tensor(w,wRef,new_dim):
    assert new_dim<=wRef.shape[0]
    wNew=wRef[:new_dim,...]
    if w is not None:
        assert w.shape[1:]==wRef.shape[1:]
        assert w.shape[0]<=new_dim and new_dim<=wRef.shape[0]
        wNew[:w.shape[0]]=w
    q,r=qr(wNew.reshape(new_dim,-1).T)
    wNew=q.T.reshape(new_dim,*wRef.shape[1:])
    return wNew

def invert_permutation(permutation):
    permutation=np.asarray(permutation)
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return tuple(inv)

#========== comparisons ==========

@dataclass
class TRGLayer:
    tensor_shape:'tuple(int)'
    u:'torch.Tensor'
    vL:'torch.Tensor'
    vR:'torch.Tensor'
    z:'torch.Tensor'

def TRG(B,C,max_dim):
    #       |
    #       C        0   0 3 
    #   --B   B--   2A3   B  
    #       C        1   2 1 
    #       |
    chi=B.shape[0]
    max_dim=min(chi**2,max_dim)
    BL,BR=split_tensor(B,(0,2),(3,1),max_dim=max_dim)
    CU,CD=split_tensor(C,(0,3),(2,1),max_dim=max_dim)
    A=contract('iab,cdj,kac,bdl->ijkl',CD,CU,BR,BL)
    return A


def HOTRG4x4_layer(T,max_dim,flip_second_row=False):
    #     w        i m  
    #   T---T     kTlTo     
    #  w|   |wd    j n    
    #   T---T     qTrTt  
    #     wd       p s
    Td=T.permute(1,0,2,3).conj() if flip_second_row else T
    env_w0=contract('ijkl,mnlo,jpqr,nsrt,IJkL,MNLo,JpqR,NsRt->imIM',T,T,Td,Td,T.conj(),T.conj(),Td.conj(),Td.conj())
    w0=get_isometry_from_transfer_tensor(env_w0,(0,1),(2,3),max_dim=max_dim,hermitian=True)
    env_w1=contract('ijkl,mnlo,jpqr,nsrt,iJKL,mNLo,JpQR,NsRt->kqKQ',T,T,Td,Td,T.conj(),T.conj(),Td.conj(),Td.conj())
    w1=get_isometry_from_transfer_tensor(env_w1,(0,1),(2,3),max_dim=max_dim,hermitian=True)
    T=contract('ijkl,mnlo,jpqr,nsrt,Iim,Jps,Kkq,Lot->IJKL',T,T,Td,Td,w0,w0.conj(),w1,w1.conj())
    return T,[w0,w1]


def HOTRG4x4_layer1(TTTT,max_dim):
    #     w        i m  
    #   T---T     kTlTo     
    #  w|   |wd    j n    
    #   T---T     qTrTt  
    #     wd       p s
    env_w0=contract('impskqot,IMpskqot->imIM',TTTT,TTTT.conj())
    w0=get_isometry_from_transfer_tensor(env_w0,(0,1),(2,3),max_dim=max_dim,hermitian=True)
    env_w1=contract('impskqot,impsKQot',TTTT,TTTT.conj())
    w1=get_isometry_from_transfer_tensor(env_w1,(0,1),(2,3),max_dim=max_dim,hermitian=True)
    T=contract('impskqot,Iim,Jps,Kkq,Lot->IJKL',TTTT,w0,w0.conj(),w1,w1.conj())
    return T,[w0,w1]



def HOTRG4x4_layers(T0,nLayers,options={},return_tensors=False):
    mcf_options=MCF_options(**{k[4:]:v for k,v in options.items() if k[:4]=='mcf_'})

    T,logTotal=T0,0
    Ts,logTotals=[T],[0]
    layers=[]
    for iLayer in tqdm(list(range(nLayers)),leave=False):
        norm=gauge_invariant_norm(T)
        T=T/norm
        logTotal=2*(logTotal+norm.log())
        
        T,layer=HOTRG4x4_layer(T,max_dim=options['max_dim'],flip_second_row=options['flip_second_row'])

        if options.get('mcf_enabled',False):
            T,hh=fix_gauge(T,Ts[-1],options=mcf_options)
            layer.append(hh)
            
        layers.append(layer)
        Ts.append(T);logTotals.append(logTotal)
    return (layers,Ts,logTotals) if return_tensors else layers

def TRG_layers(T0,nLayers,options={},return_tensors=False):
    mcf_options=MCF_options(**{k[4:]:v for k,v in options.items() if k[:4]=='mcf_'})

    T,logTotal=T0,0
    Ts,logTotals=[T],[0]
    layers=[]
    for iLayer in tqdm(list(range(nLayers)),leave=False):
        norm=gauge_invariant_norm(T)
        T=T/norm
        logTotal=2*(logTotal+norm.log())
        
        T,layer=TRG(T,T,max_dim=options['max_dim']),None
        
        if options.get('mcf_enabled',False):
            T,hh=fix_gauge(T,Ts[-1],options=mcf_options)

        layers.append(layer)
        Ts.append(T);logTotals.append(logTotal)
    return (layers,Ts,logTotals) if return_tensors else layers
