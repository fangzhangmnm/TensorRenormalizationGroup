# https://arxiv.org/pdf/1412.0732.pdf
# https://arxiv.org/pdf/1509.07484.pdf



import torch
from opt_einsum import contract_path
from opt_einsum import contract as _contract
from safe_svd import svd,sqrt
from torch.linalg import qr
from dataclasses import dataclass
from tqdm.auto import tqdm
from math import prod
import numpy as np
from ScalingDimensions import get_entanglement_entropy

def contract(eq,*Ts):
    #print([T.shape for T in Ts])
    #print(contract_path(eq,*Ts))
    return _contract(eq,*Ts)

@dataclass
class TNRLayer:
    tensor_shape:'tuple(int)'
    u:'torch.Tensor'
    vL:'torch.Tensor'
    vR:'torch.Tensor'
    z:'torch.Tensor'
    
@dataclass
class TNR_options:
    max_nIter:int
    max_dim_TNR:int
    max_dim_TRG:int
    threshold_TTdiff:float
        

def TNR_layer(T,options:TNR_options):
    dimT0,dimT1=T.shape[0],T.shape[2]
    dimW=min(options.max_dim_TNR,dimT0*dimT1)
    dimB=min(options.max_dim_TRG,dimW**2)

    vLRef=get_isometry_from_transfer_tensor(T,(0,2),(1,3),max_dim=dimW)
    vRRef=get_isometry_from_transfer_tensor(T,(0,3),(1,2),max_dim=dimW)
    u=torch.eye(dimT0**2).reshape(dimT0,dimT0,dimT0,dimT0)
    if dimW<dimT0*dimT1:
        vL,vR=vLRef[:1,...],vRRef[:1,...]
        pbar=tqdm(leave=False)
        for cur_dimW in tqdm(range(1,dimW+1),leave=False):
            vL=extend_isometry_tensor(vL,vLRef,cur_dimW)
            vR=extend_isometry_tensor(vR,vRRef,cur_dimW)
            TTerr=build_TTerr_rel(T=T,vL=vL,vR=vR,u=u)
            Bee=get_entanglement_entropy(svd_tensor(build_B(T=T,vL=vL,vR=vR,u=u),(0,2),(3,1))[1])
            
            for _iter in range(options.max_nIter):
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

                pbar.set_postfix({'cur_dimW':cur_dimW,'udiff':udiff,'Bee':Bee})
                pbar.update()
                if TTerrdiff<options.threshold_TTdiff:
                    break
            if TTerrdiff>=options.threshold_TTdiff:
                print('not converged, err=',TTerrdiff)
        pbar.close()
    else:
        vL,vR=vLRef,vRRef


    B=build_B(T=T,vL=vL,vR=vR,u=u)
    BL,BR=split_tensor(B,(0,2),(3,1),max_dim=dimB)
    env_z=build_env_z(BL=BL,BR=BR,vL=vL,vR=vR) # we must use the environment to determine the split of C
    z=get_isometry_from_transfer_tensor(env_z,(0,1),(2,3),max_dim=dimB,hermitian=True)
    Tn=build_A(BL=BL,BR=BR,vL=vL,vR=vR,z=z)

    return Tn,TNRLayer(tensor_shape=T.shape,vL=vL,vR=vR,u=u,z=z)


from fix_gauge import fix_gauge,MCF_options
from HOTRGZ2 import gauge_invariant_norm

def TNR_layers(T0,nLayers,options,return_tensors=False):
    print('Generating TNR layers')
    tnr_options=TNR_options(**{k[4:]:v for k,v in options.items() if k[:4]=='tnr_'})
    mcf_options=MCF_options(**{k[4:]:v for k,v in options.items() if k[:4]=='mcf_'})
    
    T,logTotal=T0,0
    Ts,logTotals=[T],[0]
    layers=[]
    for iLayer in tqdm(list(range(nLayers)),leave=False):
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

    
# def build_C(vL,vR):
#     # \     /   0 3  0       0
#     # vR---vL    C    w2   2v 
#     #  |   |    2 1   1     1 
#     # vRd--vLd
#     # /     \
#     upb=contract('Azx,Bwx->ABzw',vL,vR)
#     lwb=contract('Azx,Bwx->ABzw',vL.conj(),vR.conj())
#     C=contract('ABzw,CDzw->ADCB',upb,lwb)
#     #C=C+contract('ADCB->CBAD',C.conj())
#     return C



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
    u,s,vh=svd(M)
    u=u.reshape(shape1+(-1,))
    vh=vh.reshape((-1,)+shape2)
    return u,s,vh
    
def qr_tensor(M,idx1,idx2):
    # return q,r, M=q@r
    shape1=tuple(M.shape[i] for i in idx1)
    shape2=tuple(M.shape[i] for i in idx2)
    M=M.permute(idx1+idx2).reshape(prod(shape1),prod(shape2))
    q,r=qr(M)
    q=q.reshape(shape1+shape1)
    r=r.reshape(shape1+shape2)
    return q,r
    

def split_tensor(M,idx1,idx2,max_dim=None):
    shape1=tuple(M.shape[i] for i in idx1)
    shape2=tuple(M.shape[i] for i in idx2)
    M=M.permute(idx1+idx2).reshape(prod(shape1),prod(shape2))
    u,s,vh=svd(M)
    s=sqrt(s)
    u,vh=(u*s)[:,:max_dim],((vh.T*s).T)[:max_dim,:]
    u=u.reshape(shape1+(-1,))
    vh=vh.reshape((-1,)+shape2)
    return u,vh

def get_isometry_from_environment(M,idx1,idx2):
    shape1=tuple(M.shape[i] for i in idx1)
    shape2=tuple(M.shape[i] for i in idx2)
    M=M.conj().permute(idx1+idx2).reshape(prod(shape1),prod(shape2))
    u,_,vh=svd(M)
    uvh=u@vh
    uvh=uvh.reshape(shape1+shape2)
    uvh=uvh.permute(invert_permutation(idx1+idx2))
    return uvh

def get_isometry_from_transfer_tensor(E,idx1,idx2,max_dim,hermitian=False):
    #wd w E ~= E
    n=len(E.shape)//2
    if hermitian:
        E=(E.permute(idx1+idx2)+E.permute(idx2+idx1).conj()).permute(invert_permutation(idx1+idx2))/2
    u,_,_=svd_tensor(E,idx1,idx2)
    w=u[...,:max_dim].permute((n,)+tuple(range(n))).conj()
    return w

def extend_isometry_tensor(w,wRef,new_dim):
    assert w.shape[1:]==wRef.shape[1:]
    assert w.shape[0]<=new_dim and new_dim<=wRef.shape[0]
    wNew=wRef[:new_dim,...]
    if w is not None and w.shape[0]>0:
        wNew[:w.shape[0]]=w
    q,r=torch.linalg.qr(wNew.reshape(new_dim,-1).T)
    wNew=q.T.reshape(new_dim,*w.shape[1:])
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

''' 
def TNR_layer(T,options:TNR_options):
    dimT=T.shape[0]
    dimW=min(options.max_dim_TNR,dimT**2)
    layer=TNRLayer(
        tensor_shape=T.shape,
        u=generate_random_isometry(dimT**2,dimT**2).reshape(dimT,dimT,dimT,dimT),
        w=generate_random_isometry(dimW,dimT**2).reshape(dimW,dimT,dimT),
        v=generate_random_isometry(dimW,dimT**2).reshape(dimW,dimT,dimT),
        )
    with tqdm(range(options.nIter),leave=False) as pbar:
        for _iter in pbar:
            Eu=build_env_u(T,layer)
            u,s,vh=svd(Eu.reshape(dimT**2,dimT**2))
            uNew=(u@vh).reshape(dimT,dimT,dimT,dimT)
            #uNew=torch.eye(dimT**2).reshape(dimT,dimT,dimT,dimT)
            diffu=(layer.u-uNew).norm()
            layer.u=uNew
            
            Ew=build_env_w(T,layer)
            u,s,vh=svd(Ew.reshape(dimW,dimT**2))
            wNew=(u@vh).reshape(dimW,dimT,dimT)
            diffw=(layer.w-wNew).norm()
            layer.w=wNew
            
            Ev=build_env_v(T,layer)
            u,s,vh=svd(Ev.reshape(dimW,dimT**2))
            vNew=(u@vh).reshape(dimW,dimT,dimT)
            diffv=(layer.v-vNew).norm()
            layer.v=vNew
            
            T1,T2=build_wuT_diff(T,layer)
            diffT=(T1-T2).norm()
            
            pbar.set_postfix({'dT':diffT,'du':diffu,'dw':diffw,'dv':diffv})
    T=forward_TNR(T,layer=layer,options=options)
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
def forward_TNR(T,layer:TNRLayer,options:TNR_options):
    B=build_B(T,layer)
    C=build_C(T,layer)
    T=TRG(B,C,options.max_dim_TRG)
    return T
    
def TNR_layer(T,options:TNR_options):
    dimT0,dimT1=T.shape[0],T.shape[1]
    dimW=min(options.max_dim_TNR,dimT0*dimT1)
    dimB=min(options.max_dim_TRG,dimW**2)

    w0=get_isometry_from_environment(T,(0,2),(1,3),max_dim=dimW)
    v0=get_isometry_from_environment(T,(0,3),(1,2),max_dim=dimW)
    u0=torch.eye(dimT0**2).reshape(dimT0,dimT0,dimT0,dimT0)


    layer=TNRLayer(
        tensor_shape=T.shape,
        u=u0,v=v0,w=w0,z=None)
    with tqdm(range(options.nIter),leave=False) as pbar:
        for _iter in pbar:
            def _r(oldValue,newValue):
                return newValue,(newValue-oldValue).norm()/oldValue.norm()

            layer.w.requires_grad_(True)
            build_upb(T=T,w=layer.w,v=layer.v,u=layer.u).norm().backward()
            layer.w,wdiff=_r(layer.w.detach(),isometrize_tensor(layer.w.grad.conj(),(0,),(1,2)))
            layer.w.requires_grad_(False)

            layer.v.requires_grad_(True)
            build_upb(T=T,w=layer.w,v=layer.v,u=layer.u).norm().backward()
            layer.v,vdiff=_r(layer.v.detach(),isometrize_tensor(layer.v.grad.conj(),(0,),(1,2)))
            layer.v.requires_grad_(False)
            
            layer.u.requires_grad_(True)
            build_upb(T=T,w=layer.w,v=layer.v,u=layer.u).norm().backward()
            layer.u,udiff=_r(layer.u.detach(),isometrize_tensor(layer.u.grad.conj(),(0,1),(2,3)))
            layer.u.requires_grad_(False)
            
            T1,T2=build_wuT_diff(T,w=layer.w,v=layer.v,u=layer.u)
            tdiff=(T1-T2).norm()
            
            pbar.set_postfix({'du':udiff,'dw':wdiff,'dv':vdiff,'dT':tdiff})

    B=build_B(T=T,w=layer.w,v=layer.v,u=layer.u)
    BL,BR=split_tensor(B,(0,2),(3,1),max_dim=dimB)
    env_z=build_env_z(BL=BL,BR=BR,w=layer.w,v=layer.v)
    layer.z=get_isometry_from_environment(env_z,(0,1),(2,3),max_dim=dimB)
    T=build_A(BL=BL,BR=BR,w=layer.w,v=layer.v,z=layer.z)
    return T,layer
'''