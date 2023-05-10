from opt_einsum import contract
import torch
#from safe_svd import svd,sqrt # TODO is it necessary???
from torch.linalg import svd
from torch import sqrt
from tqdm.auto import tqdm
import dataclasses
from dataclasses import dataclass
from math import prod
import numpy as np


def dcontract(derivative,eq,*tensors,**kwargs):
    assert all(tensor is not None for tensor in tensors)
    assert len(list(tensor for tensor in tensors if id(tensor)==id(derivative)))==1, f'{id(derivative)%3533} {[id(t)%3533 for t in tensors]}'
    idx = next(i for i, tensor in enumerate(tensors) if id(tensor)==id(derivative))
    eq_terms=eq.split(',')
    eq=','.join(eq_terms[:idx]+eq_terms[idx+1:])+'->'+eq_terms[idx]
    tensors=tensors[:idx]+tensors[idx+1:]
    return contract(eq,*tensors,**kwargs)

def svd_tensor_to_isometry(M,idx1=(0,),idx2=(1,)):
    shape1=tuple(M.shape[i] for i in idx1)
    shape2=tuple(M.shape[i] for i in idx2)
    M=M.permute(idx1+idx2).reshape(prod(shape1),prod(shape2))
    u,_,vh=torch.linalg.svd(M,full_matrices=False)
    uvh=(u@vh).reshape(shape1+shape2).permute(tuple(np.argsort(idx1+idx2)))
    return uvh

def get_isometry_from_environment(M,idx1=(0,),idx2=(1,)):
    return svd_tensor_to_isometry(M,idx1,idx2).conj()

    
def contract_all_legs(T1,T2:torch.Tensor)->torch.Tensor:
    T1i,T2i=[*range(len(T1.shape))],[*range(len(T2.shape))]
    return contract(T1,T1i,T2,T2i)

def contract_all_legs_but_one(T1,T2:torch.Tensor,i:int)->torch.Tensor:
    T1i,T2i=[*range(len(T1.shape))],[*range(len(T2.shape))]
    T1i[i],T2i[i]=-1,-2
    return contract(T1,T1i,T2,T2i,[-1,-2])

def sum_all_legs_but_one(T,i:int)->torch.Tensor:
    Ti=[*range(len(T.shape))]
    return contract(T,Ti,[i])

def apply_matrix_to_leg(T:torch.Tensor,M:torch.Tensor,i:int)->torch.Tensor:
    Ti,Mi=[*range(len(T.shape))],[-1,i]
    Tni=Ti.copy();Tni[i]=-1
    return contract(T,Ti,M,Mi,Tni)

def apply_vector_to_leg(T:torch.Tensor,M:torch.Tensor,i:int)->torch.Tensor:
    Ti,Mi=[*range(len(T.shape))],[i]
    Tni=Ti.copy()
    return contract(T,Ti,M,Mi,Tni)


@dataclass
class MCF_options:
    enabled:bool=True
    eps:float=1e-6
    max_iter:int=50
    enabled_unitary:bool=True
    phase_iter1:int=3
    phase_iter2:int=10

def fix_phase_2D(T,Tref,options:MCF_options=MCF_options()):
    #if Tref[0,0,0,0]<0:Tref=-Tref
    #if T[0,0,0,0]<0:T=-T

    hs=[torch.eye(T.shape[i]) for i in range(4)]
    for _j in range(options.phase_iter1):
        ds=[torch.ones(T.shape[i]) for i in range(4)]
        for _i in range(1,max(T.shape)):
            i=_i%max(T.shape)
            if len(T.shape)==4:
                TT,TTref=T[:,:i,:i,:i],Tref[:,:i,:i,:i]
            else:
                TT,TTref=T[:,:i,:i,:i,0],Tref[:,:i,:i,:i,0]
            rho1=contract('ijkl,ijkl->i',TT,TTref)
            di=torch.where(rho1>0,1.,-1.)
            ds[0],ds[1]=ds[0]*di,ds[1]*di
            #T=contract('ijkl,i,j->ijkl',T,di,di)
            T=apply_vector_to_leg(T,di,0)
            T=apply_vector_to_leg(T,di,1)
            
            if len(T.shape)==4:
                TT,TTref=T[:i,:i,:,:i],Tref[:i,:i,:,:i]
            else:
                TT,TTref=T[:i,:i,:,:i,0],Tref[:i,:i,:,:i,0]
            rho1=contract('ijkl,ijkl->k',TT,TTref)
            di=torch.where(rho1>0,1.,-1.)
            ds[2],ds[3]=ds[2]*di,ds[3]*di
            #T=contract('ijkl,k,l->ijkl',T,di,di)
            T=apply_vector_to_leg(T,di,2)
            T=apply_vector_to_leg(T,di,3)

        hs=[torch.diag(di)@h for di,h in zip(ds,hs)]
        T,hs1=fix_gauge_svd_2D(T,Tref,nIter=options.phase_iter2)
        hs=[h1@h for h1,h in zip(hs1,hs)]

    return T,hs


def minimal_canonical_form(T:torch.Tensor,options:MCF_options=MCF_options())->'tuple[torch.Tensor,list[torch.Tensor]]':
        # The minimal canonical form of a tensor network
        # https://arxiv.org/pdf/2209.14358.pdf
    spacial_dim=len(T.shape)//2
    hh=[torch.eye(T.shape[i]) for i in range(spacial_dim*2)]
    if options.enabled:
        for iIter in range(options.max_iter):
            total_diff=0
            for k in range(spacial_dim):
                tr_rho=contract_all_legs(T,T.conj())
                rho1=contract_all_legs_but_one(T,T.conj(),2*k)
                rho2=contract_all_legs_but_one(T,T.conj(),2*k+1).T
                rho_diff=rho1-rho2
                assert (rho_diff-rho_diff.T.conj()).norm()/tr_rho<1e-7
                total_diff+=rho_diff.norm()**2/tr_rho
                g1=torch.matrix_exp(-rho_diff/(4*spacial_dim*tr_rho))
                g2=torch.matrix_exp(rho_diff/(4*spacial_dim*tr_rho)).T
                hh[2*k]=g1@hh[2*k]
                hh[2*k+1]=g2@hh[2*k+1]
                T=apply_matrix_to_leg(T,g1,2*k)
                T=apply_matrix_to_leg(T,g2,2*k+1)
            if total_diff<options.eps**2:
                break
    return T,hh
    
    
# it seems unitary is already restored because rho_diff is always zero
def fix_unitary_gauge(T,Tref,options:MCF_options=MCF_options()):
    spacial_dim=len(T.shape)//2
    hh=[torch.eye(T.shape[i]) for i in range(spacial_dim*2)]
    for iIter in range(options.max_iter):
        total_diff=0
        for k in range(spacial_dim):
            tr_rho=contract_all_legs(T,Tref.conj())
            rho1=contract_all_legs_but_one(T,Tref.conj(),2*k)
            rho2=contract_all_legs_but_one(T,Tref.conj(),2*k).T
            rho_diff=rho1-rho2
            print(rho_diff.norm()/tr_rho)
            assert (rho_diff+rho_diff.T.conj()).norm()/tr_rho<1e-7
            rho_diff=(rho_diff-rho_diff.T.conj())/2
            total_diff+=rho_diff.norm()**2/tr_rho
            g1=torch.matrix_exp(-rho_diff/(4*spacial_dim*tr_rho))
            g2=g1
            #g2=torch.matrix_exp(rho_diff/(4*spacial_dim*tr_rho))
            hh[2*k]=g1@hh[2*k]
            hh[2*k+1]=g2@hh[2*k+1]
            T=apply_matrix_to_leg(T,g1,2*k)
            T=apply_matrix_to_leg(T,g2,2*k+1)
        if total_diff<options.eps**2:
            break
    return T,hh
    
    

def fix_gauge_svd_2D(T,Tref,nIter=10):
    dim1,dim2=T.shape[0],T.shape[2]
    h1,h2=torch.eye(dim1),torch.eye(dim2)
    for i in range(nIter):
        env_h1=dcontract(h1,'ijkl,IJKL,Ii,Jj,Kk,Ll',T,Tref,h1,h1.conj().clone(),h2,h2.conj().clone())
        h1=get_isometry_from_environment(env_h1)
        env_h2=dcontract(h2,'ijkl,IJKL,Ii,Jj,Kk,Ll',T,Tref,h1,h1.conj().clone(),h2,h2.conj().clone())
        h2=get_isometry_from_environment(env_h2)
    return contract('ijkl,Ii,Jj,Kk,Ll->IJKL',T,h1,h1.conj(),h2,h2.conj()),[h1,h1.conj(),h2,h2.conj()]


# def fix_phase_2D(T,Tref):
#     spacial_dim=len(T.shape)//2
#     ds=[torch.ones(T.shape[i]) for i in range(spacial_dim*2)]
#     # chose the dimension which having the biggest impact when flipping
#     for _i in range(3*max(T.shape)):
#         rho1=contract('ijkl,ijkl->i',T,Tref)-contract('iikl,iikl->i',T,Tref)
#         rho2=contract('ijkl,ijkl->k',T,Tref)-contract('ijkk,ijkk->k',T,Tref)
#         argmax1,max1=torch.argmax(rho1),torch.max(-rho1)
#         argmax2,max2=torch.argmax(rho2),torch.max(-rho2)
#         argmax,max_,axis=(argmax1,max1,0) if max1>max2 else (argmax2,max2,1)

        
#         if max_>0:
#             print('Fliping indice {} at axis {} with value {}'.format(argmax,axis,max_))
#             di=torch.ones(T.shape[axis]);di[argmax]=-1
#             ds[axis*2+0]=ds[axis*2+0]*di
#             ds[axis*2+1]=ds[axis*2+1]*di
#             T=apply_vector_to_leg(T,di,axis*2+0)
#             T=apply_vector_to_leg(T,di,axis*2+1)
#         else:
#             break
#     return T,[torch.diag(di) for di in ds]


# def fix_phase_2D(T,Tref):
#     #if Tref[0,0,0,0]<0:Tref=-Tref
#     #if T[0,0,0,0]<0:T=-T
    
#     spacial_dim=len(T.shape)//2
#     ds=[torch.ones(T.shape[i]) for i in range(spacial_dim*2)]
#     for _i in range(0,4*max(T.shape)):
#         i=_i%max(T.shape)
#         ii=_i//max(T.shape)
#         if i==0:
#             continue
#         # if ii%2==0:
#         #     TT,TTref=T[:,:,:i,:i,...],Tref[:,:,:i,:i,...]
#         #     rho1=contract('ijkk,ijKK->i',TT,TTref)-contract('iikk,iiKK->i',TT,TTref)
#         # else:
#         TT,TTref=T[:,:,:i,:i,...],Tref[:,:,:i,:i,...]
#         rho1=contract('ijkl,ijkl->i',TT,TTref)-contract('iikl,iikl->i',TT,TTref)
#         di=torch.where(rho1>=0,1.,-1.)
#         ds[0],ds[1]=ds[0]*di,ds[1]*di
#         #T=contract('ijkl,i,j->ijkl',T,di,di)
#         T=apply_vector_to_leg(T,di,0)
#         T=apply_vector_to_leg(T,di,1)
        
#         # if ii%2==0:
#         #     TT,TTref=T[:i,:i,:,:,...],Tref[:i,:i,:,:,...]
#         #     rho1=contract('iikl,IIkl->k',TT,TTref)-contract('iikk,IIkk->k',TT,TTref)
#         # else:
#         TT,TTref=T[:i,:i,:,:,...],Tref[:i,:i,:,:,...]
#         rho1=contract('ijkl,ijkl->k',TT,TTref)-contract('ijkk,ijkk->k',TT,TTref)
#         di=torch.where(rho1>=0,1.,-1.)
#         ds[2],ds[3]=ds[2]*di,ds[3]*di
#         #T=contract('ijkl,k,l->ijkl',T,di,di)
#         T=apply_vector_to_leg(T,di,2)
#         T=apply_vector_to_leg(T,di,3)
#     return T,[torch.diag(di) for di in ds]


def fix_phase(T,Tref):
    _fix_phase={4:fix_phase_2D,5:fix_phase_2D}[len(T.shape)]
    return _fix_phase(T,Tref)
    
def fix_gauge(T,Tref=None,options:MCF_options=MCF_options()):
    T,hh=minimal_canonical_form(T,options=options)
    if Tref is not None and T.shape==Tref.shape:
        T,hh1=fix_phase(T,Tref,options=options)
        hh=[h1@h for h1,h in zip(hh1,hh)]
    return T,hh
            

    

# Legacy




def fix_HOTRG_gauges(Ts,layers):
    layers,Ts=[dataclasses.replace(layer) for layer in layers],Ts.copy()
    spacial_dim=len(Ts[0].shape)//2
    stride=spacial_dim
    for i in tqdm(range(1,len(Ts)),leave=False):
        Ts[i],hh=minimal_canonical_form(Ts[i])
        if i>=stride and Ts[i].shape==Ts[i-stride].shape:
            Ts[i],hh1=fix_phase(Ts[i],Ts[i-stride])
            hh=[h1@h for h1,h in zip(hh1,hh)]
        if i-1>=0:
            layers[i-1].hh=hh[-2:]+hh[:-2]
        if i+1<len(Ts):
            hhinv=[torch.inverse(h) for h in hh]
            if layers[i].gg:
                layers[i].gg=[[g@hinv for g,hinv in zip(ggg,hhinv)]for ggg in layers[i].gg]
            else:
                layers[i].gg=[hhinv.copy(),hhinv.copy()]

    return Ts,layers


'''
def fix_phase1_2D(T):
    if T[0,0,0,0]<0:T=-T
    for i in range(1,max(T.shape)):
        if i<T.shape[0]:
            TT=T[:,:i,:i,:i]
            di=torch.where(contract('ijkl->i',TT)>=0,1.,-1.)
            T=contract('ijkl,i,j->ijkl',T,di,di)
        if i<T.shape[2]:
            TT=T[:i,:i,:,:i]
            di=torch.where(contract('ijkl->k',TT)>=0,1.,-1.)
            T=contract('ijkl,k,l->ijkl',T,di,di)
    return T

    '''

def fix_gauge_2D(T,Tref):
    T,_=fix_gauge_ij(T)
    T=T.permute(2,3,0,1)
    T,_=fix_gauge_ij(T)
    T=T.permute(2,3,0,1)
    T,_=fix_phase_2D(T,Tref)
    return T

def fix_gauges(Ts:'list[torch.Tensor]',is_HOTRG=False):
    Ts=Ts.copy()
    spacial_dim=len(Ts[0].shape)//2
    stride=spacial_dim if is_HOTRG else 1
    for i in range(stride,len(Ts)):
        if Ts[i].shape==Ts[i-stride].shape:
            if spacial_dim==2:
                Ts[i]=fix_gauge_2D(Ts[i],Ts[i-stride])
            else:
                raise NotImplementedError
    return Ts
                
                


#def fix_phase_ij(T):
#    # makes the component of Ti??? with largest abs positive
#    TT=T.reshape(T.shape[0],-1)
#    di=torch.where(torch.max(TT,dim=1).values>=-torch.min(TT,dim=1).values,1,-1)
#    
#    T=contract('ijkl,i,j->ijkl',T,di,di)
#    return T
#
#fix_gauge_phase_iter=2
#
#def fix_phase(T,Tref):
#    for j in range(fix_gauge_phase_iter):
#        T=fix_phase_ij(T)
#        T=T.permute(2,3,0,1)
#        T=fix_phase_ij(T)
#        T=T.permute(2,3,0,1)
#    return T



#def fix_phase(T,Tref):
#    for j in range(1):
#        di=contract('ijkl,ijkl->i',T,Tref).sign()
#        T=contract('ijkl,i,j->ijkl',T,di,di)
#        
#        di=contract('ijkl,ijkl->k',T,Tref).sign()
#        T=contract('ijkl,k,l->ijkl',T,di,di)
#        
#        di=contract('ijkl,ijkl->j',T,Tref).sign()
#        T=contract('ijkl,i,j->ijkl',T,di,di)
#        
#        di=contract('ijkl,ijkl->l',T,Tref).sign()
#        T=contract('ijkl,k,l->ijkl',T,di,di)
#    return T

def fix_gauge_ij(T):
    #      vi     0
    #    --T--   2T3
    #      v      1
    A=contract('ijkl,Ijkl->iI',T,T.conj())
    u,s,vh=svd(A) # I don't know what I'm doing but it works better than eig in ensuring sign-fixing
    
    v,vi=u,u.T
    T=contract('ijkl,Ii,jJ->IJkl',T,vi,v)
    return T,vi
    
def fix_gauges1(Ts:'list[torch.Tensor]',is_HOTRG=False):
    Ts=Ts.copy()
    spacial_dim=len(Ts[0].shape)//2
    stride=spacial_dim if is_HOTRG else 1
    for i in tqdm(range(len(Ts)),leave=False):
        Ts[i],_=minimal_canonical_form(Ts[i])
        if i>=stride and Ts[i].shape==Ts[i-stride].shape:
            Ts[i],_=fix_phase(Ts[i],Ts[i-stride])
    return Ts