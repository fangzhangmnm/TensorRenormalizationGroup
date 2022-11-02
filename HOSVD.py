import torch
from tqdm.auto import tqdm as tqdm
from opt_einsum import contract

from safe_svd import svd,sqrt # TODO is it necessary???

from HOTRGZ2 import RepMat,RepDim,HOTRGLayer,gauge_invariant_norm


def get_w_HOSVD(MM:torch.Tensor,max_dim,dimRn:"tuple[int]"=None):
    # w MM wh
    if dimRn is None:
        #S,U=torch.linalg.eigh(MM)#ascending, U S Uh=MM #will fail when there's a lot of zero eigenvalues
        #S,U=S.flip(0),U.flip(-1)
        U,S,Vh=svd(MM)
        w=(U.T)[:max_dim]
        return w
    else:
        MM0,MM1=MM[:dimRn[0],:dimRn[0]],MM[dimRn[0]:,dimRn[0]:]
        #S0,U0=torch.linalg.eigh(MM0)#ascending, U S Uh=MM
        #S1,U1=torch.linalg.eigh(MM1)
        U0,S0,Vh0=svd(MM0)
        U1,S1,Vh1=svd(MM1)
        S,U=[S0,S1],[U0,U1]
        max_dim=min(max_dim,sum(dimRn))
        chosenEigens=sorted([(-s,0,i) for i,s in enumerate(S0)]+[(-s,1,i) for i,s in enumerate(S1)])[:max_dim]
        chosenEigens.sort(key=lambda x:x[1])
        
        shift=[0,dimRn[0]]
        dimRnn=[0,0]
        w=torch.zeros((max_dim,sum(dimRn)))
        for i,(s,rep,col) in enumerate(chosenEigens):
            w[i,shift[rep]:shift[rep]+dimRn[rep]]=U[rep][:,col]
            dimRnn[rep]+=1
        return w,tuple(dimRnn)

# Not needed automatically satisfied why
#def get_w_HOSVD_nonlinear(MM:torch.Tensor,max_dim,dimRn:"tuple[int]"=None):
#    # w MM wh
#    if dimRn is None:
#        #S,U=torch.linalg.eigh(MM)#ascending, U S Uh=MM #will fail when there's a lot of zero eigenvalues
#        #S,U=S.flip(0),U.flip(-1)
#        w=None
#        S=None
#        for _iter in range(10):
#            MMw=MM@w.T.conj() if w is not None else MM
#            Sold=S
#            U,S,Vh=svd(MMw)
#            w=(U.T)[:max_dim]
#            if Sold is not None:
#                print((S[:max_dim]-Sold[:max_dim]).norm())
#        print('---')
#        return w
#    else:
#        raise NotImplementedError
#
#get_w_HOSVD=get_w_HOSVD_nonlinear 

def _RepMatDim(a,b):
    return RepMat(a,b,a,b),RepDim(a,b,a,b)

def _HOSVD_layer_3D(T1,T2,max_dim,dimR:"tuple[tuple[int]]"=None):
    MM1=contract('ijklmn,jopqrs,itulmn,tovqrs->kpuv',T1,T2,T1.conj(),T2.conj())
    MM2=contract('ijklmn,jopqrs,itklun,topqvs->mruv',T1,T2,T1.conj(),T2.conj())
    if dimR:
        P1,dimRn1=_RepMatDim(dimR[1][0],dimR[1][1])
        MM1=contract('ijIJ,aij,AIJ->aA',MM1,P1,P1.conj())
        P2,dimRn2=_RepMatDim(dimR[2][0],dimR[2][1])
        MM2=contract('ijIJ,aij,AIJ->aA',MM2,P2,P2.conj())

        w1,dimRnn1=get_w_HOSVD(MM1,max_dim=max_dim,dimRn=dimRn1)
        wP1=contract('ab,bij->aij',w1,P1)
        w2,dimRnn2=get_w_HOSVD(MM2,max_dim=max_dim,dimRn=dimRn2)
        wP2=contract('ab,bij->aij',w2,P2)
        
        dimR_next=(dimRnn1,dimRnn2,dimR[0])
    else:
        MM1=MM1.reshape(T1.shape[2]*T2.shape[2],-1)
        MM2=MM2.reshape(T1.shape[4]*T2.shape[4],-1)

        w1=get_w_HOSVD(MM1,max_dim=max_dim,dimRn=None)
        wP1=w1.reshape(-1,T1.shape[2],T2.shape[2])
        w2=get_w_HOSVD(MM2,max_dim=max_dim,dimRn=None)
        wP2=w2.reshape(-1,T1.shape[4],T2.shape[4])
        
        dimR_next=None

    Tn=contract('ijklmn,jopqrs,akp,blq,cmr,dns->abcdio',T1,T2,wP1,wP1.conj(),wP2,wP2.conj())
    return Tn,HOTRGLayer(tensor_shape=T1.shape,ww=[w1,w2],dimR=dimR,dimR_next=dimR_next)
    
def _HOSVD_layer_2D(T1,T2,max_dim,dimR:"tuple[tuple[int]]"=None):
    MM=contract('ijkl,jmno,ipql,pmro->knqr',T1,T2,T1.conj(),T2.conj())
    if dimR:
        P,dimRn=_RepMatDim(dimR[1][0],dimR[1][1])
        MM=contract('ijIJ,aij,AIJ->aA',MM,P,P.conj())

        w,dimRnn=get_w_HOSVD(MM,max_dim=max_dim,dimRn=dimRn)
        wP=contract('ab,bij->aij',w,P)
        
        dimR_next=(dimRnn,dimR[0])
    else:
        MM=MM.reshape(T1.shape[2]*T2.shape[2],-1)

        w=get_w_HOSVD(MM,max_dim=max_dim)
        wP=w.reshape(-1,T1.shape[2],T2.shape[2])
        
        dimR_next=None
        
    Tn=contract('ijkl,jmno,akn,blo->abim',T1,T2,wP,wP.conj())
    return Tn,HOTRGLayer(tensor_shape=T1.shape,ww=[w],dimR=dimR,dimR_next=dimR_next)

def HOSVD_layer(T1,T2,max_dim,dimR:"tuple[tuple[int]]"=None)->"tuple[torch.Tensor,HOTRGLayer]":
    _HOSVD_layer={4:_HOSVD_layer_2D,6:_HOSVD_layer_3D}[len(T1.shape)]
    return _HOSVD_layer(T1,T2,max_dim=max_dim,dimR=dimR)





def _HOSVD_layer_2D_PEPS(T1,T2,max_dim,max_dim_P,dimR:"tuple[tuple[int]]"=None):
    MM1=contract('ijklA,jmnoB,ipqlA,pmroB->knqr',T1,T2,T1.conj(),T2.conj())
    MMP=contract('ijkla,jmnob,iJklA,JmnoB->abAB',T1,T2,T1.conj(),T2.conj())
    if dimR:
        P1,dimRn1=_RepMatDim(dimR[1][0],dimR[1][1])
        MM1=contract('ijIJ,aij,AIJ->aA',MM1,P1,P1.conj())
        PP,dimRnP=_RepMatDim(dimR[2][0],dimR[2][1])
        MMP=contract('ijIJ,aij,AIJ->aA',MMP,PP,PP.conj())

        w1,dimRnn1=get_w_HOSVD(MM1,max_dim=max_dim,dimRn=dimRn1)
        wP,dimRnnP=get_w_HOSVD(MMP,max_dim=max_dim_P,dimRn=dimRnP)

        wP1=contract('ab,bij->aij',w1,P1)
        wPP=contract('ab,bij->aij',wP,PP)
        
        dimR_next=(dimRnn1,dimR[0],dimRnnP)
    else:
        MM1=MM1.reshape(T1.shape[2]*T2.shape[2],-1)
        MMP=MMP.reshape(T1.shape[4]*T2.shape[4],-1)

        w1=get_w_HOSVD(MM1,max_dim=max_dim)
        wP=get_w_HOSVD(MMP,max_dim=max_dim_P)

        wP1=w1.reshape(-1,T1.shape[2],T2.shape[2])
        wPP=wP.reshape(-1,T1.shape[4],T2.shape[4])
        
        dimR_next=None
        
    Tn=contract('ijkla,jmnob,Jkn,Klo,Aab->JKimA',T1,T2,wP1,wP1.conj(),wPP)
    return Tn,HOTRGLayer(tensor_shape=T1.shape,ww=[w1,wP],dimR=dimR,dimR_next=dimR_next)


def HOSVD_layer_PEPS(T1,T2,max_dim,dimR:"tuple[tuple[int]]"=None)->"tuple[torch.Tensor,HOTRGLayer]":
    _HOSVD_layer_PEPS={5:_HOSVD_layer_2D_PEPS}[len(T1.shape)]
    return _HOSVD_layer_PEPS(T1,T2,max_dim=max_dim,dimR=dimR)





'''
def HOSVD_layers(T0,max_dim,nLayers,dimR:"tuple[tuple[int]]"=None,return_tensors=False,HOSVD_layer=HOSVD_layer):    
    spacial_dim=len(T0.shape)//2
    T,logTotal=T0,0
    if return_tensors:
        Ts,logTotals=[T],[0]
    layers=[]
    for ilayer in tqdm(list(range(nLayers)),leave=False):
        norm=gauge_invariant_norm(T)
        T=T/norm
        logTotal=2*(logTotal+norm.log())
        T,layer=HOSVD_layer(T,T,max_dim=max_dim,dimR=dimR)
        dimR=layer.dimR_next

        #uncomment the following line to sanity check if T can be reproduced by the layers
        #assert ((forward_layer(Ts[-1]/gauge_invariant_norm(Ts[-1]),Ts[-1]/gauge_invariant_norm(Ts[-1]),layer)-T).norm()==0)

        layers.append(layer)
        if return_tensors:
            Ts.append(T);logTotals.append(logTotal)
    return (layers,Ts,logTotals) if return_tensors else layers

'''