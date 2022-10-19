import torch
from tqdm.auto import tqdm as tqdm
from opt_einsum import contract
import torch.utils.checkpoint
import itertools
from collections import namedtuple
from dataclasses import dataclass
import math
def _toN(t):
    return t.detach().cpu().tolist() if isinstance(t,torch.Tensor) else t

from safe_svd import svd,sqrt # TODO is it necessary???
#======================== Z2 =================================

def RepDim(dimV1R1,dimV1R2,dimV2R1,dimV2R2):
    return (dimV1R1*dimV2R1+dimV1R2*dimV2R2,dimV1R1*dimV2R2+dimV1R2*dimV2R1)

def RepMat(dimV1R1,dimV1R2,dimV2R1,dimV2R2):
    dimV1=dimV1R1+dimV1R2
    dimV2=dimV2R1+dimV2R2
    P=torch.zeros([dimV1*dimV2,dimV1,dimV2])
    counter=0
    for i in range(dimV1R1):
        for j in range(dimV2R1):
            P[counter,i,j]=1
            counter+=1
    for i in range(dimV1R2):
        for j in range(dimV2R2):
            P[counter,dimV1R1+i,dimV2R1+j]=1
            counter+=1
    for i in range(dimV1R1):
        for j in range(dimV2R2):
            P[counter,i,dimV2R1+j]=1
            counter+=1
    for i in range(dimV1R2):
        for j in range(dimV2R1):
            P[counter,dimV1R1+i,j]=1
            counter+=1
    return P

def Z2_sectors(T,dimR):
    if len(T.shape)==2*len(dimR): dimR=[d for d in dimR for _ in range(2)]
    assert len(T.shape)==len(dimR) and all(i==sum(j) for i,j in zip(T.shape,dimR))
    for sector in itertools.product(range(2),repeat=len(dimR)):
        begin=[sum(dimR[leg][:rep]) for leg,rep in enumerate(sector)]
        end=[sum(dimR[leg][:rep+1]) for leg,rep in enumerate(sector)]
        slices=[slice(b,e) for b,e in zip(begin,end)]
        yield sector,slices

def Z2_sector_norm(T,dimR):
    sqrnorm=torch.zeros(2)
    for sector,slices in Z2_sectors(T,dimR):
        sqrnorm[sum(sector)%2]+=T[slices].norm()**2
    return sqrnorm**.5

def project_Z2(T,dimR,weights=[1,0],tolerance=float('inf')):
    Tn=torch.zeros(T.shape)
    sqrnorm=torch.zeros(2)
    for sector,slices in Z2_sectors(T,dimR):
        sqrnorm[sum(sector)%2]+=T[slices].norm()**2
        Tn[slices]=T[slices]*weights[sum(sector)%2]
    norm=sqrnorm**.5
    assert not(weights[1]==0 and norm[1]>norm[0]*tolerance)
    assert not(weights[0]==0 and norm[0]>norm[1]*tolerance)
    return Tn

#============================= Forward Layers ======================================

@dataclass
class HOTRGLayer:
    ww:'list[torch.Tensor]'
    dimR:'tuple[tuple[int]]'=None
    dimR_next:'tuple[tuple[int]]'=None
    gg:'list[list[torch.Tensor]]'=None

def _forward_layer_2D(Ta,Tb,layer:HOTRGLayer):
    #         |                   
    #    /g0-Ta-g1\       0       2
    #  -w     |    w-    2T3  -> 0T'1  
    #    \h0-Tb-h1/       1       3
    #         |                      
    ww,dimR,gg=layer.ww,layer.dimR,layer.gg
    if gg:
        Ta=contract('ijkl,Kk,Ll->ijKL',Ta,*gg[0])
        Tb=contract('ijkl,Kk,Ll->ijKL',Tb,*gg[1])
    if dimR:
        P=RepMat(dimR[1][0],dimR[1][1],dimR[1][0],dimR[1][1])
        wP=contract('ab,bij->aij',ww[0],P)
    else:
        wP=ww[0].reshape(-1,Ta.shape[2],Tb.shape[2])
    Tn=contract('ijkl,jmno,akn,blo->abim',Ta,Tb,wP,wP.conj())
    return Tn

def _forward_layer_3D(Ta,Tb,layer:HOTRGLayer):
    #       g4|                         5--6
    #    /g1-T1-g2\      50      34     |1--2
    #  -w   g8|g3  w-    2T3  -> 0T'1   7| 8|
    #    \g5-T2-g6/       14      52     3--4
    #         |g7 
    ww,dimR,gg=layer.ww,layer.dimR,layer.gg
    if gg:
        Ta=contract('ijklmn,Kk,Ll,Mm,Nn->ijKLMN',Ta,*gg[0])
        Tb=contract('ijklmn,Kk,Ll,Mm,Nn->ijKLMN',Tb,*gg[1])
    if dimR:
        P1=RepMat(dimR[1][0],dimR[1][1],dimR[1][0],dimR[1][1])
        wP1=contract('ab,bij->aij',ww[0],P1)
        P2=RepMat(dimR[2][0],dimR[2][1],dimR[2][0],dimR[2][1])
        wP2=contract('ab,bij->aij',ww[1],P2)
    else:
        wP1=ww[0].reshape(-1,Ta.shape[2],Tb.shape[2])
        wP2=ww[1].reshape(-1,Ta.shape[4],Tb.shape[4])
    Tn=contract('ijklmn,jopqrs,akp,blq,cmr,dns->abcdio',Ta,Tb,wP1,wP1.conj(),wP2,wP2.conj())
    return Tn

def _checkpoint(function,args,args1,use_checkpoint=True):
    if use_checkpoint and any(x.requires_grad for x in args):
        def wrapper(*args):
            return function(*args,**args1)
        return torch.utils.checkpoint.checkpoint(wrapper,*args)
    else:
        return function(*args,**args1)
    
def forward_layer(Ta,Tb,layer:HOTRGLayer,use_checkpoint=False)->torch.Tensor:
    _forward_layer={4:_forward_layer_2D,6:_forward_layer_3D}[len(Ta.shape)]
    return _checkpoint(_forward_layer,[Ta,Tb],{'layer':layer})

def gauge_invariant_norm(T):
    spacial_dim=len(T.shape)//2
    if spacial_dim==2:
        norm=contract('iijj->',T)
    elif spacial_dim==3:
        norm=contract('iijjkk->',T)
    #norm=T.norm()
    #print(norm)
    return norm
    
def forward_tensor(T0,layers:'list[HOTRGLayer]',use_checkpoint=False,return_layers=False):
    T,logTotal=T0,0
    if return_layers: 
        Ts,logTotals=[T],[0]
    for layer in tqdm(layers,leave=False):
        norm=gauge_invariant_norm(T)
        T=T/norm
        logTotal=2*(logTotal+norm.log())
        T=forward_layer(T,T,layer=layer,use_checkpoint=use_checkpoint)
        if return_layers: 
            Ts.append(T);logTotals.append(logTotal)
    return (Ts,logTotals) if return_layers else (T,logTotal)

def forward_observable_tensor(T0,T0_op,layers:'list[HOTRGLayer]',
        start_layer=0,checkerboard=False,use_checkpoint=False,return_layers=False,cached_Ts=None):
    spacial_dim=len(T0.shape)//2
    T,logTotal=forward_tensor(T0,layers=layers[:start_layer],use_checkpoint=use_checkpoint,return_layers=return_layers)
    T_op=T0_op
    if return_layers:
        Ts,T,logTotals,logTotal=T,T[-1],logTotal,logTotal[-1]
        T_ops=[None]*start_layer+[T_op]
    for ilayer,layer in tqdm(list(enumerate(layers))[start_layer:],leave=False):
        norm=gauge_invariant_norm(T)
        T,T_op=T/norm,T_op/norm
        logTotal=2*(logTotal+norm.log())
        if cached_Ts:
            T1=cached_Ts[ilayer+1]
        else:
            T1=forward_layer(T,T,layer=layer,use_checkpoint=use_checkpoint)
        T2=forward_layer(T,T_op,layer=layer,use_checkpoint=use_checkpoint)
        T3=forward_layer(T_op,T,layer=layer,use_checkpoint=use_checkpoint)
        T3=-T3 if (checkerboard and ilayer<spacial_dim) else T3
        T,T_op=T1,(T2+T3)/2
        if return_layers:
            Ts.append(T);T_ops.append(T_op);logTotals.append(logTotal)
    return (Ts,T_ops,logTotals) if return_layers else (T,T_op,logTotal)

    
def forward_observalbe_tensor_moments(T0_moments:'list[torch.Tensor]',layers:'list[HOTRGLayer]',
        checkerboard=False,use_checkpoiont=False,return_layers=False,cached_Ts=None):
    # -T'[OO]- = -T[OO]-T[1]- + 2 -T[O]-T[O]- + -T[1]-T[OO]-      
    spacial_dim=len(T0.shape)//2
    logTotal=0
    Tms=T0_moments.copy()
    if return_layers:
        Tmss,logTotals=[Tms],[logTotal]
    for ilayer,layer in tqdm(list(enumerate(layers)),leave=False):
        norm=gauge_invariant_norm(Tms[0])
        Tms=[x/norm for x in Tms]
        logTotal=2*(logTotal+norm.log())
        Tms1=[torch.zeros_like(Tms[0])]*len(Tms)
        for a in range(len(Tms)):
            for b in range(len(Tms)):
                if a+b<len(Tms1):
                    if a+b==0 and cached_Ts:
                        Tms1[a+b]=cached_Ts[iLayers+1]
                    else:
                        Tms1[a+b]=math.comb(a+b,b)\
                            *forward_layer(Tms[a],Tms[b],layer=layer,use_checkpoint=use_checkpoint)
        Tms=Tms1
        if return_layers:
            Tmss.append(Tms);logTotals.append(logTotal)
            return (Tmss,logTotals) if return_layers else (Tms,logTotal)
    
    
def forward_two_observable_tensors(T0,T0_op1,T0_op2,coords:"list[int]",layers:'list[HOTRGLayer]',checkerboard=False,use_checkpoint=False,cached_Ts=None):
    spacial_dim=len(T0.shape)//2
    nLayers=len(layers)
    T,T_op1,T_op2,T_op12,logTotal=T0,T0_op1,T0_op2,None,0
    for ilayer,layer in tqdm(list(enumerate(layers)),leave=False):
        norm=gauge_invariant_norm(T)
        T,T_op1,T_op2,T_op12=(t/norm if t is not None else None for t in (T,T_op1,T_op2,T_op12))
        logTotal=2*(logTotal+norm.log())
        #Evolve vacuum T
        if cached_Ts:
            T1=cached_Ts[ilayer+1]
        else:
            T1=forward_layer(T,T,layer=layer,use_checkpoint=use_checkpoint)
        #Evolve defected T depends on whether the two defects are in the same coarse-grained block
        if not all(c==0 for c in coords):
            c=coords[0]%2
            T2=forward_layer(T_op1,T,layer=layer,use_checkpoint=use_checkpoint)
            if c==0:
                T3=forward_layer(T_op2,T,layer=layer,use_checkpoint=use_checkpoint)
            elif c==1:
                T3=forward_layer(T,T_op2,layer=layer,use_checkpoint=use_checkpoint)
                T3=-T3 if checkerboard and ilayer<spacial_dim else T3
            coords=coords[1:]+[coords[0]//2]
            T,T_op1,T_op2=T1,T2,T3
        elif T_op1 is not None and T_op2 is not None:
            T2=forward_layer(T_op1,T_op2,layer=layer,use_checkpoint=use_checkpoint)
            T2=-T2 if checkerboard and ilayer<spacial_dim else T2
            T,T_op12,T_op1,T_op2=T1,T2,None,None
        else:
            T2=forward_layer(T_op12,T,layer=layer,use_checkpoint=use_checkpoint)
            T,T_op12=T1,T2
    return T,T_op12,logTotal
    
    
    
def trace_tensor(T):
    eq={4:'aabb->',6:'aabbcc->'}[len(T.shape)]
    return contract(eq,T)

def trace_two_tensors(T):
    eq={4:'abcc,badd->',6:'abccdd,baeeff->'}[len(T.shape)]
    return contract(eq,T,T)

def get_w_random(dimRn0,dimRn1,max_dim):
    max_dim-=max_dim%2
    dimRnn0,dimRnn1=max_dim//2,max_dim-max_dim//2
    w=torch.zeros((max_dim,dimRn0+dimRn1))
    w[:dimRnn0,:dimRn0]=generate_random_isometry(dimRnn0,dimRn0)
    w[dimRnn0:,dimRn0:]=generate_random_isometry(dimRnn1,dimRn1)
    return w,dimRnn0,dimRnn1

def generate_random_isometry(dim0,dim1):
    dim=max(dim0,dim1)
    A=torch.randn(dim,dim)
    U=torch.matrix_exp(A-A.t())
    U=U[:dim0,:dim1]
    return U

def generate_isometries_random(dimR:"tuple[tuple[int]]",max_dim,nLayers):
    layers=[]
    spacial_dim=len(dimR)
    for ilayer in range(nLayers):
        ww=[]
        dimRn=(dimR[-1],)
        for i in range(1,spacial_dim):
            w,dimR0,dimR1=get_w_random(dimR[i][0],dimR[i][1],max_dim)
            ww.append(w)
            dimRn.append([dimR0,dimR1])
            dimRn+=((dimR0,dimR1),)
        layers.append(HOTRGLayer(ww=ww,dimR=dimR,dimR_next=dimRn))
        dimR=dimRn
    return layers

#============================= Generate Isometries ======================================

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

        

def _HOTRG_layer_3D(T1,T2,max_dim,dimR:"tuple[tuple[int]]"=None):
    MM1=contract('ijklmn,jopqrs,itulmn,tovqrs->kpuv',T1,T2,T1.conj(),T2.conj())
    MM2=contract('ijklmn,jopqrs,itklun,topqvs->mruv',T1,T2,T1.conj(),T2.conj())
    if dimR:
        P1=RepMat(dimR[1][0],dimR[1][1],dimR[1][0],dimR[1][1])
        dimRn1=RepDim(dimR[1][0],dimR[1][1],dimR[1][0],dimR[1][1])
        MM1=contract('ijIJ,aij,AIJ->aA',MM1,P1,P1.conj())
        P2=RepMat(dimR[2][0],dimR[2][1],dimR[2][0],dimR[2][1])
        dimRn2=RepDim(dimR[2][0],dimR[2][1],dimR[2][0],dimR[2][1])
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
    return Tn,HOTRGLayer(ww=[w1,w2],dimR=dimR,dimR_next=dimR_next)
    
def _HOTRG_layer_2D(T1,T2,max_dim,dimR:"tuple[tuple[int]]"=None):
    MM=contract('ijkl,jmno,ipql,pmro->knqr',T1,T2,T1.conj(),T2.conj())
    if dimR:
        P=RepMat(dimR[1][0],dimR[1][1],dimR[1][0],dimR[1][1])
        dimRn=RepDim(dimR[1][0],dimR[1][1],dimR[1][0],dimR[1][1])
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
    return Tn,HOTRGLayer(ww=[w],dimR=dimR,dimR_next=dimR_next)

def HOTRG_layer(T1,T2,max_dim,dimR:"tuple[tuple[int]]"=None)->"tuple[torch.Tensor,HOTRGLayer]":
    _HOTRG_layer={4:_HOTRG_layer_2D,6:_HOTRG_layer_3D}[len(T1.shape)]
    return _HOTRG_layer(T1,T2,max_dim=max_dim,dimR=dimR)


def HOTRG_layers(T0,max_dim,nLayers,dimR:"tuple[tuple[int]]"=None,return_tensors=False,HOTRG_layer=HOTRG_layer):    
    spacial_dim=len(T0.shape)//2
    T,logTotal=T0,0
    if return_tensors:
        Ts,logTotals=[T],[0]
    layers=[]
    for ilayer in tqdm(list(range(nLayers)),leave=False):
        norm=gauge_invariant_norm(T)
        T=T/norm
        logTotal=2*(logTotal+norm.log())
        T,layer=HOTRG_layer(T,T,max_dim=max_dim,dimR=dimR)
        dimR=layer.dimR_next

        #uncomment the following line to sanity check if T can be reproduced by the layers
        #assert ((forward_layer(Ts[-1]/gauge_invariant_norm(Ts[-1]),Ts[-1]/gauge_invariant_norm(Ts[-1]),layer)-T).norm()==0)

        layers.append(layer)
        if return_tensors:
            Ts.append(T);logTotals.append(logTotal)
    return (layers,Ts,logTotals) if return_tensors else layers


    