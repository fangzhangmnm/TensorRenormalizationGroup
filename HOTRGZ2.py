import torch
from tqdm.auto import tqdm as tqdm
from opt_einsum import contract
import torch.utils.checkpoint
import itertools as itt
import functools
from collections import namedtuple
from dataclasses import dataclass
import math
import numpy as np
import copy

def _toN(t):
    if isinstance(t,list):
        return [_toN(tt) for tt in t]
    elif isinstance(t,torch.Tensor):
        return t.detach().cpu().tolist()
    else:
        return t
    
def _toP(t):
    if isinstance(t,list):
        return [_toP(tt) for tt in t]
    elif isinstance(t,torch.Tensor):
        return t.detach().cpu().numpy()
    else:
        return t

#from safe_svd import svd,sqrt # TODO is it necessary???
from torch.linalg import svd
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

def _RepMat(a,b):
    return RepMat(a,b,a,b)

def Z2_sectors(T,dimR):
    if len(T.shape)==2*len(dimR): dimR=[d for d in dimR for _ in range(2)]
    assert len(T.shape)==len(dimR) and all(i==sum(j) for i,j in zip(T.shape,dimR))
    for sector in itt.product(range(2),repeat=len(dimR)):
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

def is_isometry(g):
    return torch.isclose(g@g.T.conj(),torch.eye(g.shape[0])).all()


@dataclass
class HOTRGLayer:
    tensor_shape:'tuple(int)'
    ww:'list[torch.Tensor]'
    dimR:'tuple[tuple[int]]'=None
    dimR_next:'tuple[tuple[int]]'=None
    gg:'list[list[torch.Tensor]]'=None
    hh:'list[list[torch.Tensor]]'=None
    _gg1=None
    
    def _gg(self,iNode,iLeg):
        # if self._gg1 is None and BypassGilt.v[iNode]:
        #     self.prepare_bypass_gilt()
        # return self._gg1[iNode][iLeg] if BypassGilt.v[iNode] else self.gg[iNode][iLeg]
        return self.gg[iNode][iLeg]

    def make_gg_isometric(self):
        if self.gg is None: return
        def make_isometric(g):
            u,s,vh=svd(g)
            g=u@vh
            assert is_isometry(g)
            return g
        for i in range(len(self.gg)):
            for j in range(0,len(self.gg[i]),2):
                self.gg[i][j]=make_isometric(self.gg[i][j])
                # self.gg[i][j+1]=make_isometric(self.gg[i][j+1])
                self.gg[i][j+1]=self.gg[i][j].conj() # not quite correct, since w depends on g[i][j+1], but we erased it
                assert torch.isclose(self.gg[i][j]@self.gg[i][j+1].T.conj(),torch.eye(self.gg[i][j].shape[0])).all()
        # self.gg=[[make_isometric(g) for g in gg] for gg in self.gg]
        # self.gg=None

    def get_isometry(self,i):
        #         h0
        #         g00                   
        #    /g02-Ta-g03\       0       2
        #h2-w0    |g..  w0-h3  2T3  -> 0T'1  
        #    \g12-Tb-g13/       1       3
        #         g11                      
        #         h1
        iAxis=i//2
        if iAxis==0: #first virtual leg
            w=torch.eye(self.tensor_shape[i])
            if self.gg:
                w=self._gg(i,i)@w
            if self.hh:
                w=self.hh[i]@w
        elif iAxis<len(self.tensor_shape)//2: #other virtual legs
            w=self.ww[iAxis-1]
            if self.dimR:
                P=_RepMat(self.dimR[iAxis][0],self.dimR[iAxis][1])
                w=contract('ab,bij->aij',w,P)
            else:
                w=w.reshape(-1,self.tensor_shape[i],self.tensor_shape[i])
            if i%2==1:
                w=w.conj()
            if self.gg:
                w=contract('aij,iI,jJ->aIJ',w,self._gg(0,i),self._gg(1,i))
            if self.hh:
                w=contract('aij,Aa->Aij',w,self.hh[i])
        else: #physical leg
            w=self.ww[iAxis-1]
            if self.dimR:
                P=_RepMat(self.dimR[iAxis][0],self.dimR[iAxis][1])
                w=contract('ab,bij->aij',w,P)
            else:
                w=w.reshape(-1,self.tensor_shape[i],self.tensor_shape[i])
        return w
    def get_insertion(self):
        if self.gg:
            return self._gg(0,1).T@self._gg(1,0)
        else:
            return torch.eye(self.tensor_shape[0])
    def delete_PEPS_(self):
        if(len(self.tensor_shape)%2==0):
            print('Warning! Theres no PEPS.')
            return
        self.tensor_shape=self.tensor_shape[:-1]
        self.ww=self.ww[:-1]
        if self.dimR:
            self.dimR=self.dimR[:-1]
            self.dimR_next=self.dimR_next[:-1]
    # def prepare_bypass_gilt(self):
    #     self._gg1=[[to_unitary(g) for g in ggg]for ggg in self.gg]
        
# class BypassGilt:
#     v=[False,False]
#     def __init__(self,*v):
#         self.u=v
#     def __enter__(self):
#         BypassGilt.v,self.u=self.u,BypassGilt.v
#     def __exit__(self, exc_type, exc_value, exc_tb):
#         BypassGilt.v,self.u=self.u,BypassGilt.v
        

def _forward_layer(Ta,Tb,layer:HOTRGLayer):
    assert layer.tensor_shape==Ta.shape and layer.tensor_shape==Tb.shape
    isometries=[layer.get_isometry(i) for i in range(len(layer.tensor_shape))]
    insertion=layer.get_insertion()
    eq={4:'ijkl,Jmno,jJ,xi,ym,akn,blo->abxy',
        5:'ijklA,JmnoB,jJ,xi,ym,akn,blo,CAB->abxyC',
        6:'ijklmn,Jopqrs,jJ,xi,yo,akp,blq,cmr,dns->abcdxy',
        }[len(layer.tensor_shape)]
    T=contract(eq,Ta,Tb,insertion,*isometries)
    return T


# def _forward_layer_2D(Ta,Tb,layer:HOTRGLayer):
#     #         h0
#     #         g00                   
#     #    /g02-Ta-g03\       0       2
#     #h2-w     |g     w-h3  2T3  -> 0T'1  
#     #    \g12-Tb-g13/       1       3
#     #         g11                      
#     #         h1
    
#     ww,dimR,gg,hh,T0Shape=layer.ww,layer.dimR,layer.gg,layer.hh,layer.tensor_shape
#     assert T0Shape==Ta.shape and T0Shape==Tb.shape
#     if gg:
#         Ta=contract('ijkl,Ii,Jj,Kk,Ll->IJKL',Ta,*gg[0])
#         Tb=contract('ijkl,Ii,Jj,Kk,Ll->IJKL',Tb,*gg[1])
#     if dimR:
#         P=RepMat(dimR[1][0],dimR[1][1],dimR[1][0],dimR[1][1])
#         wP=contract('ab,bij->aij',ww[0],P)
#     else:
#         wP=ww[0].reshape(-1,Ta.shape[2],Tb.shape[2])
#     Tn=contract('ijkl,jmno,akn,blo->imab',Ta,Tb,wP,wP.conj())
#     if hh:
#         Tn=contract('ijkl,Ii,Jj,Kk,Ll->IJKL',Tn,*hh)
#     Tn=contract('ijab->abij',Tn)
#     return Tn

# def _forward_layer_3D(Ta,Tb,layer:HOTRGLayer):
#     #      g|                         5--6
#     #    /g-T1-g\      50      34     |1--2
#     # h-w  g|g  w-h   2T3  -> 0T'1    7| 8|
#     #    \g-T2-g/      14      52      3--4
#     #       |g 
#     ww,dimR,gg,hh,T0Shape=layer.ww,layer.dimR,layer.gg,layer.hh,layer.tensor_shape
#     assert T0Shape==Ta.shape and T0Shape==Tb.shape
#     if gg:
#         Ta=contract('ijklmn,Ii,Jj,Kk,Ll,Mm,Nn->IJKLMN',Ta,*gg[0])
#         Tb=contract('ijklmn,Ii,Jj,Kk,Ll,Mm,Nn->IJKLMN',Tb,*gg[1])
#     if dimR:
#         P1=RepMat(dimR[1][0],dimR[1][1],dimR[1][0],dimR[1][1])
#         wP1=contract('ab,bij->aij',ww[0],P1)
#         P2=RepMat(dimR[2][0],dimR[2][1],dimR[2][0],dimR[2][1])
#         wP2=contract('ab,bij->aij',ww[1],P2)
#     else:
#         wP1=ww[0].reshape(-1,Ta.shape[2],Tb.shape[2])
#         wP2=ww[1].reshape(-1,Ta.shape[4],Tb.shape[4])
#     Tn=contract('ijklmn,jopqrs,akp,blq,cmr,dns->ioabcd',Ta,Tb,wP1,wP1.conj(),wP2,wP2.conj())
#     if hh:
#         Tn=contract('ijklmn,Ii,Jj,Kk,Ll,Mm,Nn->IJKLMN',Tn,*hh)
#     Tn=contract('ijabcd->abcdij')
#     return Tn

    

def _checkpoint(function,args,args1,use_checkpoint=True):
    if use_checkpoint and any(x.requires_grad for x in args):
        def wrapper(*args):
            return function(*args,**args1)
        return torch.utils.checkpoint.checkpoint(wrapper,*args)
    else:
        return function(*args,**args1)
    
def forward_layer(Ta,Tb,layer:HOTRGLayer,use_checkpoint=False)->torch.Tensor:
    #_forward_layer={4:_forward_layer_2D,6:_forward_layer_3D}[len(Ta.shape)]
    return _checkpoint(_forward_layer,[Ta,Tb],{'layer':layer},use_checkpoint=use_checkpoint)

def gauge_invariant_norm(T):
    contract_path={4:'iijj->',5:'iijjk->k',6:'iijjkk->',7:'iijjkkl->l'}[len(T.shape)]
    norm=contract(contract_path,T).norm()
    if norm<1e-6:#fallback
        norm=T.norm()
    #norm=T.norm()
    #print(norm)
    return norm
    
def to_unitary(g):
    u,s,vh=svd(g)
    return u@vh
    
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
        #with BypassGilt(False,True):
        T2=forward_layer(T,T_op,layer=layer,use_checkpoint=use_checkpoint)
        #with BypassGilt(True,False):
        T3=forward_layer(T_op,T,layer=layer,use_checkpoint=use_checkpoint)
        T3=-T3 if (checkerboard and ilayer<spacial_dim) else T3
        T,T_op=T1,(T2+T3)/2
        if return_layers:
            Ts.append(T);T_ops.append(T_op);logTotals.append(logTotal)
    return (Ts,T_ops,logTotals) if return_layers else (T,T_op,logTotal)

    
def forward_observalbe_tensor_moments(T0_moments:'list[torch.Tensor]',layers:'list[HOTRGLayer]',
        checkerboard=False,use_checkpoint=False,return_layers=False,cached_Ts=None):
    print('WARNING NOT TESTED')
    # -T'[OO]- = -T[OO]-T[1]- + 2 -T[O]-T[O]- + -T[1]-T[OO]-      
    spacial_dim=len(T0_moments[0].shape)//2
    logTotal=0
    Tms=T0_moments.copy()
    if return_layers:
        Tmss,logTotals=[Tms],[logTotal]
    for ilayer,layer in tqdm(list(enumerate(layers)),leave=False):
        norm=gauge_invariant_norm(Tms[0])
        logTotal=2*(logTotal+norm.log())
        Tms=[x/norm for x in Tms]
        Tms1=[torch.zeros_like(Tms[0])]*len(Tms)
        for a in range(len(Tms)):
            for b in range(len(Tms)):
                if a+b<len(Tms1):
                    if a+b==0 and cached_Ts:
                        Tms1[a+b]=cached_Ts[ilayer+1]
                    else:
                        Tms1[a+b]=math.comb(a+b,b)\
                            *forward_layer(Tms[a],Tms[b],layer=layer,use_checkpoint=use_checkpoint)
        Tms=Tms1
        if return_layers:
            Tmss.append(Tms);logTotals.append(logTotal)
            return (Tmss,logTotals) if return_layers else (Tms,logTotal)
    
def get_lattice_size(nLayers,spacial_dim):
    return tuple(2**(nLayers//spacial_dim+(1 if i<nLayers%spacial_dim else 0)) for i in range(spacial_dim))

def get_dist_torus_2D(x,y,lattice_size):
    d1=x**2+y**2
    d2=(lattice_size[0]-x)**2+y**2
    d3=x**2+(lattice_size[1]-y)**2
    d4=(lattice_size[0]-x)**2+(lattice_size[1]-y)**2
    return functools.reduce(np.minimum,[d1,d2,d3,d4])**.5

def forward_coordinate(coords):
    return coords[1:]+(coords[0]//2,)




def forward_observable_tensors(T0,T0_ops:list,positions:'list[tuple[int]]',
        layers:'list[HOTRGLayer]',checkerboard=False,use_checkpoint=False,cached_Ts=None,user_tqdm=True):
    spacial_dim=len(T0.shape)//2
    nLayers=len(layers)
    lattice_size=get_lattice_size(nLayers,spacial_dim=spacial_dim)
    assert all(isinstance(c,int) and 0<=c and c<s for coords in positions for c,s in zip(coords,lattice_size)),"coordinates must be integers in the range [0,lattice_size)\n"+str(positions)+" "+str(lattice_size)
    assert all(positions[i]!=positions[j] for i,j in itt.combinations(range(len(positions)),2))
    assert len(positions)==len(T0_ops)
    T,T_ops,logTotal=T0,T0_ops.copy(),0
    _tqdm=tqdm if user_tqdm else lambda x,leave:x
    for ilayer,layer in _tqdm(list(enumerate(layers)),leave=False):
        norm=gauge_invariant_norm(T)
        logTotal=2*(logTotal+norm.log())
        T,T_ops=T/norm,[T_op/norm for T_op in T_ops]
        # check if any two points are going to merge
        iRemoved=[]
        T_ops_new,positions_new=[],[]
        for i,j in itt.combinations(range(len(positions)),2):
            if forward_coordinate(positions[i])==forward_coordinate(positions[j]):
                i,j=(i,j) if positions[i][0]%2==0 else (j,i)
                #print(positions[i],positions[j])
                assert positions[i][0]%2==0 and positions[j][0]%2==1
                #with BypassGilt(True,True):
                T_op_new=forward_layer(T_ops[i],T_ops[j],layer,use_checkpoint=use_checkpoint)
                if checkerboard and ilayer<spacial_dim:
                    T_op_new=-T_op_new
                T_ops_new.append(T_op_new)
                positions_new.append(forward_coordinate(positions[i]))
                assert (not i in iRemoved) and (not j in iRemoved)
                iRemoved.extend([i,j])
        # forward other points with T
        for i in range(len(positions)):
            if i not in iRemoved:
                if positions[i][0]%2==0:
                    #with BypassGilt(False,True):
                    T_op_new=forward_layer(T_ops[i],T,layer,use_checkpoint=use_checkpoint)
                else:
                    #with BypassGilt(True,False):
                    T_op_new=forward_layer(T,T_ops[i],layer,use_checkpoint=use_checkpoint)
                    if checkerboard and ilayer<spacial_dim:
                        T_op_new=-T_op_new
                T_ops_new.append(T_op_new)
                positions_new.append(forward_coordinate(positions[i]))
        # forward T
        if cached_Ts:
            T_new=cached_Ts[ilayer+1]
        else:
            T_new=forward_layer(T,T,layer=layer,use_checkpoint=use_checkpoint)
        T,T_ops,positions=T_new,T_ops_new,positions_new
    if len(positions)==0:
        return T,T,logTotal
    else:
        assert len(positions)==1
        return T,T_ops[0],logTotal

    
    
def trace_tensor(T):
    eq={4:'aabb->',6:'aabbcc->'}[len(T.shape)]
    return contract(eq,T)

def trace_two_tensors(T,T1=None):
    T1=T if T1 is None else T1
    eq={4:'abcc,badd->',6:'abccdd,baeeff->'}[len(T.shape)]
    return contract(eq,T,T)
 
def reflect_tensor_axis(T):
    Ai=[2*i+j for i in range(len(T.shape)//2) for j in range(2)]
    Bi=[2*i+1-j for i in range(len(T.shape)//2) for j in range(2)]
    return contract(T,Ai,Bi)
    
def permute_tensor_axis(T):
    Ai=[*range(len(T.shape))]
    Bi=Ai[2:]+Ai[:2]
    return contract(T,Ai,Bi)
#==================

import importlib
import HOSVD,GILT,fix_gauge
importlib.reload(HOSVD)
importlib.reload(GILT)
importlib.reload(fix_gauge)

from HOSVD import HOSVD_layer
from GILT import GILT_HOTRG,GILT_options
from fix_gauge import minimal_canonical_form,fix_unitary_gauge,fix_phase,MCF_options
    


def HOTRG_layer(T1,T2,max_dim,dimR=None,options:dict={},Tref=None):
    T1old,T2old=T1,T2
    gilt_options=GILT_options(**{k[5:]:v for k,v in options.items() if k[:5]=='gilt_'})
    mcf_options=MCF_options(**{k[4:]:v for k,v in options.items() if k[:4]=='mcf_'})
    if options.get('gilt_enabled',False):
        assert not dimR
        T1,T2,gg=GILT_HOTRG(T1,T2,options=gilt_options)
    else:
        gg=None
    
    Tn,layer=HOSVD_layer(T1,T2,max_dim=max_dim,dimR=dimR)
    layer.gg=gg

    if options.get('gilt_make_isometric',False):
        layer.make_gg_isometric()
        Tn= forward_layer(T1old,T2old,layer)
    
    if options.get('mcf_enabled',False):
        assert not dimR
        Tn,hh=minimal_canonical_form(Tn,options=mcf_options)
        if Tref is not None and Tn.shape==Tref.shape:
            Tn,hh1=fix_phase(Tn,Tref)
            hh=[h1@h for h1,h in zip(hh1,hh)]
            if options.get('mcf_enabled_unitary',False):
                Tn,hh1=fix_unitary_gauge(Tn,Tref)
                hh=[h1@h for h1,h in zip(hh1,hh)]
        hh=hh[-2:]+hh[:-2]
    else:
        hh=None
    layer.hh=hh
    if options.get('hotrg_sanity_check'):
        Tn1= forward_layer(T1old,T2old,layer)
        assert (Tn-Tn1).abs().max()<1e-6

    return Tn,layer
    
    
def HOTRG_layers(T0,max_dim,nLayers,
        dimR:"tuple[tuple[int]]"=None,
        options:dict={},
        return_tensors=False):    
    print('Generating HOTRG layers')
    spacial_dim=len(T0.shape)//2
    stride=spacial_dim
    T,logTotal=T0,0
    Ts,logTotals=[T],[0]
    layers=[]
    for iLayer in tqdm(list(range(nLayers)),leave=False):
        norm=gauge_invariant_norm(T)
        T=T/norm
        logTotal=2*(logTotal+norm.log())
        
        Tref=Ts[iLayer+1-stride] if iLayer+1-stride>=0 else None
        Told=T
        T,layer=HOTRG_layer(T,T,max_dim=max_dim,dimR=dimR,Tref=Tref,options=options)
        dimR=layer.dimR_next
        
        if options.get('hotrg_sanity_check',False):
            assert ((forward_layer(Told,Told,layer)-T).norm()/T.norm()<=options.get('hotrg_sanity_check_tol',1e-7))

        layers.append(layer)
        Ts.append(T);logTotals.append(logTotal)
            
    print('HOTRG layers generated')
    return (layers,Ts,logTotals) if return_tensors else layers

    
    
#==================
'''
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
    for iLayer in range(nLayers):
        ww=[]
        dimRn=(dimR[-1],)
        for i in range(1,spacial_dim):
            w,dimR0,dimR1=get_w_random(dimR[i][0],dimR[i][1],max_dim)
            ww.append(w)
            dimRn.append([dimR0,dimR1])
            dimRn+=((dimR0,dimR1),)
        Tshape=(sum(x) for x in dimR) #todo not sure
        layers.append(HOTRGLayer(T0shape=Tshape,ww=ww,dimR=dimR,dimR_next=dimRn))
        dimR=dimRn
    return layers


    
'''







'''
def forward_two_observable_tensors(T0,T0_op1,T0_op2,coords:"tuple[int]",layers:'list[HOTRGLayer]',checkerboard=False,use_checkpoint=False,cached_Ts=None):
    spacial_dim=len(T0.shape)//2
    nLayers=len(layers)
    lattice_size=get_lattice_size(nLayers,spacial_dim=spacial_dim)
    #print(coords,lattice_size)
    assert all(isinstance(c,int) and 0<=c and c<s for c,s in zip(coords,lattice_size))
    assert not all(c==0 for c in coords)
    T,T_op1,T_op2,T_op12,logTotal=T0,T0_op1,T0_op2,None,0
    for ilayer,layer in tqdm(list(enumerate(layers)),leave=False):
        norm=gauge_invariant_norm(T)
        logTotal=2*(logTotal+norm.log())
        T,T_op1,T_op2,T_op12=(t/norm if t is not None else None for t in (T,T_op1,T_op2,T_op12))
        #Evolve vacuum T
        if cached_Ts:
            T1=cached_Ts[ilayer+1]
        else:
            T1=forward_layer(T,T,layer=layer,use_checkpoint=use_checkpoint)
        #Evolve defected T depends on whether the two defects are in the same coarse-grained block
        #print(coords)
        if coords==(0,)*spacial_dim:
            T2=forward_layer(T_op12,T,layer=layer,use_checkpoint=use_checkpoint)
            T,T_op12=T1,T2
        elif coords==(1,)+(0,)*(spacial_dim-1):
            T2=forward_layer(T_op1,T_op2,layer=layer,use_checkpoint=use_checkpoint)
            T2=-T2 if checkerboard and ilayer<spacial_dim else T2
            T,T_op12,T_op1,T_op2=T1,T2,None,None
        else:
            c=coords[0]%2
            T2=forward_layer(T_op1,T,layer=layer,use_checkpoint=use_checkpoint)
            if c==0:
                T3=forward_layer(T_op2,T,layer=layer,use_checkpoint=use_checkpoint)
            elif c==1:
                T3=forward_layer(T,T_op2,layer=layer,use_checkpoint=use_checkpoint)
                T3=-T3 if checkerboard and ilayer<spacial_dim else T3
            T,T_op1,T_op2=T1,T2,T3
        coords=forward_coordinate(coords)
    assert coords==(0,)*spacial_dim
    return T,T_op12,logTotal





'''