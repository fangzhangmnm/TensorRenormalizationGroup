# Scaling dimensions from linearized tensor renormalization group transformations
# https://arxiv.org/pdf/2102.08136.pdf

# Renormalization of tensor networks using graph independent local truncations
# https://arxiv.org/pdf/1709.07460.pdf
# https://github.com/Gilt-TNR/Gilt-TNR
# https://github.com/Gilt-TNR/Gilt-TNR/blob/master/GiltTNR3D.py

import torch
from tqdm.auto import tqdm
import numpy as np
from opt_einsum import contract
import itertools as itt
from dataclasses import dataclass

from safe_svd import svd,sqrt # TODO is it necessary???

# Basic idea:

# Given a subgraph and a specific leg in that subgraph, the environment tensor E = break the leg, the mapping from the two ends of the broken leg to the external legs of the subgraph
# We can insert arbitrary projector R to that leg if that's in the kernel of E
# We chose R_ab=t_i' U_abi, where E_ab,external=U_ab,i t_i Vh_i,external, and only change t_i which are small: t_i'=t_i**2/(t_i**2+gilt_eps**2), we also do gilt_nIter iterations

# To see how it works
# E factorizes as E(UV and IR entanglement to outside) \otimes I(UV entanglement inside)
# TODO

#svd,sqrt=torch.linalg.svd,torch.sqrt

if torch.get_default_dtype() not in {torch.float64}:
    print('[GILT] Warning! float32 is not precise enough, leads to bad RG behavior')


def Split_Matrix(Q):
    u,s,vh=svd(Q) # is it necessary to split?
    s=sqrt(s).diag()
    return u@s,s@vh

@dataclass
class GILT_options:
    enabled:bool=True
    eps:float=1e-6
    nIter:int=2
    split_insertion:bool=True
    TRG_method:str='BA'
    fix_gauge:bool=True
    cube_apply_inner:bool=True
    
def GILT_getuvh(EEh,options:GILT_options=GILT_options()):
    
    d=EEh.shape[0]
    uu,vvh=torch.eye(d),torch.eye(d)
    for _iter in range(options.nIter):
        if _iter==0:
            U,S,_=svd(EEh.reshape(d**2,d**2))
        else:
            uvUS=contract('aA,Bb,abc,c->ABc',u,vh,U,S).reshape(d**2,d**2)
            U,S,_=svd(uvUS)
        U=U.reshape(d,d,d**2)
        t=contract('aac->c',U)
        Sn=S/torch.max(S)
        t=t*(Sn**2/(Sn**2+options.eps**2))
        Q=contract('abc,c->ab',U,t)
        if options.split_insertion:
            #from utils import show_matrix,show_hist,plt
            #show_matrix(Q);plt.show()
            #show_matrix(Q@Q);plt.show()
            #show_hist(svd(Q)[1].numpy());plt.show()
            
            u,vh=Split_Matrix(Q)
        else:
            # not make sense, introduces numerical error!
            u,vh=Q,torch.eye(d)
        uu,vvh=uu@u,vh@vvh
    return uu,vvh
    
def GILT_getEEh(As,Ais:"list[list[str]]"):
    def process(edgeid,legid,tensorid,replicaid):
        if edgeid is None:
            #contract between corresponding replicas
            return 'T'+str(tensorid)+'L'+str(legid)
        else:
            #internal legs
            return 'R'+str(replicaid)+'E'+str(edgeid)
    R1Ais=[[process(edgeid,legid,tensorid,0) for legid,edgeid in enumerate(Ai)]for tensorid,Ai in enumerate(Ais)]
    R2Ais=[[process(edgeid,legid,tensorid,1) for legid,edgeid in enumerate(Ai)]for tensorid,Ai in enumerate(Ais)]
    AAis=[list(filter(lambda x:x[0]=='R',R1Ai+R2Ai)) for R1Ai,R2Ai in zip(R1Ais,R2Ais)]
    Ti=['R0Eu','R0Ev','R1Eu','R1Ev']
    AAs=[contract(A,R1Ai,A,R2Ai,AAi) for A,R1Ai,R2Ai,AAi in zip(As,R1Ais,R2Ais,AAis)]
    #print(AAis)
    T=contract(*itt.chain(*zip(AAs,AAis)),Ti)
    #print(R1Ais);print(R2Ais);print(AAis);print(Ti)
    #for AA,AAi in zip(AAs,AAis):
    #    print(AA.shape)
    #    print(AAi)
    #print(T.shape)
    #print(Ti)
    return T
    
    
#======================================================




#def GILT_Square_one(As,options:GILT_options=GILT_options()):
#    # A1- -A2      A1u vA2    0
#    #  | O |   -->  | U |    2A3
#    # A3---A4      A3---A4    1
#    A1i=[None,'13',None,'u']
#    A2i=[None,'24','v',None]
#    A3i=['13',None,None,'34']
#    A4i=['24',None,'34',None]
#    EEh=GILT_getEEh(As,[A1i,A2i,A3i,A4i])
#    u,vh=GILT_getuvh(EEh,options=options)
#    assert not u.isnan().any() and not vh.isnan().any()
#    return u,vh

def GILT_Square_one(As,leg,options:GILT_options=GILT_options()):
    # leg: 12 for example
    # A1- -A2      A1u vA2    0
    #  | O |   -->  | U |    2A3
    # A3---A4      A3---A4    1
    Ais=[
        [None,'13',None,'12'],
        [None,'24','12',None],
        ['13',None,None,'34'],
        ['24',None,'34',None],
    ]
    assert leg in {'12','34','13','24'}
    flag=False
    for Ai in Ais:
        if leg in Ai:
            if not flag:
                Ai[Ai.index(leg)]='u'
                flag=True
            else:
                Ai[Ai.index(leg)]='v'
    EEh=GILT_getEEh(As,Ais)
    u,vh=GILT_getuvh(EEh,options=options)
    assert not u.isnan().any() and not vh.isnan().any()
    return u,vh
    
def GILT_Cube_one(As,leg,options:GILT_options=GILT_options()):
    # leg: 12 for example
    #   A5+------+A6
    #     |`.    |`.            0
    #     | A1+-u  v-+A2      5`|  
    #     |   |  |   |        2-o-3  
    #   A7+---|--+A8 |          |`4
    #      `. |   `. |          1
    #       A3+------+A4
    Ais=[
        [None,'13',None,'12',None,'15'],
        [None,'24','12',None,None,'26'],
        ['13',None,None,'34',None,'37'],
        ['24',None,'34',None,None,'48'],
        [None,'57',None,'56','15',None],
        [None,'68','56',None,'26',None],
        ['57',None,None,'78','37',None],
        ['68',None,'78',None,'48',None],
    ]
    assert leg in {'12','34','56','78','13','24','57','68','15','26','37','48'}
    flag=False
    for Ai in Ais:
        if leg in Ai:
            if not flag:
                Ai[Ai.index(leg)]='u'
                flag=True
            else:
                Ai[Ai.index(leg)]='v'
    EEh=GILT_getEEh(As,Ais)
    u,vh=GILT_getuvh(EEh,options=options)
    assert not u.isnan().any() and not vh.isnan().any()
    return u,vh
    
    
    
from HOTRGZ2 import HOTRG_layer
import importlib
import fix_gauge
importlib.reload(fix_gauge)
from fix_gauge import fix_gauge_2D

def GILT_HOTRG2D(T1,T2,options:GILT_options=GILT_options()):
    #      O  | O                 
    #    /v1-T1-u1\       0       2
    #  -w     |    w-    2T3  -> 0T'1  
    #    \v2-T2-u2/       1       3
    #      O  | O 
    
    #Y1,Y2=T1,T2

    u1,vh1=GILT_Square_one([T2,T2,T1,T1],leg='34',options=options)
    T1=contract('ijkl,Kk,Ll->ijKL',T1,vh1,u1.T)

    u2,vh2=GILT_Square_one([T2,T2,T1,T1],leg='12',options=options)
    T2=contract('ijkl,Kk,Ll->ijKL',T2,vh2,u2.T)
    
    gg=[[vh1,u1.T],[vh2,u2.T]]
    #Y1=contract('ijkl,Kk,Ll->ijKL',Y1,*gg[0])
    #Y2=contract('ijkl,Kk,Ll->ijKL',Y2,*gg[1])
    #print((T1-Y1).norm(),(T2-Y2).norm())
    return T1,T2,gg
    
def GILT_HOTRG3D(T1,T2,options:GILT_options=GILT_options()):
    #       g4|                         5--6
    #    /g1-T1-g2\      50      34     |1--2
    #  -w   g8|g3  w-    2T3  -> 0T'1   7| 8|
    #    \g5-T2-g6/       14      52     3--4
    #         |g7 
    
    Y1,Y2=T1,T2
    
    T21s=[T2,T2,T2,T2,T1,T1,T1,T1]
    T12s=[T1,T1,T1,T1,T2,T2,T2,T2]
    contract23='ijklmn,Kk,Ll->ijKLmn'
    contract45='ijklmn,Mm,Nn->ijklMN'
    
    u,vh=GILT_Cube_one(T21s,leg='34',options=options)
    T1,g1,g2=contract(contract23,T1,vh,u.T),vh,u.T
    u,vh=GILT_Cube_one(T21s,leg='78',options=options)
    T1,g1,g2=contract(contract23,T1,vh,u.T),vh@g1,u.T@g2
    if options.cube_apply_inner:
        u,vh=GILT_Cube_one(T12s,leg='12',options=options)
        T1,g1,g2=contract(contract23,T1,vh,u.T),vh@g1,u.T@g2
        u,vh=GILT_Cube_one(T12s,leg='56',options=options)
        T1,g1,g2=contract(contract23,T1,vh,u.T),vh@g1,u.T@g2
    
    u,vh=GILT_Cube_one(T21s,leg='37',options=options)
    T1,g3,g4=contract(contract45,T1,vh,u.T),vh,u.T
    u,vh=GILT_Cube_one(T21s,leg='48',options=options)
    T1,g3,g4=contract(contract45,T1,vh,u.T),vh@g3,u.T@g4
    if options.cube_apply_inner:
        u,vh=GILT_Cube_one(T12s,leg='15',options=options)
        T1,g3,g4=contract(contract45,T1,vh,u.T),vh@g3,u.T@g4
        u,vh=GILT_Cube_one(T12s,leg='26',options=options)
        T1,g3,g4=contract(contract45,T1,vh,u.T),vh@g3,u.T@g4
    
    u,vh=GILT_Cube_one(T21s,leg='12',options=options)
    T2,g5,g6=contract(contract23,T2,vh,u.T),vh,u.T
    u,vh=GILT_Cube_one(T21s,leg='56',options=options)
    T2,g5,g6=contract(contract23,T2,vh,u.T),vh@g5,u.T@g6
    if options.cube_apply_inner:
        u,vh=GILT_Cube_one(T12s,leg='34',options=options)
        T2,g5,g6=contract(contract23,T2,vh,u.T),vh@g5,u.T@g6
        u,vh=GILT_Cube_one(T12s,leg='78',options=options)
        T2,g5,g6=contract(contract23,T2,vh,u.T),vh@g5,u.T@g6
    
    u,vh=GILT_Cube_one(T21s,leg='15',options=options)
    T2,g7,g8=contract(contract45,T2,vh,u.T),vh,u.T
    u,vh=GILT_Cube_one(T21s,leg='26',options=options)
    T2,g7,g8=contract(contract45,T2,vh,u.T),vh@g7,u.T@g8
    if options.cube_apply_inner:
        u,vh=GILT_Cube_one(T12s,leg='37',options=options)
        T2,g7,g8=contract(contract45,T2,vh,u.T),vh@g7,u.T@g8
        u,vh=GILT_Cube_one(T12s,leg='48',options=options)
        T2,g7,g8=contract(contract45,T2,vh,u.T),vh@g7,u.T@g8
    
    gg=[[g1,g2,g3,g4],[g5,g6,g7,g8]]
    
    Y1=contract('ijklmn,Kk,Ll,Mm,Nn->ijKLMN',Y1,*gg[0])
    Y2=contract('ijklmn,Kk,Ll,Mm,Nn->ijKLMN',Y2,*gg[1])
    print((Y1-T1).norm(),(Y2-T2).norm())
    
    return T1,T2,gg
    
    
def GILT_HOTRG_layer(T1,T2,max_dim,dimR=None,options:GILT_options=GILT_options()):
    if options.enabled:
        assert not dimR
        spacial_dim=len(T1.shape)//2
        _GILT_HOTRG={2:GILT_HOTRG2D,3:GILT_HOTRG3D}[spacial_dim]
        T1,T2,gg=_GILT_HOTRG(T1,T2,options=options)
        Tn,layer=HOTRG_layer(T1,T2,max_dim=max_dim,dimR=dimR)
        layer.gg=gg
        return Tn,layer
    else:
        return HOTRG_layer(T1,T2,max_dim=max_dim,dimR=dimR)
    
    

#============= GILT On TRG=============

from TRG import TRG_AB
from HOTRGZ2 import gauge_invariant_norm

import inspect


def GILT_SquareA(A,options:GILT_options=GILT_options()):
    # Not good precision. Why?
    # A- -A      vAu vAu 
    # | O |  -->  | O |
    # A---A      vAu-vAu
    for i in range(4):
        u,vh=GILT_Square_one([A,A,A,A],leg='12')
        A=contract('abcd,Cc,dD->abCD',A,vh,u)
        A=contract('abcd->dcab',A)
    return A

def GILT_SquareABCD(A,B,C,D,options:GILT_options=GILT_options()):
    # A- -B       Au vB     0
    # | O |  -->  | O |    2A3
    # C---D       C---D     1
    for i in range(2):
        for i in range(4):
            u,vh=GILT_Square_one([A,B,C,D],leg='12')
            A,B=contract('abcd,dD->abcD',A,u),contract('abcd,Cc->abCd',B,vh)
            CCW='abcd->dcab'
            A,B,C,D=contract(CCW,B),contract(CCW,D),contract(CCW,A),contract(CCW,C)#rotate CCW
        A,B,C,D=B,A,D,C
    return A,B,C,D

def GILT_SquareAB(A,B,options:GILT_options=GILT_options()):
    # A---B       Au vB     0
    # | O |  -->  | O |    2A3
    # B---A      vB---Au    1
    for i in range(4):
        u,vh=GILT_Square_one([A,B,B,A],leg='12')
        A,B=contract('abcd,dD->abcD',A,u),contract('abcd,Cc->abCd',B,vh)
        A,B=contract('abcd->dcab',B),contract('abcd->dcab',A)#rotate 
    return A,B

def evolve_TRG_GILT_2D(T0,nLayers,max_dim,return_layers=False,options:GILT_options=GILT_options()):
    T,logTotal=T0,0
    if return_layers: 
        Ts,logTotals=[T],[logTotal]
    for i in tqdm(range(nLayers),leave=False):
        norm=gauge_invariant_norm(T)
        T=T/norm
        logTotal=2*(logTotal+norm.log())
        if options.TRG_method=='A':
            A=GILT_SquareA(T,options=options)
            T=TRG_AB(A,A,max_dim)
        elif options.TRG_method=='AB':
            A,B=GILT_SquareAB(T,T,options=options)
            T=TRG_AB(A,B,max_dim)
        elif options.TRG_method=='BA':
            A,B=GILT_SquareAB(T,T,options=options)
            T=TRG_AB(B,A,max_dim)
        elif options.TRG_method=='BAAB':
            A,B=GILT_SquareAB(T,T,options=options)
            A,B=GILT_SquareAB(B,A,options=options)
            T=TRG_AB(A,B,max_dim)
        elif options.TRG_method=='BABA':
            A,B=GILT_SquareAB(T,T,options=options) 
            A,B=GILT_SquareAB(B,A,options=options)
            T=TRG_AB(B,A,max_dim)
        elif options.TRG_method=='ABCD':
            assert False
        else:
            assert False 
        
        if options.fix_gauge and T.shape==Ts[-1].shape:
            T,uu=fix_gauge_2D(T,Ts[-1])
        if return_layers: 
            Ts.append(T);logTotals.append(logTotal)
    if return_layers:
        return Ts,logTotals
    return T,logTotal
    


