from opt_einsum import contract
import torch
import numpy as np

from HOSVD import _HOSVD_layer_3D,gauge_invariant_norm
def _XTRG_layer_3D(Ta,Tb,max_dim):
    Tn,layer=_HOSVD_layer_3D(Ta,Tb,max_dim=max_dim)
    Tn=contract('abcdio->ioabcd',Tn)
    return Tn,layer

def XTRG_3D_defect(T0,T0_op,max_dim,nLayers):
    T,T_op,logTotal=T0,T0_op,0
    for i in tqdm(range(nLayers),leave=False):
        norm=gauge_invariant_norm(T)
        logTotal=2*(logTotal+norm.log())
        T,T_op=T/norm,T_op/norm
        T_op=_XTRG_layer_3D(T_op,T,max_dim=max_dim)[0]
        T=_XTRG_layer_3D(T,T,max_dim=max_dim)[0]
    return T,T_op