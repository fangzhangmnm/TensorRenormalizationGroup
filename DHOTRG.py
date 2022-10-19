# Second Renormalization of Tensor-Network States
# https://arxiv.org/pdf/0809.0182.pdf
# Automatic Differentiation for Second Renormalization of Tensor Networks
# https://arxiv.org/pdf/1912.02780.pdf


import torch
from torch.utils.checkpoint import checkpoint
import numpy as np
from opt_einsum import contract
from tqdm.auto import tqdm
from HOTRGZ2 import forward_layer,trace_tensor,HOTRG_layer

def get_isometry_from_environment_tensor(E,dimR,dimRnn,Z2_projection=False):
    if Z2_projection:
        dimRn=RepDim(dimR[0],dimR[1],dimR[0],dimR[1])
        w=torch.zeros(E.shape)
        
        norms0=torch.sqrt(E[:dimRnn[0],:dimRn[0]].norm()**2+E[dimRnn[0]:,dimRn[0]:].norm()**2)
        norms1=torch.sqrt(E[:dimRnn[0],dimRn[0]:].norm()**2+E[dimRnn[0]:,:dimRn[0]].norm()**2)
        assert norms1<1e-6*norms0
        
        U,S,Vh=torch.linalg.svd(E[:dimRnn[0],:dimRn[0]],full_matrices=False)
        w[:dimRnn[0],:dimRn[0]]=U@Vh
        U,S,Vh=torch.linalg.svd(E[dimRnn[0]:,dimRn[0]:],full_matrices=False)
        w[dimRnn[0]:,dimRn[0]:]=U@Vh
        return w
    else:
        U,S,Vh=torch.linalg.svd(E,full_matrices=False)
        assert U.shape[0]==U.shape[1]
        w=U@Vh
        return w

def get_isometry_difference(a,b):
    if(a.shape[0]<a.shape[1]):
        return torch.norm(contract('iA,jA,jB,kB->ik',a,b,b,a)-torch.eye(a.shape[0]))
    else:
        return torch.norm(contract('Ai,Aj,Bj,Bk->ik',a,b,b,a)-torch.eye(a.shape[1]))
    
        
def calc_environment_tensor(T0,isometries,dimRs,layer,axis,cache=None,return_cache=False):
    torch.cuda.empty_cache()
    
    isometries[layer][axis].requires_grad_(True)
    isometries[layer][axis].grad=None
    
    if cache is None:
        T,logTotal=T0.detach(),0
        for i in range(layer):
            T=forward_layer(T,T,isometries[i],dimRs[i])
            norm=torch.linalg.norm(T)
            T=T/norm
            logTotal=2*logTotal+torch.log(norm)
            cache=[T.detach(),logTotal.detach()]
    else:
        T,logTotal=cache[0].detach(),cache[1].detach()
    
    for i in range(layer,len(isometries)):
        T=forward_layer(T,T,isometries[i],dimRs[i],use_checkpoint=True)
        norm=torch.linalg.norm(T)
        T=T/norm
        logTotal=2*logTotal+torch.log(norm)
    logZ=(logTotal+torch.log(trace_tensor(T)))/2.**len(isometries)
    logZ.backward()
    del T,logTotal,norm,logZ

    E=isometries[layer][axis].grad.detach()
    isometries[layer][axis].requires_grad_(False)
    isometries[layer][axis].grad=None
    
    torch.cuda.empty_cache()
    
    if return_cache:
        return E,cache
    else:
        return E
    
def _toN(x):
    x=x.detach().cpu().numpy()
    if len(x.shape)==0:
        x=x.item()
    return x
        
def update_single_layer(T0,isometries,dimRs,layer,axis,inner_iter=1,Z2_projection=False):
    cache=None
    wOld=isometries[layer][axis].detach()
    wDiff_inner=[]
    if layer<len(isometries)-1:
        for i in range(inner_iter):
            E,cache=calc_environment_tensor(T0,isometries,dimRs,layer,axis,cache=cache,return_cache=True)
            with torch.no_grad():
                wNew=get_isometry_from_environment_tensor(E,dimR=dimRs[layer][axis+1],dimRnn=dimRs[layer+1][axis],\
                                                          Z2_projection=Z2_projection)
                wDiff_inner.append(_toN(get_isometry_difference(wNew,isometries[layer][axis])))
                isometries[layer][axis].data=wNew
    elif layer==len(isometries)-1:
        T=T0
        for i in range(layer-1):
            T=forward_layer(T,T,isometries[i],dimRs[i])
            T=T/torch.linalg.norm(T)
        ww=HOTRG_layer(T,max_dim=isometries[layer][axis].shape[0],dimR=dimRs[layer])[1]
        isometries[layer][axis].data=ww[axis]
    wDiff=_toN(get_isometry_difference(isometries[layer][axis],wOld))
    return wDiff,wDiff_inner

def update_isometries(T0,isometries,dimRs,inner_iter=1,Z2_projection=False):
    wDiffs=[[0 for w in ww]for ww in isometries]
    wDiff_inners=[[[] for w in ww]for ww in isometries]
    nLayers=len(isometries)
    spacial_dim=len(T0.shape)//2
    #sequence=[*range(nLayers-1,-1,-1)]+[*range(nLayers)]
    sequence=[*range(nLayers)]
    sequence=[(i,j) for i in sequence for j in range(spacial_dim-1)]
    for i,j in tqdm(sequence,leave=False):
        shape=isometries[i][j].shape
        if shape[0]<shape[1]:
            wDiffs[i][j],wDiff_inners[i][j]=update_single_layer(T0,isometries,dimRs,i,j,inner_iter=inner_iter,Z2_projection=Z2_projection)
    return wDiffs,wDiff_inners
            
