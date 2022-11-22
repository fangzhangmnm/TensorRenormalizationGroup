import torch
from opt_einsum import contract_path
from opt_einsum import contract
from safe_svd import svd,sqrt
from dataclasses import dataclass
from tqdm.auto import tqdm
from math import prod
import numpy as np
from ScalingDimensions import get_entanglement_entropy
from TNR import svd_tensor,qr_tensor


# https://arxiv.org/pdf/2104.08283.pdf
def fast_disentangling(A:torch.Tensor):
    # i   j
    # uuuuu
    # I   J   0  1
    # AAAAA   AAAA
    # a   b   2  3
    dimi,dimj,dima,dimb=A.shape
    print(dimi,dimj,dima,dimb)
    assert dimi<=dima and dimj<=dimb
    r=torch.randn(dimi,dimj)
    rA=contract('IJ,IJab->ab',r,A) # a*,b*
    u,s,vh=svd(rA)
    aL,aR=u[:,0].conj(),vh[0,:]# a, b
    AaL=contract('IJab,b->IJa',A,aR).reshape(dimi*dimj,dima) # IJa*
    AaR=contract('IJab,a->IJb',A,aL).reshape(dimi*dimj,dimb) # IJb*
    VLh=svd(AaL)[2][:dimi,:] # ia*
    VRh=svd(AaR)[2][:dimj,:] # jb*
    Bh=contract('ia,jb,IJab->ijIJ',VLh,VRh,A.conj()) # ijI*J*
    U=qr_tensor(Bh,(0,1),(2,3))[0] #ijI*j*
    return U
    