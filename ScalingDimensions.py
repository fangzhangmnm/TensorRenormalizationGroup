import torch
import numpy as np
from tqdm.auto import tqdm
from opt_einsum import contract

svd=torch.linalg.svd

from HOTRGZ2 import gauge_invariant_norm

def _toN(t):
    return t.detach().cpu().tolist() if isinstance(t,torch.Tensor) else t

def fix_normalize(T,norms,volume_scaling=2,is_HOTRG=False):
    if not is_HOTRG:
        # evolve(T/norm)=T
        # q T=evolve(q T)=q**scaling * norm**scaling T
        q=norms[-1]**(volume_scaling/(1-volume_scaling))
    else:
        # evolve(...evolve(T/norms[0])/norms[1]...)=T
        # q T=evolve(...evolve(q T)...)
        #    =q**(scaling**dim) * norms[0]**(scaling**dim) *...* norms[-1]**(scaling)
        spacial_dim=len(T.shape)//2
        norms=([1]*spacial_dim+norms)[-spacial_dim:]
        norms=[norms[-1]]+norms[:-1]#why
        q=1
        for axis in range(spacial_dim):
            q=q * norms[axis]**(volume_scaling**(spacial_dim-axis))
        q=q**(1/(1-volume_scaling**spacial_dim))
    return q*T

def get_transfer_matrix_spectrum_2D(T,loop_length:int=2):
    M=T
    for i in range(loop_length-2):# Must preserve edge orders
        M=contract('kKab,lLbc->klKLac',M,T).reshape(M.shape[0]*T.shape[0],M.shape[1]*T.shape[1],M.shape[2],T.shape[3])
    if loop_length>1:
        M=contract('kKab,lLba->klKL',M,T).reshape(M.shape[0]*T.shape[0],M.shape[1]*T.shape[1])
    else:
        M=contract('kKaa->kK',T)
    #s2,_=torch.lobpcg(M@M.T,k=k,method="ortho");s=s2**.5
    u,s,vh=svd(M)
    return u,s,vh


def get_transfer_matrix_spectrum_3D(T,loop_length:'tuple[int]'=(1,1)):
    loop_length=tuple(loop_length)
    if loop_length==(1,1):
        M=contract('ijkkmm->ij',T)
    elif loop_length==(2,1):
        M=contract('ijklmm,IJlknn->iIjJ',T,T).reshape(T.shape[0]*T.shape[0],T.shape[1]*T.shape[1])
    elif loop_length==(1,2):
        M=contract('ijkkmn,IJllnm->iIjJ',T,T).reshape(T.shape[0]*T.shape[0],T.shape[1]*T.shape[1])
    else:
        raise ValueError
    #s2,_=torch.lobpcg(M@M.T,k=k,method="ortho");s=s2**.5
    u,s,vh=svd(M)
    return u,s,vh


#def get_center_charge(spectrum,aspect=1):
#    return torch.log(spectrum[0])*12/((2*torch.pi)/aspect)

#def get_scaling_dimensions(spectrum,aspect=1):
#    return -torch.log(spectrum/spectrum[0])/((2*torch.pi)/aspect)

    
def get_central_charge(spectrum,scaling=np.exp(2*np.pi)):
    return 12*torch.log(spectrum[0])/torch.log(torch.as_tensor(scaling))

def get_scaling_dimensions(spectrum,scaling=np.exp(2*np.pi)):
    return -torch.log(spectrum/spectrum[0])/torch.log(torch.as_tensor(scaling))

def get_entanglement_entropy(spectrum):
    spectrum=spectrum/spectrum.sum()
    logSpectrum=spectrum.log().nan_to_num(nan=0)
    return -(logSpectrum*spectrum).sum()


def get_half_circle_density_matrix(u,loop_length):
    assert loop_length%2==0
    psi=u[:,0]
    psi/=torch.norm(psi)
    dim=int(psi.shape[0]**.5)
    psi=psi.reshape(dim,dim)
    rho=contract('ij,Ij->iI',psi,psi.conj())
    rho/=contract('ii->',rho)
    return rho

import pandas as pd
import matplotlib.pyplot as plt

def show_scaling_dimensions(Ts,loop_length=2,num_scaling_dims=8,volume_scaling=2,is_HOTRG=False,reference_scaling_dimensions=None, reference_center_charge=None):
    curve=[]
    
    def pad(v):
        return np.pad(_toN(v),(0,num_scaling_dims))[:num_scaling_dims]

    spacial_dim=len(Ts[0].shape)//2

    norms=list(map(gauge_invariant_norm,Ts))
    for iLayer,A in tqdm([*enumerate(Ts)]):
        A=fix_normalize(A,is_HOTRG=is_HOTRG,volume_scaling=volume_scaling,norms=norms[:iLayer+1])

        if spacial_dim==2:
            if is_HOTRG:
                aspect=[loop_length,loop_length*2][iLayer%2]
            else:
                aspect=loop_length
            u,s,_=get_transfer_matrix_spectrum_2D(A,loop_length=loop_length)
        elif spacial_dim==3:
            assert loop_length==1
            if is_HOTRG:
                loop_length1=[(1,1),(2,1),(1,1)][iLayer%3]
                aspect=[1,2,2][iLayer%3]
            else:
                raise NotImplementedError
            u,s,_=get_transfer_matrix_spectrum_3D(A,loop_length=loop_length1)

        s=s**aspect
        #print(s[:10])

        center_charge=get_central_charge(s)
        scaling_dimensions=get_scaling_dimensions(s)
        min_entropy=-torch.max(s/s.sum()).log()
        transfer_entropy=get_entanglement_entropy(s)
        
        #if loop_length%2==1:
        #    u,_,_=get_transfer_matrix_spectrum_2D(A,loop_length=loop_length-1)
        #    rho=get_half_circle_density_matrix(u,loop_length-1)
        #else:
        #    rho=get_half_circle_density_matrix(u,loop_length)
        #_,s1,_=svd(rho)
        #bipartite_entropy=get_entanglement_entropy(s1)
        
        
        newRow={'layer':iLayer,
                'center_charge':center_charge,
                'scaling_dimensions':pad(scaling_dimensions),
                'min_entropy':min_entropy,
                'transfer_entropy':transfer_entropy,
                #'bipartite_entropy':bipartite_entropy,
                'eigs':pad(s),
                'norm':norms[iLayer]}
        newRow={k:_toN(v) for k,v in newRow.items()}
                
        curve.append(newRow)

    curve=pd.DataFrame(curve)
    #plt.plot(curve['layer'],curve['norm'],'.-',color='black')
    #plt.xlabel('RG Step')
    #plt.ylabel('norm of tensor')
    #plt.show()
    
    eigs=np.array(curve['eigs'].tolist()).T
    for eig in eigs:
        plt.plot(curve['layer'],eig/eigs[0],'.-',color='black')
    plt.xlabel('RG Step')
    plt.ylabel('eigenvalues of normalized transfer matrix')
    plt.ylim([0,1])
    plt.show()
    
    sdsds=np.array(curve['scaling_dimensions'].tolist()).T
    if reference_scaling_dimensions is not None:
        for sdsd in reference_scaling_dimensions:
            plt.plot(curve['layer'],np.ones_like(curve['layer'])*sdsd,'-',color='lightgrey')
        plt.ylim([0,max(reference_scaling_dimensions)*1.1])
    else:
        plt.ylim([np.average(sdsds[-1])*-.1,np.average(sdsds[-1])*1.5])

    for sdsd in sdsds:
        plt.plot(curve['layer'],sdsd,'.-',color='black')
    plt.xlabel('RG Step')
    plt.ylabel('scaling dimensions')
    plt.show()
    
    if reference_center_charge is not None:
        plt.plot(curve['layer'],np.ones_like(curve['layer'])*reference_center_charge,'-',color='lightgrey')
        plt.ylim([0,reference_center_charge*2])
    else:
        plt.ylim([np.average(curve['center_charge'])*-.1,np.average(curve['center_charge'])*1.5])
    plt.plot(curve['layer'],curve['center_charge'],'.-',color='black')
    plt.xlabel('RG Step')
    plt.ylabel('central charge')
    plt.show()
    
    for item in ['min_entropy','transfer_entropy']:
        break
        plt.plot(curve['layer'],curve[item],'.-',color='black')
        plt.xlabel('RG Step')
        plt.ylabel(item)
        plt.show()
    
    return curve

def NWSE(T):
    return contract('nswe->nwse',T).reshape(T.shape[0]*T.shape[2],-1)
def NESW(T):
    return contract('nswe->nesw',T).reshape(T.shape[0]*T.shape[3],-1)

def effective_rank(M):
    assert len(M.shape)==2
    u,s,vh=svd(M)
    s=s[s>0]
    p=s/torch.sum(s)
    entropy=-torch.sum(p*torch.log(p))
    return torch.exp(entropy)

def show_effective_rank(Ts):
    curve=[]

    for i,A in tqdm([*enumerate(Ts)]):
        _,s,_=torch.linalg.svd(NWSE(A))
        s=s/s[0]
        s=np.array(_toN(s))
        if(s.shape[0]<30):
            s=np.pad(s,(0,30-s.shape[0]))
        else:
            s=s[:30]
        er=effective_rank(NWSE(A))
        er1=effective_rank(NESW(A))
        newRow={'layer':i,'entanglement_spectrum':s,'effective_rank_nwse':er,'effective_rank_nesw':er1}
        
        newRow={k:_toN(v) for k,v in newRow.items()}
        curve.append(newRow)
    curve=pd.DataFrame(curve)

    ss=np.array(curve['entanglement_spectrum'].tolist())
    ee=curve['effective_rank_nwse'].tolist()
    ee1=curve['effective_rank_nesw'].tolist()
    iii=curve['layer']
    for sss in ss.T:
        plt.plot(iii,sss,'-k')
    plt.title(f'')
    plt.xlabel('RG Step')
    plt.ylabel('normalized eigenvalues')
    plt.show()

    plt.plot(iii,ee,'-k',label='nwse')
    #plt.plot(iii,ee1,label='nesw')
    plt.ylabel('effective rank')
    #plt.legend()
    plt.show()
    return curve
    
def show_diff(Ts,stride=1):
    curve=[]

    for i,A in tqdm([*enumerate(Ts)]):
        newRow={'layer':i}
        if i-stride>=0 and A.shape==Ts[i-stride].shape:
            newRow['diff']=(Ts[i]-Ts[i-stride]).norm()/Ts[i].norm()
            #newRow['diff1']=contract('ijkk->ij',Ts[i]-Ts[i-stride]).norm()
            #newRow['diff2']=contract('iikl->kl',Ts[i]-Ts[i-stride]).norm()
            #if (Ts[i]-Ts[i-stride]).norm()>2e-1 and i>10 and i<15:
            #    print(i)
                
        newRow={k:_toN(v) for k,v in newRow.items()}
        curve.append(newRow)
    curve=pd.DataFrame(curve)
    plt.plot(curve['layer'],curve['diff'],'.-',color='black',label='$|T\'-T|$')
    #plt.plot(curve['layer'],curve['diff1'],'.-',label='diff1')
    #plt.plot(curve['layer'],curve['diff2'],'.-',label='diff2')
    #plt.legend()
    plt.xlabel('RG Step')
    plt.ylabel('$|T\'-T|/|T|$')
    plt.yscale('log')
    plt.ylim((1e-7,2))
    plt.show()
    return curve
    
from HOTRGZ2 import reflect_tensor_axis,permute_tensor_axis

def show_asymmetry(Ts):
    curve=[]
    for i,A in enumerate(Ts):
        newRow={'layer':i}
        Arot=permute_tensor_axis(A)
        Aref=reflect_tensor_axis(A)
        if A.shape==Arot.shape:
            newRow['asym_rot']=_toN((Arot-A).norm()/A.norm())
        newRow['asym_ref']=_toN((Aref-A).norm()/A.norm())
        newRow={k:_toN(v) for k,v in newRow.items()}
        curve.append(newRow)

    curve=pd.DataFrame(curve)
    plt.plot(curve['layer'],curve['asym_rot'],'.-',label='rotation')
    plt.plot(curve['layer'],curve['asym_ref'],'x-',label='reflection')
    plt.legend()
    plt.xlabel('RG Step')
    plt.ylabel('$|T\'-T|/|T|$')
    #plt.yscale('log')
    plt.ylim((0,1))
    plt.show()
    return curve