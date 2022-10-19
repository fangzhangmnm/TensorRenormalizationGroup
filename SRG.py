if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='SRG for 2D and 3D AKLT')


    #parser.add_argument('--create',action='store_true')
    #parser.add_argument('--train',action='store',metavar='ITER',help='number of iteration')

    subparsers=parser.add_subparsers(dest='command')

    parser1=subparsers.add_parser('create',help='create a datapoint')

    parser1.add_argument('datapoint_name')
    parser1.add_argument('--override',action='store_true',help='override existing datapoint')
    parser1.add_argument('--model',choices=['AKLT2D','AKLT3D','Ising2D','Ising3D'],required=True)

    parser1.add_argument('--dtype',default='float64')
    parser1.add_argument('--device',default='cpu')
    parser1.add_argument('--max_dim',type=int,default=16)
    parser1.add_argument('--nLayers',type=int,default=20)
    parser1.add_argument('--nLayers_HOSVD',type=int,default=0,help='Do HOSVD after SRG') #not implemented
    parser1.add_argument('--init_with_HOSVD',action='store_true',help='Use HOTRG to initialize weights') #not implemented

    parser1.add_argument('--a1',type=float)
    parser1.add_argument('--a2',type=float)
    parser1.add_argument('--a3',type=float)
    parser1.add_argument('--beta',type=float)
    parser1.add_argument('--h',type=float)

    parser2=subparsers.add_parser('train',help='train a datapoint')
    parser2.add_argument('datapoint_names',nargs='+',help='list of datapoints to train')
    parser2.add_argument('--iter',type=int,default=1,help='number of iteration')
    parser2.add_argument('--device',default=None,help='override device setting')

    args=parser.parse_args()
import torch
from torch.utils.checkpoint import checkpoint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from opt_einsum import contract
from torch.profiler import profile, record_function, ProfilerActivity
def _toN(t):
    return t.detach().cpu().numpy()
def printDiff(name,value,ref):
    if ref is not None:
        print(name+':',value,'diff(abs):',value-ref)
    else:
        print(name+':',value)
    
import ast
def eval_np_array_literal(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))
# SRG BaseClass, can be used for HOTRG-like and XTRG-like contraction
class SRG(torch.nn.Module):
    def __init__(self,params,options):
        super(SRG,self).__init__()
        self.dtype={'float64':torch.float64,'float32':torch.float32}[options.get('dtype','float64')]
        self.device=options.get('device','cpu')
        self.max_dim=options.get('max_dim',16)
        self.nLayers=options.get('nLayers',20)
        self.use_checkpoint=options.get('use_checkpoint',True)
        self.observable_checkerboard=False
        
        self.params=torch.nn.ParameterDict({
            k:torch.nn.Parameter(torch.tensor(v,dtype=self.dtype,device=self.device)) for k,v in params.items()
        })
        self.persistent={}
        self.persistent['logZ']=0
        
    def __str__(self):
        rtval=""
        for k,v in self.params.items():
            rtval+=k+':'+v+'\n'
        rtval+='dtype:'+self.dtype+'\n'
        rtval+='device:'+self.device+'\n'
        rtval+='max_dim:'+self.max_dim+'\n'
        rtval+='nLayers:'+self.nLayers+'\n'
        rtval+='nSite:'+2**nLayers+'\n'
        
    def set_params(self,params):
        self.params=torch.nn.ParameterDict({
            k:torch.nn.Parameter(torch.tensor(v,dtype=self.dtype,device=self.device)) for k,v in params.items()
        })
        
    def toT(self,t):
        return torch.tensor(t,dtype=self.dtype,device=self.device)
    
    def generate_random_Isometry(self,dim1,dim2):
        dim=max(dim1,dim2)
        A=torch.randn(dim,dim,dtype=self.dtype,device=self.device)
        U=torch.matrix_exp(A-A.t())
        U=U[:dim1,:dim2]
        return U
    
    def TRG_same_T(self,T,*w):
        return self.TRG(T,T,*w)
    
    def _checkpoint(self,F,*ww):
        requires_grad=False
        for w in ww:
            if w.requires_grad:
                requires_grad=True
            #print(w.shape,w.requires_grad,end=" ")
        #print('')
        if self.use_checkpoint and requires_grad:
            return torch.utils.checkpoint.checkpoint(F,*ww)
        else:
            return F(*ww)
    
    def forward_tensor(self):
        logTotal=0
        T=self.get_T0()
        for i in range(self.nLayers):
            w=self.ws[(i*self.w_per_layer):((i+1)*self.w_per_layer)]
            T=self._checkpoint(self.TRG_same_T,T,*w)
                
            norm=torch.linalg.norm(T)
            T=T/norm
            logTotal=2*logTotal+torch.log(norm)
        return T,logTotal
    
    def forward_tensor_with_observable(self,T_op,contract_method=None,start_layer=0):
        T=self.get_T0()
        for i in range(start_layer):
            w=self.ws[(i*self.w_per_layer):((i+1)*self.w_per_layer)]
            T=self._checkpoint(self.TRG_same_T,T,*w)
            
        logTotal=0
        contracted=torch.zeros(self.nLayers,dtype=self.dtype,device=self.device)
        for i in range(start_layer,self.nLayers):
            w=self.ws[(i*self.w_per_layer):((i+1)*self.w_per_layer)]
            T1=self._checkpoint(self.TRG_same_T,T,*w)
            T2=self._checkpoint(self.TRG,T,T_op,*w)
            T3=self._checkpoint(self.TRG,T_op,T,*w)
            if self.observable_checkerboard and i<self.spacial_dim:
                T3=-T3

            T,T_op=T1,(T2+T3)/2
            norm=torch.linalg.norm(T)
            T,T_op=T/norm,T_op/norm
            logTotal=2*logTotal+torch.log(norm)
            
            if contract_method is not None:
                Z=contract(T,contract_method)
                Z_op=contract(T_op,contract_method)
                contracted[i]=Z_op/Z
            
        return T,T_op,logTotal,contracted
    
    
    def dlogZ(self,param):
        self.requires_grad_(False)
        self.params[param].requires_grad_(True)
        self.zero_grad()
        logZ=self.forward()
        logZ.backward()
        result=_toN(self.params[param].grad)
        self.params[param].requires_grad_(False)
        return result
    
    def update_single_layer(self,layer):
        print("update_single_layer",layer)
        self.requires_grad_(False)
        
        for i in range(layer*self.w_per_layer,(layer+1)*self.w_per_layer):
            self.ws[i].requires_grad_(True)
        self.zero_grad()
        
        logZ=self.forward()
        logZ.backward()
        
        with torch.no_grad():
            for i in range(layer*self.w_per_layer,(layer+1)*self.w_per_layer):
                E=self.ws[i].grad
                dim1,dim2=E.shape[0],E.shape[2]
                E=E.reshape(dim1*dim1,dim2)
                U,S,Vh=torch.linalg.svd(E,full_matrices=False)
                UVh=U@Vh
                #UVh=svd2UVh(E)
                del U,S,Vh,E
                
                #calculate diff
                UVh_old=self.ws[i].reshape(dim1*dim1,dim2)
                self.persistent['ws_diff'][i]=_toN(torch.norm(UVh_old.t()@UVh@UVh.t()@UVh_old-torch.eye(dim2,device=UVh.device)))
                del UVh_old
                    
                self.ws[i].data=UVh.reshape(dim1,dim1,dim2)
                del UVh
                torch.cuda.empty_cache()
        for i in range(layer*self.w_per_layer,(layer+1)*self.w_per_layer):
            self.ws[i].requires_grad_(False)
        return _toN(logZ)
        
    def optimize(self,nIter):
        self.persistent['ws_diff']=np.zeros(len(self.ws))
        
        torch.cuda.empty_cache()
        if nIter>1:
            pbar2=tqdm(range(nIter), leave=False)
            pbar2.set_postfix({k:_toN(v) for k,v in self.params.items()})
        else:
            pbar2=range(nIter)
        for i in pbar2:
            pbar=tqdm([*range(self.nLayers-1,-1,-1)]+[*range(self.nLayers)], leave=False)
            for j in pbar:
                ws_shape=self.ws[j*self.w_per_layer].shape
                if ws_shape[0]**2>ws_shape[2]:
                    logZ=self.update_single_layer(j)
                #else:
                #    print(f'Skip layer {j} shape={ws_shape}')
        #lock all grads
        for param in self.params.values(): 
            param.requires_grad_(False)
        for i in range(self.nLayers): #slightly faster
            self.ws[i].requires_grad_(False)
        
        self.persistent['logZ_diff']=np.abs(self.persistent['logZ']-logZ)
        self.persistent['logZ']=logZ
        
        # normalized by layer weight, number of elements in tensor
        # NOT USED but multiply by number of elements in last tensor to better match the effects in output
        self.persistent['ws_diff_normalized']=np.zeros(len(self.ws))
        for i in range(self.nLayers):
            for j in range(self.w_per_layer):
                ij=i*self.w_per_layer+j
                self.persistent['ws_diff_normalized'][ij]=self.persistent['ws_diff'][ij]/2**i#/torch.numel(self.ws[ij])*torch.numel(self.ws[-1])
        # ignore the last layers we take trace directly
        # use 10-norm so layers of large error has beter contribution       
        self.persistent['ws_diff_total']=np.average(self.persistent['ws_diff_normalized'][:-self.w_per_layer*self.spacial_dim])
class HOTRG(SRG):
    def __init__(self,params,options):
        super(HOTRG,self).__init__(params,options)
        self.nLayers_HOSVD=options.get('nLayers_HOSVD',0)
        self.persistent['magnetizationX']=0
        self.persistent['magnetizationY']=0
        self.persistent['magnetizationZ']=0
        self.persistent['energy']=0
    
    def create_isometries(self,start_dim,spacial_dim):
        ws=[]
        bond_dim=[start_dim]*spacial_dim
        for i in range(self.nLayers):
            for j in range(1,spacial_dim):
                old_dim=bond_dim[j]
                new_dim=min(old_dim**2,self.max_dim)
                U=self.generate_random_Isometry(old_dim**2,new_dim).view(old_dim,old_dim,new_dim)
                ws.append(U.detach())
                bond_dim[j]=new_dim
            bond_dim=bond_dim[1:]+[bond_dim[0]]
        self.ws=torch.nn.ParameterList([
            torch.nn.Parameter(v) for v in ws
        ])
        self.w_per_layer=spacial_dim-1
        self.spacial_dim=spacial_dim
        self.TRG={2:self.HOTRG2D,3:self.HOTRG3D}[self.spacial_dim]
        self.HOSVD={2:self.HOSVD2D,3:self.HOSVD3D}[self.spacial_dim]
        
        
    def HOTRG2D(self,T1,T2,w):
        return contract('ijkl,jmno,kna,lob->abim',T1,T2,w,w)#contract and rotate
    
    def HOTRG3D(self,T1,T2,w1,w2):
        return contract('ijklmn,jopqrs,kpa,lqb,mrc,nsd->abcdio',T1,T2,w1,w1,w2,w2)#contract and rotate
    
    def HOSVD2D(self,T1,T2):
        MM1=contract('ijkl,jmno,ipql,pmro->knqr',T1,T2,T1.conj(),T2.conj()).reshape(T1.shape[2]*T2.shape[2],-1)
        S1,U1=torch.linalg.eigh(MM1) #S1 ascending U S Uh=MM
        eps1=torch.sum(torch.abs(S1[-self.max_dim:])) # might be slightly minus due to numerical error
        
        MM2=contract('ijkl,jmno,ipql,pmro->knqr',T1.transpose(2,3),T2.transpose(2,3),T1.conj().transpose(2,3),T2.conj().transpose(2,3)).reshape(T1.shape[3]*T2.shape[3],-1)
        S2,U2=torch.linalg.eigh(MM2)
        eps2=torch.sum(torch.abs(S2[-self.max_dim:]))

        S,U=(S1,U1) if eps1<eps2 else (S2,U2)
        w=U[:,-self.max_dim:].reshape(T1.shape[2],T2.shape[2],-1)
        return contract('ijkl,jmno,kna,lob->abim',T1,T2,w,w),[w]
    
    def HOSVD3D(self,T1,T2):
        #print(T1.shape)
        MM1=contract('ijklmn,jopqrs,itulmn,tovqrs->kpuv',T1,T2,T1.conj(),T2.conj()).reshape(T1.shape[2]*T2.shape[2],-1)
        S1,U1=torch.linalg.eigh(MM1) #S1 ascending U S Uh=MM
        eps1=torch.sum(torch.abs(S1[-self.max_dim:])) # might be slightly minus due to numerical error
        
        MM2=contract('ijklmn,jopqrs,itulmn,tovqrs->kpuv',T1.transpose(2,3),T2.transpose(2,3),T1.conj().transpose(2,3),T2.conj().transpose(2,3)).reshape(T1.shape[3]*T2.shape[3],-1)
        S2,U2=torch.linalg.eigh(MM2)
        eps2=torch.sum(torch.abs(S2[-self.max_dim:]))

        S,U=(S1,U1) if eps1<eps2 else (S2,U2)
        w1=U[:,-self.max_dim:].reshape(T1.shape[2],T2.shape[2],-1)
        
        MM1=contract('ijklmn,jopqrs,itklun,topqvs->mruv',T1,T2,T1.conj(),T2.conj()).reshape(T1.shape[4]*T2.shape[4],-1)
        S1,U1=torch.linalg.eigh(MM1) #S1 ascending U S Uh=MM
        eps1=torch.sum(torch.abs(S1[-self.max_dim:])) # might be slightly minus due to numerical error
        
        MM2=contract('ijklmn,jopqrs,itklun,topqvs->mruv',T1.transpose(4,5),T2.transpose(4,5),T1.conj().transpose(4,5),T2.conj().transpose(4,5)).reshape(T1.shape[5]*T2.shape[5],-1)
        S2,U2=torch.linalg.eigh(MM2)
        eps2=torch.sum(torch.abs(S2[-self.max_dim:]))

        S,U=(S1,U1) if eps1<eps2 else (S2,U2)
        w2=U[:,-self.max_dim:].reshape(T1.shape[4],T2.shape[4],-1)
        #print(w1.shape,w2.shape)
        
        return contract('ijklmn,jopqrs,kpa,lqb,mrc,nsd->abcdio',T1,T2,w1,w1,w2,w2),[w1,w2]
    
    
    def generate_isometries_HOSVD(self):
        with torch.no_grad():
            logTotal=0
            T=self.get_T0()
            for i in tqdm(range(self.nLayers), leave=False):
                T,ww=self.HOSVD(T,T)
                for j in range(self.w_per_layer):
                    self.ws[i*self.w_per_layer+j].data=ww[j]
                norm=torch.linalg.norm(T)
                T=T/norm
                logTotal=2*logTotal+torch.log(norm)

            contract_all=[i for i in range(len(T.shape)//2) for j in range(2)]
            Z=contract(T,contract_all)
            self.persistent['logZ']=_toN((torch.log(Z)+logTotal)/2**self.nLayers)
    
    def forward(self):
        T,logTotal=self.forward_tensor()
        contract_all=[i for i in range(len(T.shape)//2) for j in range(2)]
        Z=contract(T,contract_all)
        return (torch.log(Z)+logTotal)/2**self.nLayers
    
    #def forward_with_observable(self,T_op,start_layer=0):
    #    contract_method=[i for i in range(len(T_op.shape)//2) for j in range(2)]
    #    _,_,_,contracted=self.forward_tensor_with_observable(T_op,contract_method=contract_method,start_layer=start_layer)
    #    return contracted
    
    def forward_and_HOTRG(self):
        T,logTotal=self.forward_tensor()
        for i in range(self.nLayers_HOSVD):
            T,_=self.HOSVD(T,T)
            norm=torch.linalg.norm(T)
            T=T/norm
            logTotal=2*logTotal+torch.log(norm)
        contract_all=[i for i in range(len(T.shape)//2) for j in range(2)]
        Z=contract(T,contract_all)
        return (torch.log(Z)+logTotal)/2**(self.nLayers+self.nLayers_HOSVD)
    
    def forward_with_observable_and_HOTRG(self,T_op,start_layer=0):
        contract_method=[i for i in range(len(T_op.shape)//2) for j in range(2)]
        T,T_op,logTotal,contracted=self.forward_tensor_with_observable(T_op,contract_method=contract_method,start_layer=start_layer)
        
        for i in range(self.nLayers_HOSVD):
            T1,ws=self.HOSVD(T,T)
            T2=self.TRG(T,T_op,*ws)
            T3=self.TRG(T_op,T,*ws)
            
            T,T_op=T1,(T2+T3)/2
            norm=torch.linalg.norm(T)
            T,T_op=T/norm,T_op/norm
            logTotal=2*logTotal+torch.log(norm)
            
        Z=contract(T,contract_method)
        Z_op=contract(T_op,contract_method)
        
        return contracted,Z_op/Z
    
    def calc_logZ(self):
        with torch.no_grad():
            logZ=_toN(self.forward_and_HOTRG())
            self.persistent['logZ_diff']=np.abs(self.persistent['logZ']-logZ)
            self.persistent['logZ']=logZ
    
    def calc_magnetization(self):
        axes=[0,1,2] if hasattr(self,'get_ST0') else [2]
        for axis in axes:
            with torch.no_grad():
                axisLabel=['X','Y','Z'][axis]
                ST0=self.get_ST0(axis) if hasattr(self,'get_ST0') else self.get_SZT0()
                a,b=self.forward_with_observable_and_HOTRG(ST0)
                self.persistent['magnetization'+axisLabel+'_per_layer']=_toN(torch.abs(a))
                magnetization=_toN(torch.abs(b))
                self.persistent['magnetization'+axisLabel+'_diff']=np.abs(self.persistent['magnetization'+axisLabel]-magnetization)
                self.persistent['magnetization'+axisLabel]=magnetization
            
    def calc_energy(self):
        with torch.no_grad():
            a,b=self.forward_with_observable_and_HOTRG(self.get_ET1(),start_layer=1)
            self.persistent['energy_per_layer']=_toN(a)
            energy=_toN(b)
            self.persistent['energy_diff']=np.abs(self.persistent['energy']-energy)
            self.persistent['energy']=energy
            
    def calc_all(self):
        self.calc_logZ()
        if hasattr(self,'get_ST0') or hasattr(self,'get_SZT0'):
            self.calc_magnetization()
        if hasattr(self,'get_ET1'):
            self.calc_energy()
class Ising2D(HOTRG):
    default_params={'beta':0.4406867935097715,'h':0}
    def __init__(self,params,options):
        super(Ising2D,self).__init__(params,options)
        self.create_isometries(start_dim=2,spacial_dim=2)
        self.observable_checkerboard=False
        
    def get_T0(self):
        beta,h=self.params['beta'],self.params['h']
        a=torch.sqrt(torch.cosh(beta))
        b=torch.sqrt(torch.sinh(beta))
        W=torch.stack([torch.stack([a,b]),torch.stack([a,-b])])
        sz=torch.stack([torch.exp(beta*h),torch.exp(-beta*h)])
        return contract('Ai,Aj,Ak,Al,A->ijkl',W,W,W,W,sz)#UDLR
    
    def get_SZT0(self):
        beta,h=self.params['beta'],self.params['h']
        a=torch.sqrt(torch.cosh(beta))
        b=torch.sqrt(torch.sinh(beta))
        W=torch.stack([torch.stack([a,b]),torch.stack([a,-b])])
        sz=torch.stack([torch.exp(beta*h),-torch.exp(-beta*h)])
        return contract('Ai,Aj,Ak,Al,A->ijkl',W,W,W,W,sz)#UDLR
    
    def get_ET1(self):
        beta,h=self.params['beta'],self.params['h']
        a=torch.sqrt(torch.cosh(beta))
        b=torch.sqrt(torch.sinh(beta))
        W=torch.stack([torch.stack([a,b]),torch.stack([a,-b])])
        sz=torch.stack([torch.exp(beta*h),torch.exp(-beta*h)])
        nBonds=self.toT(self.spacial_dim)#each site correspond to one spin and two bonds
        Eij=torch.stack([torch.stack([-nBonds-h,nBonds]),torch.stack([nBonds,-nBonds+h])]) # shouldn't x2
        w=self.ws[0]
        #'ijkl,jmno,kna,lob->abim'
        return contract('Ai,Aj,Ak,Al,A,Bj,Bm,Bn,Bo,B,AB,kna,lob->abim',W,W,W,W,sz,W,W,W,W,sz,Eij,w,w)

class Ising3D(HOTRG):
    default_params={'beta':0.2216544,'h':0}
    def __init__(self,params,options):
        super(Ising3D,self).__init__(params,options)
        self.create_isometries(start_dim=2,spacial_dim=3)
        self.observable_checkerboard=False
        
    def get_T0(self):
        beta,h=self.params['beta'],self.params['h']
        a=torch.sqrt(torch.cosh(beta))
        b=torch.sqrt(torch.sinh(beta))
        W=torch.stack([torch.stack([a,b]),torch.stack([a,-b])])
        sz=torch.stack([torch.exp(beta*h),torch.exp(-beta*h)])
        return contract('Ai,Aj,Ak,Al,Am,An,A->ijklmn',W,W,W,W,W,W,sz)
    
    def get_SZT0(self):
        beta,h=self.params['beta'],self.params['h']
        a=torch.sqrt(torch.cosh(beta))
        b=torch.sqrt(torch.sinh(beta))
        W=torch.stack([torch.stack([a,b]),torch.stack([a,-b])])
        sz=torch.stack([torch.exp(beta*h),-torch.exp(-beta*h)])
        return contract('Ai,Aj,Ak,Al,Am,An,A->ijklmn',W,W,W,W,W,W,sz)
    
    def get_ET1(self):
        beta,h=self.params['beta'],self.params['h']
        a=torch.sqrt(torch.cosh(beta))
        b=torch.sqrt(torch.sinh(beta))
        W=torch.stack([torch.stack([a,b]),torch.stack([a,-b])])
        sz=torch.stack([torch.exp(beta*h),torch.exp(-beta*h)])
        nBonds=self.toT(self.spacial_dim)#each site correspond to one spin and three bonds
        Eij=torch.stack([torch.stack([-nBonds-h,nBonds]),torch.stack([nBonds,-nBonds+h])]) # shouldn't x2
        w1,w2=self.ws[0],self.ws[1]
        #'ijklmn,jopqrs,kpa,lqb,mrc,nsd->abcdio'
        return contract('Ai,Aj,Ak,Al,Am,An,A,Bj,Bo,Bp,Bq,Br,Bs,B,AB,kpa,lqb,mrc,nsd->abcdio',W,W,W,W,W,W,sz,W,W,W,W,W,W,sz,Eij,w1,w1,w2,w2)
from scipy.special import comb
def get_CG_no_normalization(j):
    n=int(2*j)
    if n==0:
        return np.eye(1)
    CG=np.zeros((n+1,)+(2,)*n)
    for i in range(2**n):
        indices=tuple(map(int,bin(i)[2:].zfill(n)))
        m=np.sum(indices)
        CG[(m,)+indices]=1
    return CG
def get_CG(j):
    n=int(2*j)
    if n==0:
        return np.eye(1)
    CG=np.zeros((n+1,)+(2,)*n)
    for i in range(2**n):
        indices=tuple(map(int,bin(i)[2:].zfill(n)))
        m=np.sum(indices)
        CG[(m,)+indices]=1/np.sqrt(comb(n,m))
    return CG
def get_Singlet():
    return np.array([[0,1.],[-1.,0]])
def get_Lxyz(j):
    n=int(2*j+1)
    Lz=np.zeros((n,n))
    for i in range(n):
        m=i-j
        Lz[i,i]=m
    Lp=np.zeros((n,n))
    for i in range(n-1):
        m=i-j
        Lp[i+1,i]=np.sqrt(j*(j+1)-m*(m+1))
    Lm=Lp.T
    Lx=(Lp+Lm)/2
    iLy=(Lp-Lm)/2
    return Lx,iLy,Lz

class AKLT2D(HOTRG):
    default_params={'a1':np.sqrt(6)/2,'a2':np.sqrt(6)}
    def __init__(self,params,options):
        super(AKLT2D,self).__init__(params,options)
        self.create_isometries(start_dim=4,spacial_dim=2)
        self.observable_checkerboard=True
        
    def get_T0(self):
        projector=self.toT(get_CG_no_normalization(2))
        singlet=self.toT([[0,-1],[1,0]])
        ac0,ac1,ac2=self.toT(1),self.params['a1'],self.params['a2']
        deform=torch.stack([ac2,ac1,ac0,ac1,ac2])
        node=contract('aijkl,im,kn,a->amjnl',projector,singlet,singlet,deform)
        return contract('aijkl,amnop->imjnkolp',node,node).reshape(4,4,4,4)#UDLR

    def get_ST0(self,axis):
        projector=self.toT(get_CG_no_normalization(2))
        singlet=self.toT([[0,-1],[1,0]])
        ac0,ac1,ac2=self.toT(1),self.params['a1'],self.params['a2']
        deform=torch.stack([ac2,ac1,ac0,ac1,ac2])
        node=contract('aijkl,im,kn,a->amjnl',projector,singlet,singlet,deform)
        op=self.toT(get_Lxyz(2)[axis])
        return contract('aijkl,bmnop,ab->imjnkolp',node,node,op).reshape(4,4,4,4)#UDLR
    
    

class AKLT3D(HOTRG):
    default_params={'a1':np.sqrt(20/15),'a2':np.sqrt(20/6),'a3':np.sqrt(20/1)}
    def __init__(self,params,options):
        super(AKLT3D,self).__init__(params,options)
        self.create_isometries(start_dim=4,spacial_dim=3)
        self.observable_checkerboard=True
        
    def get_T0(self):
        projector=self.toT(get_CG_no_normalization(3))
        singlet=self.toT([[0,-1],[1,0]])
        ac0,ac1,ac2,ac3=self.toT(1),self.params['a1'],self.params['a2'],self.params['a3']
        deform=torch.stack([ac3,ac2,ac1,ac0,ac1,ac2,ac3])
        node=contract('aijklmn,io,kp,mq,a->aojplqn',projector,singlet,singlet,singlet,deform)
        return contract('aijklmn,aopqrst->iojpkqlrmsnt',node,node).reshape(4,4,4,4,4,4)#UDLRFB

    def get_ST0(self,axis):
        projector=self.toT(get_CG_no_normalization(3))
        singlet=self.toT([[0,-1],[1,0]])
        ac0,ac1,ac2,ac3=self.toT(1),self.params['a1'],self.params['a2'],self.params['a3']
        deform=torch.stack([ac3,ac2,ac1,ac0,ac1,ac2,ac3])
        node=contract('aijklmn,io,kp,mq,a->aojplqn',projector,singlet,singlet,singlet,deform)
        op=self.toT(get_Lxyz(3)[axis])
        return contract('aijklmn,bopqrst,ab->iojpkqlrmsnt',node,node,op).reshape(4,4,4,4,4,4)#UDLRFB
    
import os,torch
import pandas as pd
import time
import warnings

script_name=os.path.splitext(os.path.basename(__file__))[0] if '__file__' in globals() else 'notebook'

# Generate an empty checkpoint file defining the parameters and options
def CreateDataPoint(name,params,options,override=False):
    if os.path.exists(name+'.checkpoint'):
        if override:
            warnings.warn('Override existed datapoint '+name)
        else:
            warnings.warn('Skipped existed datapoint '+name)
            return
        
    
    
    # save the python dictionary containing parameters and options, 
    # later we will add model.state_dict(), which is the pytorch trained parameters, 
    # and model.persistent, which is some non-pytorch data I want to keep it for the next iteration
    torch.save({'params':params,'options':options,'iter':0},name+'.checkpoint')
    if os.path.exists(name+'.curve'):
        os.remove(name+'.curve')
    print('DataPoint Created:',name)
    print('options:',options)
    print('params:',params)

# todo test new function final_hotrg and init hotrg todo compare final hotrg with not with it





# Train an existed checkpoint file
def Train(name,nIter,Models,device=None,):
    # load the checkpoint    TODO add lock to the file
    checkpoint=torch.load(name+'.checkpoint',map_location=device)
    
    if os.path.exists(name+'.curve'):
        curve=pd.read_pickle(name+'.curve')
    else:
        curve=pd.DataFrame()
    
    # reconstruct the model
    Model=Models[checkpoint['options']['model']]
    if device is not None:
        checkpoint['options']['device']=device
        
        
    print('Training',name,'iter',checkpoint['iter'])
    print(checkpoint['params'])
    print(checkpoint['options'])
        
    model=Model(checkpoint['params'],checkpoint['options'])
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        model.persistent=checkpoint['model_persistent']
    else: # 'model_state_dict' not in checkpoint
        if checkpoint['options']['init_with_HOSVD']:
            start_time=time.time()
            model.generate_isometries_HOSVD()
            model.calc_all()
            end_time=time.time()
            newRow={**model.persistent,**(checkpoint['params']),**(checkpoint['options'])}
            newRow['iter']=checkpoint['iter']
            newRow['elapsed']=end_time-start_time
            
            curve=curve.append(newRow,ignore_index=True)
    
    if nIter>0:
        # train the model
        start_time=time.time()

        model.optimize(nIter)
        model.calc_logZ()
        model.calc_magnetization()

        end_time=time.time()
        checkpoint['iter']+=nIter

        # add row

        # add row to the curve
        newRow={**model.persistent,**(checkpoint['params']),**(checkpoint['options'])}
        newRow['iter']=checkpoint['iter']
        newRow['elapsed']=end_time-start_time

        curve=curve.append(newRow,ignore_index=True)
    
    #save the curve
    curve.to_pickle(name+'.curve')
    curve.to_csv(name+'.curve.csv')
    
    # save the model
    checkpoint['model_state_dict']=model.state_dict()
    checkpoint['model_persistent']=model.persistent
    
    torch.save(checkpoint,name+'.checkpoint')
    
    return curve



Models={'AKLT2D':AKLT2D,'AKLT3D':AKLT3D,'Ising2D':Ising2D,'Ising3D':Ising3D}

if __name__ == "__main__":
    if args.command=='create':
        options={
            'model':args.model,
            'dtype':args.dtype,
            'device':args.device,
            'max_dim':args.max_dim,
            'nLayers':args.nLayers,
            'use_checkpoint':True,
            'init_with_HOSVD':args.init_with_HOSVD,
            'nLayers_HOSVD':args.nLayers_HOSVD,
        }

        Model=Models[options['model']]
        params=Model.default_params
        for param_name in ['a1','a2','a3','beta','h']:
            if vars(args)[param_name] is not None:
                params[param_name]=vars(args)[param_name]


        CreateDataPoint(args.datapoint_name,params,options,override=args.override)

    if args.command=='train':
        _ni=max(args.iter,1)
        _ni_inner=min(args.iter,1)
        for _i in tqdm(range(_ni*len(args.datapoint_names))):
            datapoint_name=args.datapoint_names[_i%len(args.datapoint_names)]
            Train(datapoint_name,nIter=_ni_inner,Models=Models,device=args.device)

            