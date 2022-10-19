import opt_einsum as oe
import os
import sys
import re
import torch
#import cupy
import time
import numpy as np
#from utils import kronecker_product as kron
from torch.utils.checkpoint import checkpoint
from datetime import datetime
#from expm import expm
from torch.autograd import Variable

def HOTRG(tau,NBeta):
        T0= initRho3D(tau)
        M = getM()
        E = getE()
        lnZ = 0.0
        torch.cuda.empty_cache()
        for nbeta in range(0,NBeta):
           #print('Allocated cuda memory:%d, cached:%d '%(torch.cuda.memory_allocated() / 1024**2,torch.cuda.memory_reserved() / 1024**2))
           #print(nbeta)
           
           iab=nbeta%3
           mWl = Wl[nbeta].to(device)
           mWr = Wr[nbeta].to(device)
           if iab==0:
               #T,mWl,mWr=self.MergX( T0, T0, self.D)  
               T=rgX(T0,mWl,mWr)  
               if nbeta==0:
                   E=rgXE1step(E,mWl,mWr)
               else:
                   E=rgXimp(T0,E,mWl,mWr)
               M = rgXimp(T0,M,mWl,mWr)
           elif iab==1:
               #T,mWl,mWr=self.MergY( T0, T0, self.D)  
               T=rgY(T0,mWl,mWr)  
               E=rgYimp(T0,E,mWl,mWr)
               M =rgYimp(T0,M,mWl,mWr)
           else:
               #T,mWl,mWr=self.MergZ( T0, T0, self.D)  
               T=rgZ(T0,mWl,mWr)  
               E=rgZimp(T0,E,mWl,mWr)
               M =rgZimp(T0,M,mWl,mWr)

           norm=torch.norm(T)
           T = T/norm
           T0 = T
           lnZ = 2*lnZ + torch.log(norm) 
           M = M/norm
           E = E/norm

           eT=getTrace3(T)
           eM=getTrace3(M)
           eE=getTrace3(E)
           mag=eM/eT
           mE=-3*eE/eT
           Fe=-lnZ/2**(nbeta+1)/tau
           entropy= (mE-Fe)*tau
           print('[layer=%d, eM=%f, eT=%f, mag=%f, E=%f, Fe=%f, Entropy=%f]'%(nbeta,eM,eT,mag,mE,Fe, entropy))
           del mWl, mWr, T
           torch.cuda.empty_cache()
        del M, E, T0
        torch.cuda.empty_cache()
        lnZ = lnZ + torch.log(torch.abs(eT))
        #print('*** Final Free energy= %+.15f'%(-lnZ/2**(NBeta)/tau))
        print('Final: eM=%.10f, eT=%.10f, mag=%.10f E=%.10f, Fe=%.10f, Entropy=%.10f]'%(eM,eT,mag,mE,Fe, entropy))
        Fe=-lnZ/2**(NBeta)/tau
        return Fe, mag, mE, entropy

def getTrace3(Ta):  # take trace +
        #print(Ta.size)
        Vl = oe.contract('iijjkk',Ta,backend='torch')
        return Vl 

def initRho3D(tau):  #for classical Ising 
        # + index ordering l,r,front, back, d, u
        #
        h  = torch.tensor([5e-7], dtype=tau.dtype, device=tau.device)   #small magnetic field
        dim=3

        rtotal=torch.sqrt(torch.exp(-2*tau)+torch.exp(2*tau)*torch.sinh(2*h*tau/dim)**2)
        cos2= torch.exp(tau)*torch.sinh(2*tau*h/dim)/rtotal
        #sin2= torch.exp(-tau)/rtotal
        cos=torch.sqrt((1+cos2)/2)
        sin=torch.sqrt((1-cos2)/2)
        lam1=torch.sqrt(torch.exp(tau)*torch.cosh(2*tau*h/dim)+rtotal)
        lam2=torch.sqrt(torch.exp(tau)*torch.cosh(2*tau*h/dim)-rtotal)
        lam =[lam1,lam2]
        U=[[cos,sin],[sin,-cos]]
        T = []
        for i in range(2):
           for j in range(2):
              for k in range(2):
                 for l in range(2):
                    for m in range(2):
                       for n in range(2):
                          t0=0
                          for s in range(2):
                             t0=t0+U[s][i]*U[s][j]*U[s][k]*U[s][l]*U[s][m]*U[s][n]*lam[i]*lam[j]*lam[k]*lam[l]*lam[m]*lam[n]

                          T.append(t0)
        #print(T)
        T = torch.stack(T).contiguous().view(2, 2, 2, 2, 2, 2)
        return T

def getM(tau):  #for classical Ising 
        #tau,device,dtype,
        #model = self.params[-1],self.device,self.dtype,self.model
        h  = torch.tensor([1e-9], dtype=tau.dtype, device=tau.device)
        dim=3

        rtotal=torch.sqrt(torch.exp(-2*tau)+torch.exp(2*tau)*torch.sinh(2*h*tau/dim)**2)
        cos2= torch.exp(tau)*torch.sinh(2*tau*h/dim)/rtotal
        #sin2= torch.exp(-tau)/rtotal
        cos=torch.sqrt((1+cos2)/2)
        sin=torch.sqrt((1-cos2)/2)
        lam1=torch.sqrt(torch.exp(tau)*torch.cosh(2*tau*h/dim)+rtotal)
        lam2=torch.sqrt(torch.exp(tau)*torch.cosh(2*tau*h/dim)-rtotal)
        lam =[lam1,lam2]
        U=[[cos,sin],[sin,-cos]]
        T = []
        for i in range(2):
           for j in range(2):
              for k in range(2):
                 for l in range(2):
                    for m in range(2):
                       for n in range(2):
                          t0=0
                          # (-1)^s= 1-2s
                          for s in range(2):
                             t0=t0+(1-2*s)*U[s][i]*U[s][j]*U[s][k]*U[s][l]*U[s][m]*U[s][n]*lam[i]*lam[j]*lam[k]*lam[l]*lam[m]*lam[n]

                          T.append(t0)
        #print(T)
        T = torch.stack(T).contiguous().view(2, 2, 2, 2, 2, 2)
        return T

def getE(tau):  #for classical Ising 
        #tau,device,dtype,model = self.params[-1],self.device,self.dtype,self.model
        h  = torch.tensor([1e-9], dtype=tau.dtype, device=tau.device)
        dim=3

        rtotal=torch.sqrt(torch.exp(-2*tau)+torch.exp(2*tau)*torch.sinh(2*h*tau/dim)**2)
        cos2= torch.exp(tau)*torch.sinh(2*tau*h/dim)/rtotal
        #sin2= torch.exp(-tau)/rtotal
        cos=torch.sqrt((1+cos2)/2)
        sin=torch.sqrt((1-cos2)/2)
        lam1=torch.sqrt(torch.exp(tau)*torch.cosh(2*tau*h/dim)+rtotal)
        lam2=torch.sqrt(torch.exp(tau)*torch.cosh(2*tau*h/dim)-rtotal)
        lam =[lam1,lam2]
        U=[[cos,sin],[sin,-cos]]
        ebH=[[torch.exp(tau+2*h*tau/dim),torch.exp(-tau)], \
             [torch.exp(-tau),torch.exp(tau-2*h*tau/dim)]]
        T = []
        for i in range(2):
         for k in range(2):
          for l in range(2):
           for m in range(2):
            for n in range(2):
             for o in range(2):
              for p in range(2):
               for q in range(2):
                for r in range(2):
                 for s in range(2):
                   t0=0
                   # (-1)^s= 1-2s
                   for s1 in range(2):
                    for s2 in range(2):
                       t0=t0+((1-2*s1)*(1-2*s2)+ h*(2-2*s1-2*s2))* ebH[s1][s2]*\
                            U[s1][i]*U[s1][k]*U[s1][l]*U[s1][m]*U[s1][n]* \
                            lam[i]*lam[k]*lam[l]*lam[m]*lam[n]* \
                            U[s2][o]*U[s2][p]*U[s2][q]*U[s2][r]*U[s2][s]* \
                            lam[o]*lam[p]*lam[q]*lam[r]*lam[s]

                   T.append(t0)
        T = torch.stack(T).view(2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        return T


def rgX(T, Wxy,Wxz):  
        # ++  merge two x aligned tensors
        # 
        optimize='optimal'#False  #'auto-hq'
        #path_info = oe.contract_path('ijklmn,jopqrs,kpa,lqb,mrc,nsd->ioabcd',T,T,Wxy,Wxy,Wxz,Wxz,optimize=optimize)
        #print('rgX: T: ',path_info[0])
        T = oe.contract('ijklmn,jopqrs,kpa,lqb,mrc,nsd->ioabcd',T,T,Wxy,Wxy,Wxz,Wxz,optimize=[(0, 2), (0, 1), (0, 2), (1, 2), (0, 1)],backend='torch')
        return T

def MergX(T1, T2, Dcut):  
        # ++  merge two x aligned tensors
        # 
        backend='auto'
        optimize='optimal'#False
        T1.detach()
        T2.detach()
        M1 = oe.contract('ijklmn,ipklmo->jpno',T1,T1.conj(),optimize=[(0,1)],backend=backend)
        M2 = oe.contract('ijklmn,pjklmo->ipno',T2,T2.conj(),optimize=[(0,1)],backend=backend)
        T = oe.contract('jpno,jpmq->nmoq',M1,M2,optimize=[(0,1)],backend=backend)
        dim=T.size(dim=0)
        T=torch.reshape(T, (dim*dim,dim*dim))
        RealCut=min(dim*dim, Dcut)
        L, Q= torch. linalg.eigh(T)
        Wxz=Q.conj()[:,dim*dim-RealCut: dim*dim]
        Wxz=torch.reshape(Wxz, (dim,dim,RealCut))
        del T,L,Q
        torch.cuda.empty_cache()

        M1 = oe.contract('ijklmn,ipkomn->jplo',T1,T1.conj(),optimize=[(0,1)],backend=backend)
        M2 = oe.contract('ijklmn,pjkomn->iplo',T2,T2.conj(),optimize=[(0,1)],backend=backend)
        T = oe.contract('jpno,jpmq->nmoq',M1,M2,optimize=[(0,1)],backend=backend)
        dim=T.size(dim=0)
        T=torch.reshape(T, (dim*dim,dim*dim))
        RealCut=min(dim*dim, Dcut)
        L, Q= torch. linalg.eigh(T)
        Wxy=Q.conj()[:,dim*dim-RealCut: dim*dim]
        Wxy=torch.reshape(Wxy, (dim,dim,RealCut))
        del T,L,Q
        #mT1=T1.cpu()#to('cpu')
        #mT2=T2.cpu()#to('cpu')
        #mWxy=Wxy.detach().cpu()#to('cpu')
        #mWxz=Wxz.detach().cpu()#to('cpu')
        #del T1,T2,Wxy,Wxz
        #torch.cuda.empty_cache()


        #path_info= oe.contract_path('ijklmn,jopqrs,kpa,lqb,mrc,nsd->ioabcd',T1,T2,Wxy,Wxy,Wxz,Wxz,optimize=optimize)
        #print('MergX: T: ',path_info[0])
        T = oe.contract('ijklmn,jopqrs,kpa,lqb,mrc,nsd->ioabcd',T1,T2,Wxy,Wxy,Wxz,Wxz,optimize=[(0, 2), (0, 1), (0, 2), (1, 2), (0, 1)],backend=backend)
        #T = oe.contract('ijklmn,jopqrs,kpa,lqb,mrc,nsd->ioabcd',mT1,mT2,mWxy,mWxy,mWxz,mWxz,optimize=[(0, 2), (0, 1), (0, 2), (1, 2), (0, 1)],backend=backend)
        #del T1,T2
        torch.cuda.empty_cache()
        return T, Wxy, Wxz
        #return T, mWxy, mWxz

def rgY(T, Wyx,Wyz):  
        #  +  merge two y aligned tensors
        # +
        optimize='optimal'#False
        #path_info = oe.contract_path('ijklmn,oplqrs,ioa,jpb,mrc,nsd->abkqcd',T,T,Wyx,Wyx,Wyz,Wyz,optimize=optimize)
        #print('rgY: T: ',path_info[0])
        T = oe.contract('ijklmn,oplqrs,ioa,jpb,mrc,nsd->abkqcd',T,T,Wyx,Wyx,Wyz,Wyz,optimize=[(0, 2), (0, 1), (0, 2), (1, 2), (0, 1)],backend='torch')

        return T

def MergY(T1, T2, Dcut):  
        # ++  merge two y aligned tensors
        # 
        backend='auto'
        optimize='optimal'#False
        T1.detach()
        T2.detach()
        M1 = oe.contract('ijklmn,ijkpmo->lpno',T1,T1.conj(),optimize=[(0,1)],backend=backend)
        M2 = oe.contract('ijklmn,ijplmo->kpno',T2,T2.conj(),optimize=[(0,1)],backend=backend)
        T = oe.contract('jpno,jpmq->nmoq',M1,M2,optimize=[(0,1)],backend=backend)
        dim=T.size(dim=0)
        T=torch.reshape(T, (dim*dim,dim*dim))
        RealCut=min(dim*dim, Dcut)
        L, Q= torch. linalg.eigh(T)
        Wyz=Q.conj()[:,dim*dim-RealCut: dim*dim]
        Wyz=torch.reshape(Wyz, (dim,dim,RealCut))
        del T,L,Q
        torch.cuda.empty_cache()

        M1 = oe.contract('ijklmn,iokpmn->lpjo',T1,T1.conj(),optimize=[(0,1)],backend=backend)
        M2 = oe.contract('ijklmn,ioplmn->kpjo',T2,T2.conj(),optimize=[(0,1)],backend=backend)
        T = oe.contract('jpno,jpmq->nmoq',M1,M2,optimize=[(0,1)],backend=backend)
        dim=T.size(dim=0)
        T=torch.reshape(T, (dim*dim,dim*dim))
        RealCut=min(dim*dim, Dcut)
        L, Q= torch. linalg.eigh(T)
        Wyx=Q.conj()[:,dim*dim-RealCut: dim*dim]
        Wyx=torch.reshape(Wyx, (dim,dim,RealCut))
        del T,L,Q
        #mT1=T1.cpu()#to('cpu')
        #mT2=T2.cpu()#to('cpu')
        #mWyx=Wyx.detach().cpu()#to('cpu')
        #mWyz=Wyz.detach().cpu()#to('cpu')
        #del T1,T2,Wyx,Wyz
        #torch.cuda.empty_cache()

        #T = oe.contract('ijklmn,oplqrs,ioa,jpb,mrc,nsd->abkqcd',T1,T2,Wyx,Wyx,Wyz,Wyz,optimize='auto-hq',backend='torch')
        #path_info= oe.contract_path('ijklmn,oplqrs,ioa,jpb,mrc,nsd->abkqcd',T1,T2,Wyx,Wyx,Wyz,Wyz,optimize=optimize)
        #print('MergY: T: ', path_info[0])
        T = oe.contract('ijklmn,oplqrs,ioa,jpb,mrc,nsd->abkqcd',T1,T2,Wyx,Wyx,Wyz,Wyz,optimize=[(0, 2), (0, 1), (0, 2), (1, 2), (0, 1)],backend=backend)
        #T = oe.contract('ijklmn,oplqrs,ioa,jpb,mrc,nsd->abkqcd',mT1,mT2,mWyx,mWyx,mWyz,mWyz,optimize=[(0, 2), (0, 1), (0, 2), (1, 2), (0, 1)],backend=backend)
        del T1,T2
        torch.cuda.empty_cache()
        return T, Wyx, Wyz
        #return T, mWyx, mWyz

def rgZ(T, Wzx,Wzy):  
        # +  merge two z aligned tensors
        # +
        optimize='optimal'#False
        #path_info = oe.contract_path('ijklmn,opqrsm,oia,pjb,qkc,rld->abcdsn',T,T,Wzx,Wzx,Wzy,Wzy,optimize=optimize)
        #print('rgZ: ',path_info[0])
        T = oe.contract('ijklmn,opqrsm,oia,pjb,qkc,rld->abcdsn',T,T,Wzx,Wzx,Wzy,Wzy,optimize=[(0, 2), (0, 1), (0, 2), (1, 2), (0, 1)],backend='torch')
        return T

def MergZ(T1, T2, Dcut):  
        # ++  merge two z aligned tensors
        # 
        backend='auto'
        optimize='optimal'
        T1.detach()
        T2.detach()
        M1 = oe.contract('ijklmn,ioklmp->npjo',T1,T1.conj(),optimize=[(0,1)],backend=backend)
        M2 = oe.contract('ijklmn,ioklpn->mpjo',T2,T2.conj(),optimize=[(0,1)],backend=backend)
        T = oe.contract('jpno,jpmq->nmoq',M1,M2,optimize=[(0,1)],backend=backend)
        dim=T.size(dim=0)
        T=torch.reshape(T, (dim*dim,dim*dim))
        RealCut=min(dim*dim, Dcut)
        L, Q= torch. linalg.eigh(T)
        Wzx=Q.conj()[:,dim*dim-RealCut: dim*dim]
        Wzx=torch.reshape(Wzx, (dim,dim,RealCut))

        del T,L,Q
        torch.cuda.empty_cache()
        M1 = oe.contract('ijklmn,ijkomp->nplo',T1,T1.conj(),optimize=[(0,1)],backend=backend)
        M2 = oe.contract('ijklmn,ijkopn->mplo',T2,T2.conj(),optimize=[(0,1)],backend=backend)
        T = oe.contract('jpno,jpmq->nmoq',M1,M2,optimize=[(0,1)],backend=backend)
        dim=T.size(dim=0)
        T=torch.reshape(T, (dim*dim,dim*dim))
        RealCut=min(dim*dim, Dcut)
        L, Q= torch. linalg.eigh(T)
        Wzy=Q.conj()[:,dim*dim-RealCut: dim*dim]
        Wzy=torch.reshape(Wzy, (dim,dim,RealCut))
        
        del T,L,Q
        #mT1=T1.cpu()#to('cpu')
        #mT2=T2.cpu()#to('cpu')
        #mWzx=Wzx.detach().cpu()#to('cpu')
        #mWzy=Wzy.detach().cpu()#to('cpu')
        #del T1,T2,Wzx,Wzy
        #torch.cuda.empty_cache()
        #path_info = oe.contract_path('ijklmn,opqrsm,oia,pjb,qkc,rld->abcdsn',T1,T2,Wzx,Wzx,Wzy,Wzy,optimize=optimize)
        #print('MergZ: T: ',path_info[0])
        T = oe.contract('ijklmn,opqrsm,oia,pjb,qkc,rld->abcdsn',T1,T2,Wzx,Wzx,Wzy,Wzy,optimize=[(0, 2), (0, 1), (0, 2), (1, 2), (0, 1)],backend=backend)
        #T = oe.contract('ijklmn,opqrsm,oia,pjb,qkc,rld->abcdsn',mT1,mT2,mWzx,mWzx,mWzy,mWzy,optimize=[(0, 2), (0, 1), (0, 2), (1, 2), (0, 1)],backend=backend)
        del T1,T2
        torch.cuda.empty_cache()
        return T, Wzx, Wzy
        #return T, mWzx, mWzy

def rgXE1step( M, Wxy,Wxz):  
        # ++  merge two x aligned tensors
        # 
        optimize='optimal'#False
        #path_info = oe.contract_path('iklmnopqrs,kpa,lqb,mrc,nsd->ioabcd',M,Wxy,Wxy,Wxz,Wxz,optimize=optimize)
        #print('rgXE1step: ',path_info[0])
        T1 = oe.contract('iklmnopqrs,kpa,lqb,mrc,nsd->ioabcd',M,Wxy,Wxy,Wxz,Wxz,optimize= [(0, 1), (0, 3), (0, 2), (0, 1)],backend='torch')
        return T1

def rgXimp(T, M, Wxy,Wxz):  
        # ++  merge two x aligned tensors
        # 
        optimize='optimal'#False
        #path_info=oe.contract_path('ijklmn,jopqrs,kpa,lqb,mrc,nsd->ioabcd',T,M,Wxy,Wxy,Wxz,Wxz,optimize=optimize) 
        #print('rgXimp: T1: ',path_info[0])
        T1=oe.contract('ijklmn,jopqrs,kpa,lqb,mrc,nsd->ioabcd',T,M,Wxy,Wxy,Wxz,Wxz,optimize=[(0, 2), (0, 1), (0, 2), (1, 2), (0, 1)],backend='torch') 
        #path_info= oe.contract_path('ijklmn,jopqrs,kpa,lqb,mrc,nsd->ioabcd',M,T,Wxy,Wxy,Wxz,Wxz,optimize=optimize)
        #print('rgXimp: T2: ',path_info[0])
        T2= oe.contract('ijklmn,jopqrs,kpa,lqb,mrc,nsd->ioabcd',M,T,Wxy,Wxy,Wxz,Wxz,optimize=[(0, 2), (0, 1), (0, 2), (1, 2), (0, 1)],backend='torch')
        return 0.5*(T1+T2)

def rgYimp(T, M, Wyx,Wyz):  
        #  +  merge two y aligned tensors
        # +
        T1 = oe.contract('ijklmn,oplqrs,ioa,jpb,mrc,nsd->abkqcd',T,M,Wyx,Wyx,Wyz,Wyz,optimize=[(0, 2), (0, 1), (0, 2), (1, 2), (0, 1)],backend='torch')
        T2 = oe.contract('ijklmn,oplqrs,ioa,jpb,mrc,nsd->abkqcd',M,T,Wyx,Wyx,Wyz,Wyz,optimize=[(0, 2), (0, 1), (0, 2), (1, 2), (0, 1)],backend='torch')
        return 0.5*(T1+T2)

def rgZimp(T, M,  Wzx,Wzy):  
        # +  merge two z aligned tensors
        # +
        T1 = oe.contract('ijklmn,opqrsm,oia,pjb,qkc,rld->abcdsn',T,M,Wzx,Wzx,Wzy,Wzy,optimize=[(0, 2), (0, 1), (0, 2), (1, 2), (0, 1)],backend='torch')
        T2 = oe.contract('ijklmn,opqrsm,oia,pjb,qkc,rld->abcdsn',M,T,Wzx,Wzx,Wzy,Wzy,optimize=[(0, 2), (0, 1), (0, 2), (1, 2), (0, 1)],backend='torch')
        return 0.5*(T1+T2)

def evolveMandE(tau,NBeta):# one third step at a time/layer
        Dis=True
        T = initRho3D(tau)
        M = getM(tau)
        E = getE(tau)
        eM= getTrace3(M)
        #print('[eM=%f]'%(eM))
        #print('original T size:',T.size())
        lnZ=0.0
        for nbeta in range(0,NBeta):
           #print('Allocated cuda memory:%d, cached:%d '%(torch.cuda.memory_allocated() / 1024**2,torch.cuda.memory_cached() / 1024**2))
           #print('nbeta =%d'%(nbeta))
           #Sizea = list(T.size())

           # obtain Isometry tensor from antisymmetric tensor

           mWl = Wl[nbeta].to(device)
           mWr = Wr[nbeta].to(device)
           # evolution Ta 
           iab=nbeta%3
           if iab==0:
               if nbeta==0:
                   E=rgXE1step(E,mWl,mWr)
               else:
                   E=rgXimp(T,E,mWl,mWr)
               M = rgXimp(T,M,mWl,mWr)
               T = rgX(T,mWl,mWr)
           elif iab==1:
               E=rgYimp(T,E,mWl,mWr)
               M =rgYimp(T,M,mWl,mWr)
               T =rgY(T,mWl,mWr)
           else:
               E=rgZimp(T,E,mWl,mWr)
               M = rgZimp(T,M,mWl,mWr)
               T = rgZ(T,mWl,mWr)

           del mWl, mWr
           torch.cuda.empty_cache()
           normT=torch.norm(T)
           T = T/normT
           lnZ=2*lnZ + torch.log(normT)
           M = M/normT
           E = E/normT

           eT=getTrace3(T)
           eM=getTrace3(M)
           eE=getTrace3(E)
           mag=eM/eT
           mE=-3*eE/eT
           Fe=-lnZ/2**(nbeta+1)/tau
           entropy= (mE-Fe)*tau
           if Dis: print('[layer=%d, eM=%f, eT=%f, mag=%.15f, E=%.15f, Fe=%.15f, Entropy=%.15f]'%(nbeta,eM,eT,mag,mE,Fe, entropy))
           #torch.cuda.empty_cache()
           #del Wl, Wr
           #torch.cuda.empty_cache()
        return Fe,mag,mE,entropy


def forwardThird(nlayer): #HOTRG forward; one third step at a time/layer
        Dis=False
        global Params
        global NBeta
        global Tas, normT, lnZs
        tau=Params[-1]
        
        if nlayer == 0:
            #print('# init T lnZ')
            T = initRho3D(tau)
            lnZ = 0.0
        else:

            T = Tas[nlayer-1].to(device)
            lnZ = lnZs[nlayer-1].to(device)


        for nbeta in range(nlayer,NBeta):

           mWl = Params[nbeta]
           mWr = Params[nbeta+NBeta]
           #if nbeta==nlayer:
           #   mWl = Params[nbeta]
           #   mWr = Params[nbeta+NBeta]
           #else:
           #   mWl = Wl[nbeta].to(device)
           #   mWr = Wr[nbeta].to(device)

           # obtain Isometry tensor from antisymmetric tensor
           # evolution T
           iab=nbeta%3
           if iab==0:
               #print('rgX')
               T = checkpoint(rgX,T,mWl,mWr)
           elif iab==1:
               #print('rgY')
               T = checkpoint(rgY,T,mWl,mWr)
           else:
               #print('rgZ')
               T = checkpoint(rgZ,T,mWl,mWr)

           norm=torch.norm(T)
           T = T/norm
           lnZ = 2*lnZ + torch.log(norm)

           if len(Tas)<nbeta+1:
               Tas.append(T.detach().to('cpu'))
               normTs.append(norm.detach().to('cpu'))
           else:
               Tas[nbeta]=T.detach().to('cpu')
               normTs[nbeta]=norm.detach().to('cpu')
           if len(lnZs)<nbeta+1: lnZs.append(lnZ.detach().to('cpu'))
           else:                 lnZs[nbeta]=lnZ.detach().to('cpu')

           del mWl,mWr
           torch.cuda.empty_cache()
        # free energy
        ee=getTrace3(T)
        if Dis: print('ee=%f'%(ee),end=' ')
        lnZ = lnZ + torch.log(torch.abs(ee))
        del T 
        torch.cuda.empty_cache()
        return lnZ

def update_single_layerThirdOld(nlayer):  #this uses HOTRG forwardThird
        Dis=True#False
        global Params, NBeta, tau, lnZs,loss_conv
        #lnList=[(-1)*m/2**(j+1)/tau.detach().to('cpu') for (j,m) in enumerate(lnZs)]
        #print(lnList)
        loss_old = 0
        #ibeta=nlayer%3  # 0 -> 0,1; 1 -> 2,3; 2 ->4,5
        #for niter in range(self.Niter):
        not_conv=True; niter=0
        while(not_conv):
            if Dis: print('[(%d)-th iter]'%(niter),end=' ')
            for ii in range(2):
                if Dis: print(' W(%02d,%d),'%(nlayer,ii),end=' ')
                Params[nlayer+ii*NBeta].requires_grad = True
                Params[nlayer+ii*NBeta].grad = None
                loss = forwardThird(nlayer)
                loss.backward()

                with torch.no_grad():
                   E = Params[nlayer+ii*NBeta].grad
                   #print('grad: ',E)
                   D, D, D_new = E.shape
                   #print('E.shape',E.shape)
                   E = E.view(D**2, D_new)
                   # perform MERA update
                   U,S,V = torch.svd(E)
                   Params[nlayer+ii*NBeta].data = (U@V.t()).view(D,D,D_new)
                   #print('%+.15f'%(-loss.item()/2**(NBeta+1)/tau), end=' ')
                   if Dis: print('F_en=%+.10f'%(-loss.item()/2**(NBeta)/tau), end=' ')
                   #print('\nAllocated cuda memory:%d, cached:%d '%(torch.cuda.memory_allocated() / 1024**2,torch.cuda.memory_cached() / 1024**2))
                   del U,S,V,E 
                   torch.cuda.empty_cache()
                   #print('Allocated cuda memory:%d, cached:%d '%(torch.cuda.memory_allocated() / 1024**2,torch.cuda.memory_cached() / 1024**2))
                Params[nlayer+ii*NBeta].requires_grad = False 
            niter=niter+1
            rel_err= (loss_old - loss.item())/loss.item()
            not_conv= abs(rel_err) > loss_conv #and niter < self.Niter
            loss_old = loss.item()
            #niter=niter+1
            #print(' ')
        if Dis: print('After %d iter: rel err= %E'%(niter,rel_err))
        #if Dis: print('Allocated cuda memory:%d, cached:%d '%(torch.cuda.memory_allocated() / 1024**2,torch.cuda.memory_cached() / 1024**2))
        #--------------------------#
        #with torch.no_grad():
        #    loss = forwardThird(nlayer)
        #--------------------------#
        return loss.item()

def update_single_layerThird(nlayer):  #this uses HOTRG forwardThird; does not update unitary
       
        Dis=True#False
        global Params, NBeta, tau, lnZs,loss_conv, D
        loss_old = 0
        not_conv=True; niter=0
        while(not_conv):
           if Dis: print('\n[(%d)-th iter]'%(niter),end=' ')
           for ii in range(2):
                if Dis: print(' W(%02d,%d),'%(nlayer,ii),end=' ')
                SizeW=list(Params[nlayer+ii*NBeta].size())
                #print(SizeW)
                do_Diff=False
                if (SizeW[2]<SizeW[0]*SizeW[1]):
                    do_Diff=True
                #else: print('skip!')
                if do_Diff: 
                    Params[nlayer+ii*NBeta].requires_grad = True
                    Params[nlayer+ii*NBeta].grad = None
                    loss = forwardThird(nlayer)
                    loss.backward()

                    with torch.no_grad():
                       E = Params[nlayer+ii*NBeta].grad
                       #print('grad: ',E)
                       D1, D1, D_new = E.shape
                       #print('E.shape',E.shape)
                       E = E.view(D1**2, D_new)
                       # perform MERA update
                       U,S,V = torch.svd(E)
                       Params[nlayer+ii*NBeta].data = (U@V.t()).view(D1,D1,D_new)
                       #print('%+.15f'%(-loss.item()/2**(NBeta+1)/tau), end=' ')
                       if Dis: print('F_en=%+.10f'%(-loss.item()/2**(NBeta)/tau), end=' ')
                       #print('\nAllocated cuda memory:%d, cached:%d '%(torch.cuda.memory_allocated() / 1024**2,torch.cuda.memory_cached() / 1024**2))
                       del U,S,V,E 
                       torch.cuda.empty_cache()
                       #print('Allocated cuda memory:%d, cached:%d '%(torch.cuda.memory_allocated() / 1024**2,torch.cuda.memory_cached() / 1024**2))
                    Params[nlayer+ii*NBeta].requires_grad = False 
                    niter=niter+1
                    rel_err= (loss_old - loss.item())/loss.item()
                else:
                    with torch.no_grad():
                        loss = forwardThird(nlayer)
                    #Params[nlayer+ii*NBeta].data = torch.eye(SizeW[2],device=Params[nlayer+ii*NBeta].device).view(SizeW[0],SizeW[1],SizeW[2])
                    if Dis: print('F_en=%+.10f'%(-loss.item()/2**(NBeta)/tau), end=' ')
                    loss_old = loss.item()
            
           if niter ==0: 
               not_conv=False
               #print('No iteration, just evaluation')
           else: 
                not_conv= abs(rel_err) > loss_conv or niter <4 #and niter < self.Niter
                loss_old = loss.item()
                #if Dis and do_Diff: print('\nAfter %d iter: rel err= %E'%(niter,rel_err))
            #niter=niter+1
            #print(' ')
        #if Dis: print('Allocated cuda memory:%d, cached:%d '%(torch.cuda.memory_allocated() / 1024**2,torch.cuda.memory_cached() / 1024**2))
        #--------------------------#
        #with torch.no_grad():
        #    loss = forwardThird(nlayer)
        #--------------------------#
        return loss.item()


#============================ Main ==============================

device = torch.device("cuda:0")#cpu")#cuda") #("cpu"))
dtype = torch.float32#64  #32
#FeS = []  # init thermal quantities
print('file dHOTRGv1.py\n')   # will add different direction of energy E contribution
loss_conv=1e-5
NBeta = 36 # no. of isometries on different layers, starts from 1
#D = 10 
D = 6 
Niter = 3 
Nsweep = 5
depth = 3 

initialHOTRG=True#False # do we perform an initial HOTRG?
initial_values=True#False # run evolveMandE right after HOTRG, overwritten by initialHOTRG
dHOTRG = True#False#True 
initRanIsometry = False#True  # do we use random isometry instead ones from HOTRG?

#Ts=[4.585, 4.5852, 4.5854, 4.5856]#, 4.5858, 4.586, 4.587, 4.588, 4.589]
Ts=[4.58545]
for t in Ts:
    beta= 1.0/t
    #fname='test_save_NBeta'+str(NBeta)+'tau'+str(beta)
    fname='./Results/new_NBeta'+str(NBeta)+'D'+str(D)+'tau'+str(beta)
    from os.path import exists

    file_exists = os.path.exists(fname)

    if file_exists: initialHOTRG=False#True#True   # if file exits then read from file
    else: initialHOTRG=True#True

    tau = torch.tensor([beta], dtype=dtype, device=device)
    Wl = []; Wr = []
    RWl = []; RWr = []
    Tas = []; #Tbs = []; 
    lnZs = []; normTs = []; #Betas=[]


###################### HOTRG part ##############################
    if initialHOTRG:    
        print('tau =%.10f (T=%f), bond dim=%d, number of layers=%d '%(tau,1/tau,D,NBeta),'device=',device)
        print('================= HOTRG =======================================')
        T0= initRho3D(tau)
        #T= initRho3D(tau)#.detach().to('cpu')
        M = getM(tau)
        E = getE(tau)
        lnZ = 0.0
        for nbeta in range(0,NBeta):
            with torch.no_grad():
               #print(nbeta)
           
               iab=nbeta%3
               if iab==0:
                   T,mWl,mWr=MergX( T0, T0, D)  
               #if nbeta==0:
               #    E=rgXE1step(E,mWl,mWr)
               #else:
               #    E=rgXimp(T0,E,mWl,mWr)
               #M = rgXimp(T0,M,mWl,mWr)
               elif iab==1:
                   T,mWl,mWr=MergY( T0, T0, D)  
               #E=rgYimp(T0,E,mWl,mWr)
               #M = rgYimp(T0,M,mWl,mWr)
               else:
                   T,mWl,mWr=MergZ( T0, T0, D)  
               #E=rgZimp(T0,E,mWl,mWr)
               #M =rgZimp(T0,M,mWl,mWr)
           #T0=T#.to(device)
           #norm=torch.norm(T)
               norm=torch.norm(T)#.detach().to('cpu')
               T = T/norm
               T0 = T
               lnZ = 2*lnZ + torch.log(norm)#.detach().to('cpu')
           #M = M/norm
           #E = E/norm

               eT=getTrace3(T)#.detach().to('cpu')
               Fe=-(lnZ+torch.log(eT))/2**(nbeta+1)/tau
           
           ################### remove calculating mag and energy; do it later #############
           #eM=getTrace3(M)
           #eE=getTrace3(E)
           #mag=eM/eT
           #mE=-3*eE/eT
           #entropy= (mE-Fe)*tau
           #print('[layer=%d, eM=%f, eT=%f, mag=%f, E=%f, Fe=%f, Entropy=%f]'%(nbeta,eM,eT,mag,mE,Fe, entropy))
               print('[layer=%d, eT=%f, Fe=%.15f]'%(nbeta,eT,Fe))

               # do we initialize W's using HOTRG or random? when D2==Dl*Dl, we just use identity
               Wl.append(mWl.contiguous().detach().to('cpu'))
               Wr.append(mWr.contiguous().detach().to('cpu'))

               #### create random isometry if needed #####
               if initRanIsometry == True:
                  SizeW=list(mWl.size())
                  Dl=SizeW[0]
                  D2=SizeW[2]
                  if D2==Dl*Dl:
                      tWl=torch.eye(D2,dtype=dtype,device='cpu').view(Dl,Dl,D2)
                      RWl.append(tWl)
                  else: 
                      tWl=torch.randn(Dl*Dl,D2,dtype=dtype,device=device)
                      U,S,V = torch.svd(tWl)
                      tWl = (U@V.t()).view(Dl,Dl,D2)
                      tWl.detach().to('cpu')
                      RWl.append(tWl.contiguous().detach().to('cpu'))
                      del tWl
                  SizeW=list(mWr.size())
                  Dl=SizeW[0]
                  D2=SizeW[2]
                  if D2==Dl*Dl:
                      tWr=torch.eye(D2,dtype=dtype,device='cpu').view(Dl,Dl,D2)
                      RWr.append(tWr)
                  else: 
                      tWr=torch.randn(Dl*Dl,D2,dtype=dtype,device=device)
                      U,S,V = torch.svd(tWr)
                      tWr = (U@V.t()).view(Dl,Dl,D2)
                      tWr.detach().to('cpu')
                      RWr.append(tWr.contiguous().detach().to('cpu'))
                      del tWr

               # One can also initialize Tensors at all layers: (commented out)
               #if len(Tas)<nbeta+1:  
               #    Tas.append(T.detach().to('cpu'))
               #    normTs.append(norm.detach().to('cpu'))
               #else: 
               #    Tas[nbeta]=T.detach().to('cpu')
               #    normTs[nbeta]=norm.detach().to('cpu')
               #if len(lnZs)<nbeta+1: lnZs.append(lnZ.detach().to('cpu'))
               #else:                 lnZs[nbeta]=lnZ.detach().to('cpu')

           #print('Allocated cuda memory:%d, reserved:%d '%(torch.cuda.memory_allocated() / 1024**2,torch.cuda.memory_reserved() / 1024**2))
           #del mWl, mWr, T
               del mWl, mWr
               torch.cuda.empty_cache()
           #print('Allocated cuda memory:%d, reserved:%d '%(torch.cuda.memory_allocated() / 1024**2,torch.cuda.memory_reserved() / 1024**2))
#del M, E, T0
#del T0
        lnZ=lnZ+torch.log(eT)
        Fe=-lnZ/2**(NBeta)/tau
        #print('*** Final Free energy= %+.15f'%(Fe))
#print('eM=%.10f, eT=%.10f, mag=%.10f, E=%.10f, Fe=%.10f, Entropy=%.10f]'%(eM,eT,mag,mE,Fe, entropy))
        #torch.cuda.empty_cache()


#print('Allocated cuda memory:%d, cached:%d '%(torch.cuda.memory_allocated() / 1024**2,torch.cuda.memory_reserved() / 1024**2))

        ####### initialize `left' and 'right' isometries  e.g. Wxy and Wxz from HOTRG
        Params = torch.nn.ParameterList([torch.nn.Parameter(_.to(device)) for _ in Wl+Wr])
        Params.append(torch.nn.Parameter(tau))  # the last time is tau (inverse T)
#print('Allocated cuda memory:%d, cached:%d '%(torch.cuda.memory_allocated() / 1024**2,torch.cuda.memory_reserved() / 1024**2))


        if initial_values: 
            with torch.no_grad():
                Fe0,mag0,mE0,entropy0=evolveMandE(tau,NBeta)
                print('Initial after HOTRG: mag=%.10f, E=%.10f, Fe=%.10f, Entropy=%.10f]'%(mag0,mE0,Fe0, entropy0))
        ####### initialize `left' and 'right' isometries  e.g. Wxy and Wxz from random (overwriting HOTRG) 
        if initRanIsometry == True:
           Params = torch.nn.ParameterList([torch.nn.Parameter(_.to(device)) for _ in RWl+RWr])
           Params.append(torch.nn.Parameter(tau))  # the last time is tau (inverse T)
    else:
      if file_exists:   # assume there are HOTRG results and isometry saved
        print('\nLoading from file: ',fname)
        myload=torch.load(fname)
        print('beta= ',myload['beta'],' # of layers=',myload['NBeta'])
        Fe0=myload['Fe0']
        mag0=myload['mag0']
        mE0=myload['mE0']
        entropy0=myload['entropy0']
        Params=myload['Params']
        print('Previous HOTRG results: mag=%.10f, E=%.10f, Fe=%.10f, Entropy=%.10f]'%(mag0,mE0,Fe0, entropy0))
        
    #quit() #--------If you want to continue dHOTRG (second RG), comment this line------------------------------------    


    if dHOTRG == True:
        print('\n=========================== dHOTRG: re-optimiztion ======================')
        #loss_temp=loss;
        if initialHOTRG: 
            loss_temp=lnZ;
            nswp=0;
        else:
            loss_temp=0
            nswp=myload['nswp']
        not_conv=True;
        iswp=0

        while(not_conv):
            print('------------- forward ---------')
            for nlayer in range(0,NBeta):
               loss = update_single_layerThird(nlayer)
               #print(' ')
            print('------------- backward ---------')
 
            for nlayer in range(NBeta-1,-1,-1):
               loss = update_single_layerThird(nlayer)

            rel_err= (loss_temp-loss)/loss
            not_conv= abs(rel_err)>1e-8 #or nswp<40 # and nswp <model.Niter
            loss_temp=loss
            nswp=nswp+1
            iswp=iswp+1
            if(iswp%2==0):
               torch.save({'mag0':mag0,'mE0':mE0,'Fe0':Fe0,'entropy0':entropy0,\
                'beta':beta, 'NBeta':NBeta,'D':D,'nswp':nswp,\
                'Params':Params,'dtype':dtype,'device':device},fname)

        #========= another optimization to last outer layer ============
        to_lastlayer=True;
        if to_lastlayer: 
            for nlayer in range(0,NBeta):
               loss = update_single_layerThird(nlayer)
               #print(' ')
            rel_err= (loss_temp-loss)/loss
        print('==>[re-opt] Took nswp=%d sweeps; relative error=%E'%(nswp,rel_err))
        Wl = [m.detach().to('cpu') for m in Params[:NBeta]]
        Wr = [m.detach().to('cpu') for m in Params[NBeta:-1]]
        with torch.no_grad():
             Fe,mag,mE,entropy=evolveMandE(tau,NBeta)
        print('Recall: tau =%.10f (T=%f), bond dim=%d, number of layers=%d '%(tau,1/tau,D,NBeta),'device=',device)
        if initial_values: 
            print('Initial after HOTRG: mag=%.10f, E=%.10f, Fe=%.10f, Entropy=%.10f]'%(mag0,mE0,Fe0, entropy0))
        print('Final:               mag=%.10f, E=%.10f, Fe=%.10f, Entropy=%.10f]'%(mag,mE,Fe, entropy))
        torch.save({'mag0':mag0,'mag':mag,'mE0':mE0,'mE':mE,'Fe0':Fe0,'Fe':Fe,'entropy0':entropy0,\
               'entropy':entropy,'beta':beta, 'NBeta':NBeta,'D':D,'nswp':nswp,\
               'Params':Params,'dtype':dtype,'device':device},fname)


        del Params
        torch.cuda.empty_cache()
