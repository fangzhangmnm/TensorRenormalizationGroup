import torch
from opt_einsum import contract
from safe_svd import svd,sqrt # TODO is it necessary???
from tqdm.auto import tqdm




#def fix_phase_ij(T):
#    # makes the component of Ti??? with largest abs positive
#    TT=T.reshape(T.shape[0],-1)
#    di=torch.where(torch.max(TT,dim=1).values>=-torch.min(TT,dim=1).values,1,-1)
#    
#    T=contract('ijkl,i,j->ijkl',T,di,di)
#    return T
#
#fix_gauge_phase_iter=2
#
#def fix_phase(T,Tref):
#    for j in range(fix_gauge_phase_iter):
#        T=fix_phase_ij(T)
#        T=T.permute(2,3,0,1)
#        T=fix_phase_ij(T)
#        T=T.permute(2,3,0,1)
#    return T



#def fix_phase(T,Tref):
#    for j in range(1):
#        di=contract('ijkl,ijkl->i',T,Tref).sign()
#        T=contract('ijkl,i,j->ijkl',T,di,di)
#        
#        di=contract('ijkl,ijkl->k',T,Tref).sign()
#        T=contract('ijkl,k,l->ijkl',T,di,di)
#        
#        di=contract('ijkl,ijkl->j',T,Tref).sign()
#        T=contract('ijkl,i,j->ijkl',T,di,di)
#        
#        di=contract('ijkl,ijkl->l',T,Tref).sign()
#        T=contract('ijkl,k,l->ijkl',T,di,di)
#    return T

def fix_gauge_ij(T):
    #      vi     0
    #    --T--   2T3
    #      v      1
    A=contract('ijkl,Ijkl->iI',T,T.conj())
    u,s,vh=svd(A) # I don't know what I'm doing but it works better than eig in ensuring sign-fixing
    
    v,vi=u,u.T
    T=contract('ijkl,Ii,jJ->IJkl',T,vi,v)
    return T,vi

def fix_phase_2D(T,Tref):
    if Tref[0,0,0,0]<0:Tref=-Tref
    if T[0,0,0,0]<0:T=-T
    for i in range(1,T.shape[0]):
        TT=T[:,:i,:i,:i]
        TTref=Tref[:,:i,:i,:i]
        di=torch.where(contract('ijkl,ijkl->i',TT,TTref)>=0,1,-1)
        T=contract('ijkl,i,j->ijkl',T,di,di)
        
        TT=T[:i,:i,:,:i]
        TTref=Tref[:i,:i,:,:i]
        di=torch.where(contract('ijkl,ijkl->k',TT,TTref)>=0,1,-1)
        T=contract('ijkl,k,l->ijkl',T,di,di)
    return T

def fix_gauge_2D(T,Tref):
    T,u1=fix_gauge_ij(T)
    T=T.permute(2,3,0,1)
    T,u2=fix_gauge_ij(T)
    T=T.permute(2,3,0,1)
    T=fix_phase_2D(T,Tref)
    return T,[u1,u2]

def fix_gauges(Ts:'list[torch.Tensor]',is_HOTRG=False):
    Ts=Ts.copy()
    spacial_dim=len(Ts[0].shape)//2
    stride=spacial_dim if is_HOTRG else 1
    for i in range(stride,len(Ts)):
        if Ts[i].shape==Ts[i-stride].shape:
            if spacial_dim==2:
                Ts[i],uu=fix_gauge_2D(Ts[i],Ts[i-stride])
            else:
                raise NotImplementedError
    return Ts
    
def contract_all_legs(T1,T2:torch.Tensor)->torch.Tensor:
    T1i,T2i=[*range(len(T1.shape))],[*range(len(T2.shape))]
    return contract(T1,T1i,T2,T2i)

def contract_all_legs_but_one(T1,T2:torch.Tensor,i:int)->torch.Tensor:
    T1i,T2i=[*range(len(T1.shape))],[*range(len(T2.shape))]
    T1i[i],T2i[i]=-1,-2
    return contract(T1,T1i,T2,T2i,[-1,-2])

def sum_all_legs_but_one(T,i:int)->torch.Tensor:
    Ti=[*range(len(T.shape))]
    return contract(T,Ti,[i])

def apply_matrix_to_leg(T:torch.Tensor,M:torch.Tensor,i:int)->torch.Tensor:
    Ti,Mi=[*range(len(T.shape))],[-1,i]
    Tni=Ti.copy();Tni[i]=-1
    return contract(T,Ti,M,Mi,Tni)

def fix_phase1_2D(T):
    if T[0,0,0,0]<0:T=-T
    for i in range(1,max(T.shape)):
        if i<T.shape[0]:
            TT=T[:,:i,:i,:i]
            di=torch.where(contract('ijkl->i',TT)>=0,1.,-1.)
            T=contract('ijkl,i,j->ijkl',T,di,di)
        if i<T.shape[2]:
            TT=T[:i,:i,:,:i]
            di=torch.where(contract('ijkl->k',TT)>=0,1.,-1.)
            T=contract('ijkl,k,l->ijkl',T,di,di)
    return T


def minimal_canonical_form(T:torch.Tensor,eps=1e-6,max_iter=50):
        # The minimal canonical form of a tensor network
        # https://arxiv.org/pdf/2209.14358.pdf
    spacial_dim=len(T.shape)//2
    gg=[torch.eye(T.shape[i]) for i in range(spacial_dim*2)]
    for iIter in range(max_iter):
        total_diff=0
        for k in range(spacial_dim):
            tr_rho=contract_all_legs(T,T.conj())
            rho1=contract_all_legs_but_one(T,T.conj(),2*k)
            rho2=contract_all_legs_but_one(T,T.conj(),2*k+1)
            rho_diff=rho1-rho2.T.conj()
            total_diff+=rho_diff.norm()/tr_rho
            g1=torch.matrix_exp(-rho_diff/(4*spacial_dim*tr_rho))
            g2=torch.matrix_exp(rho_diff/(4*spacial_dim*tr_rho))
            gg[2*k]=g1@gg[2*k]
            gg[2*k+1]=g2@gg[2*k+1]
            T=apply_matrix_to_leg(T,g1,2*k)
            T=apply_matrix_to_leg(T,g2,2*k+1)
        if total_diff<eps:break
    return T,gg





def fix_gauges1(Ts:'list[torch.Tensor]',is_HOTRG=False):
    Ts=Ts.copy()
    spacial_dim=len(Ts[0].shape)//2
    stride=spacial_dim if is_HOTRG else 1
    for i in tqdm(range(len(Ts)),leave=False):
        Ts[i],_=minimal_canonical_form(Ts[i])
        if i>=stride and Ts[i].shape==Ts[i-stride].shape:
            #Ts[i],_=fix_unitary_gauge(Ts[i],Ts[i-stride])
            if spacial_dim==2:
                Ts[i]=fix_phase_2D(Ts[i],Ts[i-stride])
            else:
                raise NotImplementedError
    return Ts