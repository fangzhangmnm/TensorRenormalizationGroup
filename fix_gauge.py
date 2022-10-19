import torch
from opt_einsum import contract
from safe_svd import svd,sqrt # TODO is it necessary???




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

def fix_phase(T,Tref):
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
    T=fix_phase(T,Tref)
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