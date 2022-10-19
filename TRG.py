import torch
from opt_einsum import contract
from safe_svd import svd,sqrt # TODO is it necessary???

def NWSE(T):
    return contract('nswe->nwse',T).reshape(T.shape[0]*T.shape[2],-1)
def NESW(T):
    return contract('nswe->nesw',T).reshape(T.shape[0]*T.shape[3],-1)

def split_NWSE(A,max_dim):
    #   0     B       0   2
    #  2A3 ->  \     1B    C1
    #   1       C      2   0
    u,s,vh=svd(NWSE(A)) # svd: U@diag(S)@Vh==A, descending
    s1=sqrt(s).diag()
    B=(u@s1).reshape(A.shape[0],A.shape[2],-1)[:,:,:max_dim]
    C=(vh.T@s1).reshape(A.shape[1],A.shape[3],-1)[:,:,:max_dim]
    return B,C
    
def split_NESW(B,max_dim):
    return split_NWSE(contract('nswe->nsew',B),max_dim)
    
def TRG_AB(A,B,max_dim):
    #   0    A1              B1   A2--B2  0 3
    #  2A3 ->  \     B ->   /     | O |->  T
    #   1       A2        B2      B1--A1  2 1
    Anw,Ase=split_NWSE(A,max_dim)
    Bne,Bsw=split_NESW(B,max_dim)
    return contract('cax,daw,cbz,dby->xyzw',Ase,Bsw,Bne,Anw)
        
def TRG_ABCD(A,B,C,D,max_dim):
    #  A---B   0  3  D---C   0  3
    #  | O | -> T1   | O | -> T2
    #  C---D   2  1  B---A   2  1
    Anw,Ase=split_NWSE(A,max_dim)
    Bne,Bsw=split_NESW(B,max_dim)
    Cnw,Cse=split_NWSE(C,max_dim)
    Dne,Dsw=split_NESW(D,max_dim)
    T1=contract('cax,daw,cbz,dby->xyzw',Ase,Bsw,Cne,Dnw)
    T2=contract('cax,daw,cbz,dby->xyzw',Dse,Csw,Bne,Anw)
    return T1,T2