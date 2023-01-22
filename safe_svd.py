import torch
import functorch

# credits https://github.com/wangleiphy/tensorgrad/blob/master/tensornets/adlib/svd.py
def safe_inverse(x, epsilon=1E-8):
    #epsilon=epsilon*torch.max(torch.abs(x))
    return x/(x**2 + epsilon)

class SVD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        U, S, Vt = torch.linalg.svd(A,full_matrices =False)
        ctx.save_for_backward(U, S, Vt)
        return U, S, Vt

    

    @staticmethod
    def backward(ctx, dU, dS, dVt):#here dU means df/dU, dA means df/dA
        U, S, Vt = ctx.saved_tensors
        V = Vt.t().conj()
        Ut = U.t().conj()
        M = U.size(0)
        N = V.size(0)
        NS = len(S)
        
        Sinv=safe_inverse(S)
        #assert Sinv.isfinite().all()

        F = (S - S[:, None])
        F = safe_inverse(F)
        F.diagonal().fill_(0)
        #assert F.isfinite().all()

        G = (S + S[:, None])
        G = safe_inverse(G)
        G.diagonal().fill_(0)
        #assert G.isfinite().all()

        UdU = Ut @ dU
        VdV = Vt @ (dVt.t().conj())

        Su = (F+G)*(UdU-UdU.t().conj())/2
        Sv = (F-G)*(VdV-VdV.t().conj())/2
        
        #assert dS.isfinite().all()
        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt 
        #assert dA.isfinite().all()
        if (M>NS):
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU*Sinv) @ Vt 
        if (N>NS):
            dA = dA + (U*Sinv) @ dVt @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        #assert dA.isfinite().all()
        return dA
svd = SVD.apply

#torch.manual_seed(2)
#input = torch.rand(20, 16, dtype=torch.float64, requires_grad=True)
#assert torch.autograd.gradcheck(svd, input, eps=1e-6, atol=1e-4)

class SQRT(torch.autograd.Function):
    @staticmethod
    def forward(ctx,A):
        As=A.sqrt()
        ctx.save_for_backward(As)
        return As
    @staticmethod
    def backward(ctx,dAs):
        As=ctx.saved_tensors[0]
        dA=safe_inverse(2*As)*dAs
        #assert dA.isfinite().all()
        return dA
sqrt=SQRT.apply

    

# def split_matrix(M,max_dim=None):
#     u,s,vh=svd(M)
#     s=torch.diag(s**.5)
#     u,vh=u@s,s@vh
#     if max_dim is not None:
#         u,vh=u[:,:max_dim],vh[:max_dim,:]
#     return u,vh

if __name__=='__main__':
    torch.manual_seed(2)
    print('testing autograd')
    input = torch.rand(20, 16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(sqrt, input, eps=1e-6, atol=1e-4)
    assert torch.autograd.gradcheck(svd, input, eps=1e-6, atol=1e-4)
    print('testing jvp')
    from functorch import jvp
    value, grad=jvp(svd,(input,),(torch.randn_like(input),))

    print('passed')



__all__=['svd','sqrt']