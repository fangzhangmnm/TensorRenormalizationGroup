import jax
import jax.numpy as np


# credits https://github.com/wangleiphy/tensorgrad/blob/master/tensornets/adlib/svd.py
def safe_inverse(x, epsilon=1E-8):
    #epsilon=epsilon*torch.max(torch.abs(x))
    return x/(x**2 + epsilon)

@jax.custom_jvp
def svd(A):
    u,s,vh=np.linalg.svd(A,full_matrices=True)
    return u,s,vh

assert False, 'jvp not implemented'

# @svd.defjvp
# def svd_jvp(primals, tangents):
#     print('foo')
#     print([x.shape for x in primals])
#     print([x.shape for x in tangents])
#     A, = primals
#     u,s,vh=svd(A)
#     du,ds,dvh, = tangents
#     M=u.shape[0]
#     N=vh.shape[0]
#     NS=s.shape[0]

#     Sinv=safe_inverse(s)
#     #assert Sinv.isfinite().all()

#     F = (s - s[:, None])
#     F = safe_inverse(F)
#     F.diagonal().fill_(0)
#     #assert F.isfinite().all()

#     G = (s + s[:, None])
#     G = safe_inverse(G)
#     G.diagonal().fill_(0)
#     #assert G.isfinite().all()

#     udu=u.t()@du
#     vdv=vh@dvh.t().conj()
#     #assert dS.isfinite().all()

#     su = (F+G)*(udu-udu.t().conj())/2
#     sv = (F-G)*(vdv-vdv.t().conj())/2

#     dA = u @ (su + sv + np.diag(ds)) @ vh
#     #assert dA.isfinite().all()

#     if (M>NS):
#         dA = dA + (np.eye(M) - u@u.t().conj()) @ (du*Sinv) @ vh 
#     if (N>NS):
#         dA = dA + (u*Sinv) @ dvh @ (np.eye(N) - vh.t().conj()@vh)
#     #assert dA.isfinite().all()
#     return dA



if __name__=='__main__':
    from jax.test_util import check_grads
    print('testing autograd')
    A=jax.random.normal(jax.random.PRNGKey(0),shape=(20,16))
    check_grads(svd, (A,), order=1)
    print('passed')




__all__=['svd']

