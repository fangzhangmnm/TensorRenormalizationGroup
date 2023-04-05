import mpmath
import numpy as np
from scipy.special import hyp2f1
from functools import wraps

# for fitting in the loglog space
def loglog(foo):
    @wraps(foo)
    def goo(logX,*p):
        return np.log(foo(np.exp(logX),*p))
    return goo

def critical_correlation(x,A,delta): #critical temperature
    return A*x**(-2*delta)
critical_correlation.eq='{0:.2e}x^(-2 Δ),Δ={1:.4f}'
critical_correlation.p0=(1,0.1)

def high_temp_correlation(x,A,zeta,delta): #high temperature
    return A*np.exp(-x/zeta)*x**(-2*delta)
high_temp_correlation.eq='{0:.2e}e^(-x/ζ)/x^(2 Δ),ζ={1:.2e},Δ={2:.4f}'
high_temp_correlation.p0=(1,100,0.1)

# def low_temp_correlation3(x,A,zeta,delta,n): #low temperature
#     return A*(1+(zeta/x)**n)**(2*delta/n)
# low_temp_correlation3.eq='{0:.2e}(1+(ζ/x)^n)^(2 ∆/n),ζ={1:.2e},∆={2:.4f},n={3:.1f}'
# low_temp_correlation3.p0=(1,100,0.1,2)

def low_temp_correlation_1(x,A,zeta,delta):  #low temperature
    return A*(1/zeta+1/x)**(2*delta)
low_temp_correlation_1.eq='{0:.2e}(1/ζ+1/x)^(2 ∆),ζ={1:.2e},∆={2:.4f}'
low_temp_correlation_1.p0=(1,100,0.1)

def low_temp_correlation(x,A,zeta,delta,m0):  #low temperature nikko
    return A*np.exp(-x/zeta)*x**(-2*delta)+m0**2
low_temp_correlation.eq='{0:.2e}e^(-x/ζ)/x^(2 Δ)+m0^2,ζ={1:.2e},Δ={2:.4f},m0={3:.3f}'
low_temp_correlation.p0=(1,1000,0.1,0.1)



jtheta = np.vectorize(mpmath.jtheta, 'D') #https://mpmath.org/doc/current/functions/elliptic.html#jacobi-theta-functions
def torus_correlation(x,y,sizeX,sizeY):
    tau=1j * sizeY/sizeX
    # compare to the big yellow book: 
    #     q_y=e^(2ipi tau)=q^2, z_y=z/pi
    z=np.pi*(x+1j * y)/sizeX
    q=np.exp(1j*np.pi*tau) 
    coeff=np.abs(jtheta(n=1,z=0,q=q,derivative=1)/jtheta(n=1,z=z,q=q))**0.25
    A=sum(np.abs(jtheta(n=n,z=z/2,q=q))for n in [1,2,3,4])
    B=sum(np.abs(jtheta(n=n,z=0,q=q))for n in [2,3,4])
    return coeff*A/B

def get_torus_correlation_ansatz(lattice_size):
    def torus_correlation_ansatz(x,A):  #torus
        return A*torus_correlation(x,0,*lattice_size)
    torus_correlation_ansatz.eq=r'🍩(x,0)'
    torus_correlation_ansatz.eq='{0:.3f}'+torus_correlation_ansatz.eq.replace('{','{{').replace('}','}}')
    torus_correlation_ansatz.p0=(1)
    return torus_correlation_ansatz

def three_point_correlation(x,C012,delta01,delta02,delta12):
    dist01,dist02,dist12=x
    return C012/(dist01**delta01*dist02**delta02*dist12**delta12)
three_point_correlation.eq='C_012/(dist01^Δ01*dist02^Δ02*dist12^Δ12),C012={0:.2e},Δ01={1:.4f},Δ02={2:.4f},Δ12={3:.4f}'
three_point_correlation.p0=(1,0.1,0.1,0.1)


# arXiv:1602.07982v1 sec 9 

def blockK(x,beta):
    # eq 168
    return x**(beta/2)*hyp2f1(beta/2,beta/2,beta,x)

def blockG(z,delta,l):
    # eq 166
    zbar=z.conjugate()
    return blockK(z,delta+l)*blockK(zbar,delta-l)+blockK(zbar,delta+l)*blockK(z,delta-l)

def crossRatioZ(x12,x13,x14,x23,x24,x34):
    # eq 66
    u=x12**2*x34**2/(x13**2*x24**2)
    v=x23**2*x14**2/(x13**2*x24**2)
    # eq 67
    # Solve[{u == x^2 + y^2, v == 1 - 2 x + x^2 + y^2}, {x, y}]
    x=.5*(1+u-v)
    yy=.25*(-1+2*u+2*v-u**2-v**2+2*u*v)
    assert np.all(yy>=-1e-10)
    y=np.sqrt(yy)
    return x+1j*y


def four_point_channel(x12,x13,x14,x23,x24,x34,delta,deltaO,lO,C001):
    # eq 158
    z=crossRatioZ(x12,x13,x14,x23,x24,x34)
    partial_correlation=C001**2/(x12**(2*delta)*x34**(2*delta))*blockG(z,deltaO,lO)
    assert np.all(np.abs(np.imag(partial_correlation))<1e-10)
    partial_correlation=np.real(partial_correlation)
    return partial_correlation

def ising_four_sigma_correlation(dists,Anorm,deltaSigma,deltaEpsilon,CdeltadeltaEpsilon):
    x12,x13,x14,x23,x24,x34=dists
    rtval=0
    rtval+=four_point_channel(x12,x13,x14,x23,x24,x34,delta=deltaSigma,deltaO=0,lO=0,C001=1)
    rtval+=four_point_channel(x12,x13,x14,x23,x24,x34,delta=deltaSigma,deltaO=deltaEpsilon,lO=0,C001=CdeltadeltaEpsilon)
    return Anorm*rtval
ising_four_sigma_correlation.eq='A={0:.2e},Δσ={1:.4f},Δε={2:.4f},C(σσε)={3:.4f}'
ising_four_sigma_correlation.p0=(1,0.1,1.1,0.4)
ising_four_sigma_correlation.bounds=((1e-50,.01,.01,.01),(1e50,2,2,2))


