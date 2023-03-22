import mpmath
import numpy as np
from functools import wraps

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

# for fitting in the loglog space
def loglog(foo):
    @wraps(foo)
    def goo(logX,*p):
        return np.log(foo(np.exp(logX),*p))
    return goo