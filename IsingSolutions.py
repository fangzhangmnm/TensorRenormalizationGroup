import numpy as np
import matplotlib.pyplot as plt
from numpy import cosh,sinh,sqrt,cos,pi,log
import scipy.integrate


def relerr(x,ref):
    return abs(x-ref)/abs(ref)

def abserr(x,ref):
    return abs(x-ref)

class Ising2DExactSolution:
    def __init__(self):
        self.critical_beta=log(1+sqrt(2))/2
    def logZ(self,beta):
        @np.vectorize
        def _logZ(beta):
            K,L=beta,beta
            k1=cosh(2*K)*cosh(2*L)
            k2=sinh(2*K)*sinh(2*L)
            def integrant(th):
                return log(k1+sqrt(1+k2*k2-2*k2*cos(2*th)))
            I,abserr=scipy.integrate.quad(integrant,0,pi,epsabs=0,epsrel=1e-12)
            I,abserr=I/(2*pi)+log(2)/2,abserr/(2*pi)
            return I
        return _logZ(beta)
    def ddlogZ(self,beta):
        @np.vectorize
        def _ddlogZ(beta):
            ch,sh,ch4,ch8,sh4=cosh(2*beta),sinh(2*beta),cosh(4*beta),cosh(8*beta),sinh(4*beta)
            chch,shsh=ch*ch,sh*sh
            I1=8*(chch+shsh)
            def integrant(th):
                ct=cos(2*th)
                D=1-2*ct*shsh+shsh*shsh
                sD=sqrt(D)
                I3=-4*(1+2*ct)*ch4+4*ch8
                I4=(-1-2*ct+ch4)*sh4
                f=chch+sD
                df=2*sh4+I4/sD
                ddf=I1+I3/sD-I4**2/sD**3
                return (-df*df/f+ddf)/f
            I,abserr=scipy.integrate.quad(integrant,0,pi,epsabs=0,epsrel=1e-12)
            I,abserr=I/(2*pi),abserr/(2*pi)
            return I
        return _ddlogZ(beta)
        
    def magnetization (self,beta):
        return np.maximum(1-sinh(2*beta)**-4,0)**(1/8)
    def display(self):
        beta=np.linspace(0.1,1,500)
        plt.subplot(221).plot(beta,self.logZ(beta))
        plt.subplot(221).title.set_text('logZ')
        plt.subplot(222).plot(beta,self.magnetization(beta))
        plt.subplot(222).title.set_text('magnetization')
        plt.subplot(223).plot(beta,self.ddlogZ(beta))
        plt.subplot(223).title.set_text('ddlogZ')
        plt.suptitle('Ising2D Expected values')
        plt.show()
        print('critical_beta',self.critical_beta)
     
ising2d=Ising2DExactSolution()

#class FromCriticalExponent:
    

        
class Ising3DMonteCarlo:
    # credits https://arxiv.org/pdf/cond-mat/9603013.pdf
    def __init__(self):
        self.critical_beta=0.2216544
    def magnetization (self,beta):
        beta_exponent=0.3269
        theta=0.508
        a0=1.692
        a1=0.344
        a2=0.426
        beta=np.array(beta)
        with np.errstate(all='ignore'):
            t=(beta-self.critical_beta)/beta
            return np.where(beta>self.critical_beta,(a0-a1*(t**theta)-a2*t)*(t**beta_exponent),0)
    def logZ(self,beta):
        return beta*0
    def ddlogZ(self,beta):
        return beta*0
    def display(self):
        beta=np.linspace(0,0.5)
        plt.plot(beta,self.magnetization(beta))
        plt.title('magnetization')
        plt.suptitle('Ising3D Expected values')
        plt.show()
        print('critical_beta',self.critical_beta)
       

ising3d=Ising3DMonteCarlo()

__all__=['relerr','abserr','ising2d','ising3d']

if __name__=="__main__":
    ising2d.display()