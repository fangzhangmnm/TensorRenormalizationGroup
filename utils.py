import matplotlib.pyplot as plt
import matplotlib.colors
from opt_einsum import contract
from colorsys import hls_to_rgb
import numpy as np

def complex_array_to_rgb(X, theme='dark', rmax=None):
    '''Takes an array of complex number and converts it to an array of [r, g, b],
    where phase gives hue and saturaton/value are given by the absolute value.
    Especially for use with imshow for complex plots.'''
    absmax = rmax or np.abs(X).max()
    Y = np.zeros(X.shape + (3,), dtype='float')
    Y[..., 0] = np.angle(X) / (2 * np.pi) % 1
    if theme == 'light':
        Y[..., 1] = np.clip(np.abs(X) / absmax, 0, 1)
        Y[..., 2] = 1
    elif theme == 'dark':
        Y[..., 1] = 1
        Y[..., 2] = np.clip(np.abs(X) / absmax, 0, 1)
    Y = matplotlib.colors.hsv_to_rgb(Y)
    return Y

def show_tensor_ijklmn(T,max_dim=None):
    d0,d1,d2,d3,d4,d5=np.minimum(T.shape,max_dim) if max_dim is not None else T.shape
    plt.imshow(complex_array_to_rgb(contract('ijklmn->ijklmn',T[:d0,:d1,:d2,:d3,:d4,:d5]).reshape(d0*d1*d2,d3*d4*d5)))
    plt.ylabel('i,j,k')
    plt.xlabel('l,m,n')

def show_tensor_ikjl(T,max_dim=None):
    d0,d1,d2,d3=np.minimum(T.shape,max_dim) if max_dim is not None else T.shape
    plt.imshow(complex_array_to_rgb(contract('ijkl->ikjl',T[:d0,:d1,:d2,:d3]).reshape(d0*d2,d1*d3)))
    for i in range(d0+1):
        plt.axhline(i*d2-.5,color='white')
    for i in range(d1+1):
        plt.axvline(i*d3-.5,color='white')
    plt.ylabel('i,k')
    plt.xlabel('j,l')
    
def show_tensor_ijkl(T,max_dim=None):
    d0,d1,d2,d3=np.minimum(T.shape,max_dim) if max_dim is not None else T.shape
    plt.imshow(complex_array_to_rgb(contract('ijkl->ijkl',T[:d0,:d1,:d2,:d3]).reshape(d0*d1,d2*d3)))
    for i in range(d0+1):
        plt.axhline(i*d1-.5,color='white')
    for i in range(d2+1):
        plt.axvline(i*d3-.5,color='white')
    plt.ylabel('i,j')
    plt.xlabel('k,l')
    
def show_tensor_ij_k(T,max_dim=None):
    d0,d1,d2=np.minimum(T.shape,max_dim) if max_dim is not None else T.shape
    plt.imshow(complex_array_to_rgb(T[:d0,:d1,:d2].reshape(d0*d1,d2)))
    for i in range(d0+1):
        plt.axhline(i*d1-.5,color='white')
    plt.ylabel('i,j')
    plt.xlabel('k') 
    
def show_tensor_i_jk(T,max_dim=None):
    d0,d1,d2=np.minimum(T.shape,max_dim) if max_dim is not None else T.shape
    plt.imshow(complex_array_to_rgb(T[:d0,:d1,:d2].reshape(d0,d1*d2)))
    for i in range(d1+1):
        plt.axvline(i*d2-.5,color='white')
    plt.ylabel('i')
    plt.xlabel('j,k') 
        
def show_matrix(T,max_dim=None):
    d0,d1=np.minimum(T.shape,max_dim) if max_dim is not None else T.shape
    plt.imshow(complex_array_to_rgb(T[:d0,:d1]))
    plt.ylabel('i')
    plt.xlabel('j')
    
def show_hist(s):
    plt.hist(s,bins=20)
    plt.yscale('log')