import torch
import os
import numpy as np
torch.set_default_tensor_type(torch.cuda.DoubleTensor)
torch.cuda.set_device(1)

from linearized import mysvd, myeigh, get_linearized_HOTRG_autodiff, get_linearized_HOTRG_full_autodiff, get_linearized_cylinder, verify_linear_operator, check_hermicity
from ScalingDimensions import get_scaling_dimensions


filename='data/transfer_matrix_tnr_X16_I20'


options={
    'tensor_path':'data/tnr_X16_L20.pkl',
    
    'iLayer':20,
    'svd_max_iter':100,
    'svd_tol':1e-16,
    'svd_num_eigvecs':16,
}


if os.path.exists(filename+'_options.pkl'):
    _options=torch.load(filename+'_options.pkl')
    if not(options==_options):
        def tryRemove(filename):
            if os.path.exists(filename):
                os.remove(filename)
        tryRemove(filename+'_options.pkl')
        tryRemove(filename+'_eigs_cyl.pkl')

torch.save(options,filename+'_options.pkl')
print('file saved: ',filename+'_options.pkl')



layers,Ts,logTotals=torch.load(options['tensor_path'])

iLayer=options['iLayer']
T=Ts[iLayer]
assert T.shape[0]==options['max_dim']


Mc=get_linearized_cylinder(T)
check_hermicity(Mc,nTests=1)
verify_linear_operator(Mc,nTests=1)

uc,sc,_=mysvd(Mc,k=options['svd_num_eigvecs'],tol=options['svd_tol'],maxiter=options['svd_max_iter'])
uc,sc=torch.tensor(uc),torch.tensor(sc)


print(options)
print('scaling dimensions from Transfer Matrix on a Cylinder')
print(get_scaling_dimensions((sc),scaling=np.exp(2*np.pi/4)))
           
torch.save((sc,uc),filename+'_eigs_cyl.pkl')
print('file saved: ',filename+'_eigs_cyl.pkl')
