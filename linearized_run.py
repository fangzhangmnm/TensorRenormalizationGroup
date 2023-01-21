if __name__ != '__main__':
    assert False, 'This file is not meant to be imported'

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--filename', type=str, required=True) # data/tnr_X16_L10
parser.add_argument('--tensor_path', type=str, required=True) # data/tnr_X16_tensors.pkl
parser.add_argument('--iLayer', type=int, required=True) # 10
parser.add_argument('--linearized_full', action='store_true')
parser.add_argument('--gilt_enabled', action='store_true')
parser.add_argument('--gilt_eps', type=float, default=8e-7)
parser.add_argument('--gilt_nIter', type=int, default=1)
parser.add_argument('--mcf_enabled', action='store_true')
parser.add_argument('--mcf_eps', type=float, default=1e-16)
parser.add_argument('--mcf_max_iter', type=int, default=20)
parser.add_argument('--svd_method', type=str, default='eigs', choices=['svds','eigs','eigsh','mysvd','myeig_old'])
parser.add_argument('--svd_max_iter', type=int, default=200)
parser.add_argument('--svd_tol', type=float, default=1e-7)
parser.add_argument('--svd_num_eigvecs', type=int, default=16)
parser.add_argument('--svd_sanity_check', action='store_true')
parser.add_argument('--version', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()
options=vars(args)


print('loading library...')

from opt_einsum import contract # idk why but its required to avoid bug in contract with numpy arrays
import torch
import numpy as np

if options['device']=='cpu':
    torch.set_default_tensor_type(torch.DoubleTensor)
else:  
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
device=torch.device(options['device'])
torch.cuda.set_device(device)


import os
from scipy.sparse.linalg import eigs,eigsh,svds
from linearized import mysvd, myeigh, myeig_old
from linearized import get_linearized_HOTRG_autodiff, get_linearized_HOTRG_full_autodiff, get_linearized_cylinder, verify_linear_operator, check_hermicity
from ScalingDimensions import get_scaling_dimensions
from HOTRGZ2 import HOTRG_layers




options1,params,layers,Ts,logTotals=torch.load(options['tensor_path'],map_location=device)

iLayer=options['iLayer']
T=Ts[iLayer]
assert T.shape[0]==T.shape[1]
options['max_dim']=T.shape[0]


if not options['linearized_full']:
    layers_sel=HOTRG_layers(T,max_dim=T.shape[0],nLayers=2,options=options)
    Mr=get_linearized_HOTRG_autodiff(T,layers_sel)
else:
    Mr=get_linearized_HOTRG_full_autodiff(T,options)

if options['svd_sanity_check']:
    print('sanity check')
    print('hermicity of Mr')
    check_hermicity(Mr,nTests=5) # hermicity is FALSE
    print('hermicity should be false')
    print('linearity of Mr')
    verify_linear_operator(Mr,nTests=5)

print('calculating spectrum of Mr')
if options['svd_method']=='svds':
    assert False, 'should use eigenvalues instead of singular values'
    ur,sr,_=svds(Mr,k=options['svd_num_eigvecs'])
    # do not give the correct eigenvalues
elif options['svd_method']=='eigs':
    sr,ur=eigs(Mr,k=options['svd_num_eigvecs'])
elif options['svd_method']=='eigsh':
    print('warning: should not use eigsh since Mr is not hermitian')
    sr,ur=eigsh(Mr,k=options['svd_num_eigvecs'])
    # should not be used since Mr is not hermitian
elif options['svd_method']=='mysvd':
    assert False, 'should use eigenvalues instead of singular values'
    ur,sr,_=mysvd(Mr,k=options['svd_num_eigvecs'],tol=options['svd_tol'],maxiter=options['svd_max_iter'])
    # similiar results to svds but slower
elif options['svd_method']=='myeig_old':
    sr,ur=myeig_old(Mr,k=options['svd_num_eigvecs'],tol=options['svd_tol'],maxiter=options['svd_max_iter'])


print('eigenvalues',sr)
ur,sr=torch.tensor(ur),torch.tensor(sr)
# sort the eivenvalues

sr,ur=sr.abs()[sr.abs().argsort(descending=True)],ur[:,sr.abs().argsort(descending=True)]


print(options)
print('scaling dimensions from linearized TRG')
print(get_scaling_dimensions(sr,scaling=2))

filename=options['filename']
if filename[-4:]!='.pkl':
    filename+='.pkl'

torch.save((options,sr,ur),filename)
print('file saved: ',filename)
