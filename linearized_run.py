if __name__ != '__main__':
    assert False, 'This file is not meant to be imported'

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--filename', type=str, required=True) # data/tnr_X16_L10
parser.add_argument('--tensor_path', type=str, required=True) # data/tnr_X16_tensors.pkl
parser.add_argument('--iLayer', type=int, required=True) # 10
parser.add_argument('--linearized_full', action='store_true')
parser.add_argument('--linearized_use_jax', action='store_true')
parser.add_argument('--gilt_enabled', action='store_true')
parser.add_argument('--gilt_eps', type=float, default=8e-7)
parser.add_argument('--gilt_nIter', type=int, default=1)
parser.add_argument('--mcf_enabled', action='store_true')
parser.add_argument('--mcf_eps', type=float, default=1e-16)
parser.add_argument('--mcf_max_iter', type=int, default=20)
parser.add_argument('--mcf_phase_iter1', type=int, default=3)
parser.add_argument('--mcf_phase_iter2', type=int, default=10)
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

device=torch.device(options['device'])
if options['device']=='cpu':
    torch.set_default_tensor_type(torch.DoubleTensor)
else:  
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    torch.cuda.set_device(device)
import jax
jax.config.update("jax_enable_x64", True)




import os
from scipy.sparse.linalg import eigs,eigsh,svds
from linearized import mysvd, myeigh, myeig_old
from linearized import get_linearized_HOTRG_autodiff, get_linearized_HOTRG_full_autodiff
from linearized import get_linearized_HOTRG_jax, get_linearized_HOTRG_full_jax
from linearized import get_linearized_cylinder, verify_linear_operator, check_hermicity
from ScalingDimensions import get_scaling_dimensions
from HOTRGZ2 import HOTRG_layers



print('loading tensors...')
options1,params,layers,Ts,logTotals=torch.load(options['tensor_path'],map_location=device)

iLayer=options['iLayer']
T=Ts[iLayer]
assert T.shape[0]==T.shape[1]
options['max_dim']=T.shape[0]

del options1,params,layers,Ts,logTotals
torch.cuda.empty_cache()

if options['linearized_full']:
    if options['linearized_use_jax']:
        Mr=get_linearized_HOTRG_full_jax(T,options)
    else:
        Mr=get_linearized_HOTRG_full_autodiff(T,options)
else:
    layers_sel=HOTRG_layers(T,max_dim=T.shape[0],nLayers=2,options=options)
    if options['linearized_use_jax']:
        Mr=get_linearized_HOTRG_jax(T,layers_sel)
    else:
        Mr=get_linearized_HOTRG_autodiff(T,layers_sel)

if options['svd_sanity_check']:
    print('sanity check')
    print('hermicity of Mr')
    check_hermicity(Mr,nTests=5) # hermicity is FALSE
    print('hermicity should be false')
    print('linearity of Mr')
    verify_linear_operator(Mr,nTests=5)

print('calculating spectrum of Mr')
if options['svd_method']=='svds':
    print('warning!','should use eigenvalues instead of singular values, svd is not correct')
    ur,sr,_=svds(Mr,k=options['svd_num_eigvecs'])
    # do not give the correct eigenvalues
elif options['svd_method']=='eigs':
    sr,ur=eigs(Mr,k=options['svd_num_eigvecs'])
elif options['svd_method']=='eigsh':
    print('warning: should not use eigsh since Mr is not hermitian')
    sr,ur=eigsh(Mr,k=options['svd_num_eigvecs'])
    # should not be used since Mr is not hermitian
elif options['svd_method']=='mysvd':
    print('warning!','should use eigenvalues instead of singular values, svd is not correct')
    ur,sr,_=mysvd(Mr,k=options['svd_num_eigvecs'],tol=options['svd_tol'],maxiter=options['svd_max_iter'])
    # similiar results to svds but slower
elif options['svd_method']=='myeig_old':
    sr,ur=myeig_old(Mr,k=options['svd_num_eigvecs'],tol=options['svd_tol'],maxiter=options['svd_max_iter'])


print('eigenvalues',sr)
# sort the eivenvalues

#now sr,ur are numpy arrays
#sr,ur=sr.abs()[sr.abs().argsort(descending=True)],ur[:,sr.abs().argsort(descending=True)]
#translate it to numpy operations
sr,ur=sr[np.abs(sr).argsort()[::-1]],ur[:,np.abs(sr).argsort()[::-1]]

print('eigenvalues',sr)

print(options)
print('scaling dimensions from linearized TRG')
print(get_scaling_dimensions(torch.as_tensor(sr).abs(),scaling=2))

# ulr=np.zeros_like(ur)
# for i in range(ur.shape[1]):
#     #ulr[:,i]=Mr@ur[:,i]
#     #calculate the real and imaginary part of the right eigenvector
#     ulri=Mr@np.real(ur[:,i])+1j*Mr@np.imag(ur[:,i])
#     ulri=ulri/np.linalg.norm(ulri)
#     ulr[:,i]=ulri

ur,sr=torch.tensor(ur),torch.tensor(sr)


filename_txt=options['filename']
if '.' in filename_txt[-5:]:
    filename_txt=filename_txt.split('.')[0]
filename_txt=filename_txt+'_scdims.txt'
with open(filename_txt,'w') as f:
    print(get_scaling_dimensions(torch.as_tensor(sr).abs(),scaling=2).detach().cpu().numpy(),file=f)
print('file saved: ',filename_txt)




filename=options['filename']

torch.save((options,sr,ur),filename)
print('file saved: ',filename)

