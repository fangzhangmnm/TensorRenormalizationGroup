import argparse
parser = argparse.ArgumentParser()

if __name__ != '__main__':
    assert False, 'This file is not meant to be imported'

# filename and tensor_path are required
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
# parser.add_argument('--svd_max_iter', type=int, default=100)
# parser.add_argument('--svd_tol', type=float, default=1e-16)
parser.add_argument('--svd_num_eigvecs', type=int, default=16)
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
options.pop('device')


import os
from scipy.sparse.linalg import eigs,eigsh
from linearized import mysvd, myeigh, get_linearized_HOTRG_autodiff, get_linearized_HOTRG_full_autodiff, get_linearized_cylinder, verify_linear_operator, check_hermicity
from ScalingDimensions import get_scaling_dimensions
from HOTRGZ2 import HOTRG_layers

filename=options['filename']

if os.path.exists(filename+'_options.pkl'):
    _options=torch.load(filename+'_options.pkl',map_location=device)
    if not(options==_options):
        def tryRemove(filename):
            if os.path.exists(filename):
                os.remove(filename)
        tryRemove(filename+'_options.pkl')
        tryRemove(filename+'_eigs_lTRG.pkl')

torch.save(options,filename+'_options.pkl')
print('file saved: ',filename+'_options.pkl')
print(options)



layers,Ts,logTotals=torch.load(options['tensor_path'],map_location=device)

iLayer=options['iLayer']
T=Ts[iLayer]
assert T.shape[0]==T.shape[1]
options['max_dim']=T.shape[0]


if not options['linearized_full']:
    layers_sel=HOTRG_layers(T,max_dim=T.shape[0],nLayers=2,options=options)
    Mr=get_linearized_HOTRG_autodiff(T,layers_sel)
else:
    Mr=get_linearized_HOTRG_full_autodiff(T,options)

check_hermicity(Mr,nTests=5) # hermicity is FALSE
verify_linear_operator(Mr,nTests=5)

print('svd of Mr')
# ur,sr,_=mysvd(Mr,k=options['svd_num_eigvecs'],tol=options['svd_tol'],maxiter=options['svd_max_iter'])
sr,ur=eigs(Mr,k=options['svd_num_eigvecs'])


print('eigenvalues',sr)
ur,sr=torch.tensor(ur),torch.tensor(sr)
# sort the eivenvalues

sr,ur=sr.abs()[sr.abs().argsort(descending=True)],ur[:,sr.abs().argsort(descending=True)]


#print(options)
print('scaling dimensions from linearized TRG')
print(get_scaling_dimensions(sr,scaling=2))

torch.save((sr,ur),filename+'_eigs_lTRG.pkl')
print('file saved: ',filename+'_eigs_lTRG.pkl')
