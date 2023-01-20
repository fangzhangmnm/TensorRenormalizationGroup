from opt_einsum import contract # idk why but its required to avoid bug in contract with numpy arrays
import torch
import numpy as np
import os

torch.set_default_tensor_type(torch.cuda.DoubleTensor)
device=torch.device('cuda:1')
torch.cuda.set_device(device)


from scipy.sparse.linalg import eigs,eigsh
from linearized import mysvd, myeigh, get_linearized_HOTRG_autodiff, get_linearized_HOTRG_full_autodiff, get_linearized_cylinder, verify_linear_operator, check_hermicity
from ScalingDimensions import get_scaling_dimensions
from HOTRGZ2 import HOTRG_layers


filename='data/linearized_hotrg_X24_L10'

options={
    'tensor_path':'data/hotrg_X24_tensors.pkl',
    'iLayer':10,
    
    'linearized_full':False,
    
    'max_dim':16,
    'gilt_enabled':False,
    'gilt_eps':8e-7,
    'gilt_nIter':1,
    'mcf_enabled':False,
    'mcf_eps':1e-16,
    'mcf_max_iter':20,
    
    'svd_max_iter':100,
    'svd_tol':1e-16,
    'svd_num_eigvecs':16,
    
    '_version':1,
}


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



layers,Ts,logTotals=torch.load(options['tensor_path'],map_location=device)

iLayer=options['iLayer']
T=Ts[iLayer]
assert T.shape[0]==options['max_dim']



if not options['linearized_full']:
    layers_sel=HOTRG_layers(T,max_dim=options['max_dim'],nLayers=2,options=options)
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


print(options)
print('scaling dimensions from linearized TRG')
print(get_scaling_dimensions(sr,scaling=2))

torch.save((sr,ur),filename+'_eigs_lTRG.pkl')
print('file saved: ',filename+'_eigs_lTRG.pkl')
