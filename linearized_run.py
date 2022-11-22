import torch
import os
torch.set_default_tensor_type(torch.cuda.DoubleTensor)
torch.cuda.set_device(1)

from linearized import mysvd, myeigh, get_linearized_HOTRG_autodiff, get_linearized_HOTRG_full_autodiff, get_linearized_cylinder, verify_linear_operator, check_hermicity
from ScalingDimensions import get_scaling_dimensions
from HOTRGZ2 import HOTRG_layers


filename='data/linearized_hotrg_nogilt_graft_tnr_X24_I10'


options={
    'tensor_path':'data/tnr_X24_tensors.pkl',
    'max_dim':24,
    'gilt_enabled':False,
    'gilt_eps':8e-7,
    'gilt_nIter':1,
    'mcf_enabled':False,
    'mcf_eps':1e-16,
    'mcf_max_iter':20,
    
    'iLayer':10,
    'svd_max_iter':100,
    'svd_tol':1e-16,
    'svd_num_eigvecs':16,
    
    '_version':2,
}


if os.path.exists(filename+'_options.pkl'):
    _options=torch.load(filename+'_options.pkl')
    if not(options==_options):
        def tryRemove(filename):
            if os.path.exists(filename):
                os.remove(filename)
        tryRemove(filename+'_options.pkl')
        tryRemove(filename+'_eigs_lTRG.pkl')

torch.save(options,filename+'_options.pkl')
print('file saved: ',filename+'_options.pkl')



layers,Ts,logTotals=torch.load(options['tensor_path'])

iLayer=options['iLayer']
T=Ts[iLayer]
assert T.shape[0]==options['max_dim']

#layers_sel=layers[iLayer:iLayer+2]

layers_sel=HOTRG_layers(T,max_dim=options['max_dim'],nLayers=2,options=options)



Mr=get_linearized_HOTRG_autodiff(T,layers_sel)

#Mr=get_linearized_HOTRG_full_autodiff(T,options_ltrg)

check_hermicity(Mr,nTests=5)
verify_linear_operator(Mr,nTests=5)

ur,sr,_=mysvd(Mr,k=options['svd_num_eigvecs'],tol=options['svd_tol'],maxiter=options['svd_max_iter'])
ur,sr=torch.tensor(ur),torch.tensor(sr)

print(options)
print('scaling dimensions from linearized TRG')
print(get_scaling_dimensions(sr,scaling=2))

torch.save((sr,ur),filename+'_eigs_lTRG.pkl')
print('file saved: ',filename+'_eigs_lTRG.pkl')
