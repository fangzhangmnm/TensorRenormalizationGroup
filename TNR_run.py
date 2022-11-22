
import torch
import os

torch.set_default_tensor_type(torch.cuda.DoubleTensor)
torch.cuda.set_device(1)

from TNR import TNR_layers
from TNModels import Ising2D



filename='data/tnr_X24'


params=Ising2D.get_default_params()

options={
    'nLayers':10,
    'tnr_max_dim_TRG':24,
    'tnr_max_dim_TNR':12,
    'tnr_max_nIter':100,
    'tnr_threshold_TTdiff':1e-7,
    'mcf_enabled':True,
    'mcf_eps':1e-6,
    'mcf_max_iter':200,
}

if os.path.exists(filename+'_options.pkl'):
    _params,_options=torch.load(filename+'_options.pkl')
    if not(params==_params and options==_options):
        def tryRemove(filename):
            if os.path.exists(filename):
                os.remove(filename)
        tryRemove(filename+'_options.pkl')
        tryRemove(filename+'_tensors.pkl')

torch.save((params,options),filename+'_options.pkl')
print('file saved: ',filename+'_options.pkl')

model=Ising2D(params)
T0=model.get_T0()

if os.path.exists(filename+'_tensors.pkl'):
    layers,Ts,logTotals=torch.load(filename+'_tensors.pkl')
else:
    layers,Ts,logTotals=TNR_layers(T0,nLayers=options['nLayers'],options=options,return_tensors=True)
    torch.save((layers,Ts,logTotals),filename+'_tensors.pkl')
    print('file saved: ',filename+'_tensors.pkl')


