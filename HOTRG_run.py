import torch
import os

torch.set_default_tensor_type(torch.cuda.DoubleTensor)
torch.cuda.set_device(0)

from HOTRGZ2 import HOTRG_layers
from TNModels import Ising2D



filename='data/hotrg_gilt_X24'


options={
    'nLayers':60,
    'max_dim':24,
    'gilt_enabled':True,
    'gilt_eps':8e-7,
    'gilt_nIter':1,
    'mcf_enabled':True,
    'mcf_eps':1e-16,
    'mcf_max_iter':200
}

params=Ising2D.get_default_params()
params['beta']+=0




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
    layers,Ts,logTotals=HOTRG_layers(T0,
                            max_dim=options['max_dim'],nLayers=options['nLayers'],
                            options=options,
                            return_tensors=True)
    torch.save((layers,Ts,logTotals),filename+'_tensors.pkl')
    print('file saved: ',filename+'_tensors.pkl')
