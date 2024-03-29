if __name__ != '__main__':
    assert False, 'This file is not meant to be imported'

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--filename', type=str, required=True) # data/hotrg_gilt_X24
parser.add_argument('--nLayers', type=int, required=True) # 60
parser.add_argument('--max_dim', type=int, required=True) # 24
parser.add_argument('--model', type=str, required=True) # 'Ising2D'
parser.add_argument('--params', type=str, default=None)
parser.add_argument('--params_file', type=str, default=None)
parser.add_argument('--gilt_enabled', action='store_true')
parser.add_argument('--gilt_eps', type=float, default=8e-7)
parser.add_argument('--gilt_nIter', type=int, default=1)
parser.add_argument('--mcf_enabled', action='store_true')
parser.add_argument('--mcf_eps', type=float, default=1e-16)
parser.add_argument('--mcf_max_iter', type=int, default=200)
parser.add_argument('--mcf_phase_iter1', type=int, default=3)
parser.add_argument('--mcf_phase_iter2', type=int, default=10)
parser.add_argument('--hotrg_sanity_check', action='store_true')
parser.add_argument('--version', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()
options=vars(args)


print('loading library...')

from opt_einsum import contract # idk why but its required to avoid bug in contract with numpy arrays
import torch
import numpy as np
torch.set_default_tensor_type(torch.cuda.DoubleTensor)
if options['device']=='cpu':
    torch.set_default_tensor_type(torch.DoubleTensor)
else:  
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
device=torch.device(options['device'])
torch.cuda.set_device(device)


from HOTRGZ2 import HOTRG_layers
from TNModels import Models

Model=Models[options['model']]
params=Model.get_default_params()
if options['params'] is not None:
    import json
    params1=options['params']
    params1=json.loads(params1)
    params.update(params1)
if options['params_file'] is not None:
    params1=torch.load(options['params_file'])
    params.update(params1)

model=Model(params)
T0=model.get_T0()

layers,Ts,logTotals=HOTRG_layers(T0,
                        max_dim=options['max_dim'],nLayers=options['nLayers'],
                        options=options,
                        return_tensors=True)

filename=options['filename']
torch.save((options,params,layers,Ts,logTotals),filename)
print('file saved: ',filename)
