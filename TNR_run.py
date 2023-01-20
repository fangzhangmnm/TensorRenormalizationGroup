if __name__ != '__main__':
    assert False, 'This file is not meant to be imported'

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--filename', type=str, required=True) # data/tnr_X16
parser.add_argument('--nLayers', type=int, required=True) # 20
parser.add_argument('--tnr_max_dim_TRG', type=int, required=True) # 16
parser.add_argument('--tnr_max_dim_TNR', type=int, required=True) # 8
parser.add_argument('--tnr_max_nIter', type=int, default=0)
parser.add_argument('--tnr_threshold_TTdiff', type=float, default=1e-7)
parser.add_argument('--tnr_disentangling_method', type=str, choices=['fast','relaxing'], default='relaxing')
parser.add_argument('--mcf_enabled', action='store_true')
parser.add_argument('--mcf_eps', type=float, default=1e-6)
parser.add_argument('--mcf_max_iter', type=int, default=200)
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

from TNR import TNR_layers
from TNModels import Ising2D


params=Ising2D.get_default_params()
model=Ising2D(params)
T0=model.get_T0()

layers,Ts,logTotals=TNR_layers(T0,nLayers=options['nLayers'],options=options,return_tensors=True)

filename=options['filename']
if filename[-4:]!='.pkl':
    filename+='.pkl'
torch.save((options,params,layers,Ts,logTotals),filename)
print('file saved: ',filename)

