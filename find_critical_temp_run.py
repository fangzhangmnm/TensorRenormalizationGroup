if __name__ != '__main__':
    assert False, 'This file is not meant to be imported'

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--filename', type=str, required=True) # data/hotrg_gilt_X24_Tc
parser.add_argument('--nLayers', type=int, required=True) # 60
parser.add_argument('--max_dim', type=int, required=True) # 24
parser.add_argument('--model', type=str, required=True) # 'Ising2D'
parser.add_argument('--param_name', type=str, required=True) # 'beta'
parser.add_argument('--param_min', type=float, required=True) # 0.43068679350977147
parser.add_argument('--param_max', type=float, required=True) # 0.45068679350977147
parser.add_argument('--observable_name', type=str, required=True) # 'magnetization'
parser.add_argument('--tol', type=float, default=1e-8)
parser.add_argument('--gilt_enabled', action='store_true')
parser.add_argument('--gilt_eps', type=float, default=8e-7)
parser.add_argument('--gilt_nIter', type=int, default=1)
parser.add_argument('--mcf_enabled', action='store_true')
parser.add_argument('--mcf_eps', type=float, default=1e-16)
parser.add_argument('--mcf_max_iter', type=int, default=200)
parser.add_argument('--hotrg_sanity_check', action='store_true')
parser.add_argument('--version', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()
options=vars(args)



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


from HOTRGZ2 import HOTRG_layers,trace_tensor,forward_observable_tensor,trace_two_tensors
from TNModels import Models

Model=Models[options['model']]
params=Model.get_default_params()
param_name=options['param_name']


def eval_model(params):
    model=Model(params)
    T0,(T0_op,checkerboard)=model.get_T0(),model.get_observables()[options['observable_name']]
    layers,Ts,logTotals=HOTRG_layers(T0,
                        max_dim=options['max_dim'],nLayers=options['nLayers'],
                        options=options,return_tensors=True)
    Ts,T_ops,logTotals=forward_observable_tensor(T0,T0_op,
                        layers=layers,checkerboard=checkerboard,
                        return_layers=True,cached_Ts=Ts)
    T=Ts[-1]/Ts[-1].norm()
    logZ=(trace_tensor(T).log()+logTotals[-1])/2**options['nLayers']
    dNorm=torch.tensor([T.norm() for T in Ts]) # according to Lyu, this can be used to determine the phase transition
    #dNorm=T.norm() 
    obs=trace_two_tensors(T_ops[-1])/trace_two_tensors(Ts[-1])
    return T,logZ,obs,dNorm

beta_min=options['param_min']
beta_max=options['param_max']
beta_ref=Model.get_default_params()[param_name]

print('evaluating model at beta_min...')
params[param_name]=beta_min
T_min,logZ_min,obs_min,dNorm_min=eval_model(params)


print('evaluating model at beta_max...')
params[param_name]=beta_max
T_max,logZ_max,obs_max,dNorm_max=eval_model(params)


print('beta_min=',beta_min,'beta_max=',beta_max)
print('logZ_min=',logZ_min.item(),'logZ_max=',logZ_max.item())
print('obs_min=',obs_min.item(),'obs_max=',obs_max.item())
print('searching for critical temperature using bisection method')
while beta_max-beta_min>options['tol']:
    beta_new=(beta_min+beta_max)/2
    params[param_name]=beta_new
    T_new,logZ_new,obs_new,dNorm_new=eval_model(params)
    print('beta_min=',beta_min,'beta_new=',beta_new,'beta_max=',beta_max,'beta_ref',beta_ref)
    print('logZ_min=',logZ_min.item(),'logZ_new=',logZ_new.item(),'logZ_max=',logZ_max.item())
    print('obs_min=',obs_min.item(),'obs_new=',obs_new.item(),'obs_max=',obs_max.item())
    #dist_min=(logZ_min-logZ_new).abs()
    #dist_max=(logZ_max-logZ_new).abs()
    #dist_min=contract('ijkl,ijkl->',T_min,T_new).abs()
    #dist_max=contract('ijkl,ijkl->',T_max,T_new).abs()
    # dist_min=(obs_min-obs_new).abs()
    # dist_max=(obs_max-obs_new).abs()
    dist_min=(dNorm_min-dNorm_new).norm()
    dist_max=(dNorm_max-dNorm_new).norm()
    print('dist_min=',dist_min,'dist_max=',dist_max)
    if dist_min<dist_max:
        print('keeping beta_max')
        beta_min=beta_new
        T_min=T_new
        logZ_min=logZ_new
        obs_min=obs_new
        dNorm_min=dNorm_new
    else:
        print('keeping beta_min')
        beta_max=beta_new
        T_max=T_new
        logZ_max=logZ_new
        obs_max=obs_new
        dNorm_max=dNorm_new


print('critical temperature found: beta=',beta_new,' reference: ',beta_ref)

filename=options['filename']
torch.save({param_name:beta_new},filename)

    
    