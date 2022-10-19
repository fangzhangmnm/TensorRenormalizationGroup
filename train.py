import matplotlib.pylab
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from opt_einsum import contract
from tqdm.auto import tqdm
import os
import torch

def _toN(t):
    t=t.detach().cpu().numpy()
    if len(t.shape)==0:
        t=t.item()
    return t

from HOTRGZ2 import *
from TNModels import *
from DHOTRG import *
from utils import *



def create_datapoint(name,options,params,override=False):
    if os.path.exists(name+'.checkpoint') and not override:
        print(f'file {name}.checkpoint alread exists!')
        return
    checkpoint={'options':options,'params':params,'iter':0}
    torch.save(checkpoint,name+'.checkpoint')
    print(f'datapoint {name}.checkpoint created')

    
def set_default_tensor_type(device,dtype):
    if device[:4]=='cuda':
        torch.cuda.set_device(int(device[5:]) if len(device)>5 else 0)
    tmp={'cpu':torch,'cuda':torch.cuda}[device[:4]]
    torch.set_default_tensor_type({'float32':tmp.FloatTensor,'float64':tmp.DoubleTensor}[dtype])

def train(name,nIter,device=None):
    if os.path.exists(name+'.curve'):
        curve=pd.read_pickle(name+'.curve')
    else:
        curve=pd.DataFrame()
    checkpoint=torch.load(name+'.checkpoint',map_location=device)
    options=checkpoint['options']
    if device is not None:
        options['device']=device
    set_default_tensor_type(options['device'],options['dtype'])
    
    params=checkpoint['params']
    model=Models[options['Model']](params)
    
    def calculate_observables():
        rows=[{**options,**params,
               'iter':checkpoint['iter'],'layer':layer,
              } for layer in range(options['nLayers']+1)]
        for op_name,T_op0,checkerboard in model.get_T_op0s():
            Ts,T_ops,logTotals=\
                forward_observable_tensor(model.get_T0(),T_op0,\
                                  checkpoint['isometries'],checkpoint['dimRs'],\
                                  checkboard=checkerboard,return_layers=True)
            for layer in range(options['nLayers']+1):
                T,T_op,logTotal=Ts[layer],T_ops[layer],logTotals[layer]
                rows[layer]['logZ']=_toN((torch.log(trace_tensor(T))+logTotal)/2.**layer)
                rows[layer][op_name]=_toN(trace_tensor(T_op)/trace_tensor(T))
                rows[layer][op_name+'2']=_toN(trace_two_tensors(T_op)/trace_two_tensors(T))
                for axis in range(model.spacial_dim-1):
                    rows[layer][f'wDiff{axis}']=0
                    rows[layer]['wDiff_total']=0
                    if 'wDiffs' in checkpoint:
                        rows[layer]['wDiff_total']=np.array(checkpoint['wDiffs']).sum()
                        if layer<options['nLayers']:
                            rows[layer][f'wDiff{axis}']=checkpoint['wDiffs'][layer][axis]
        for row in rows:
            print(row)
        return rows
        
    if 'isometries' not in checkpoint:
        checkpoint['isometries'],checkpoint['dimRs']= \
            calc_isometries(model.get_T0(),model.get_dimR(),max_dim=options['max_dim'],nLayers=options['nLayers'])
        curve=curve.append(calculate_observables(),ignore_index=True)
        torch.save(checkpoint,name+'.checkpoint')
        curve.to_pickle(name+'.curve')
        curve.to_csv(name+'.csv')
    for _iter in range(nIter):
        
        checkpoint['wDiffs'],checkpoint['wDiff_inners']= \
            update_isometries(model.get_T0(),checkpoint['isometries'],checkpoint['dimRs'],
                              inner_iter=options['dHOTRG_inner_iter'],Z2_projection=options['Z2_projection'])
        checkpoint['iter']+=1
        curve=curve.append(calculate_observables(),ignore_index=True)
        torch.save(checkpoint,name+'.checkpoint')
        curve.to_pickle(name+'.curve')
        curve.to_csv(name+'.csv')
    
        