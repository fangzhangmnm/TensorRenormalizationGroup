import argparse


parser = argparse.ArgumentParser(description='Evaluate correlation')
parser.add_argument('--filename', type=str)
parser.add_argument('--tensors_filename', type=str)
parser.add_argument('--points_filename', type=str)
parser.add_argument('--log2Size', type=int)
parser.add_argument('--device', type=str,default='cuda:0')


args=parser.parse_args()
filename=args.filename
tensors_filename=args.tensors_filename
points_filename=args.points_filename
log2Size=args.log2Size


import torch
import pickle

device=torch.device(args.device)
if device.type=='cuda':
    torch.cuda.set_device(device.index)
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)


from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from TNModels import Ising2D,AKLT2D,AKLT2DStrange
from HOTRGZ2 import forward_observable_tensor,forward_observable_tensors,trace_tensor,trace_two_tensors,get_lattice_size,get_dist_torus_2D
def _toN(t):
    return t.detach().cpu().tolist()

options,params,layers,Ts,logTotals=torch.load(tensors_filename, map_location=device)

model=Ising2D(params)
T0=model.get_T0()
# T0_op1,T0_op2,checkerboard=model.get_SZT0(),model.get_SZT0(),False
T0_op=model.get_SZT0()
checkerboard=False

layers1=layers[:2*log2Size]
lattice_size=get_lattice_size(len(layers1),spacial_dim=len(T0.shape)//2)

coordsss=pickle.load(open(points_filename,'rb'))
print('coordsss:',coordsss)
print('lattice_size:',lattice_size)

data=[]
for coordss in tqdm(coordsss):
    # check that all coordinates are in the lattice
    assert all(isinstance(c,int) and 0<=c and c<s for coords in coordss for c,s in zip(coords,lattice_size))
    # check that all coordinates are distinct
    assert len(set(coordss))==len(coordss)
        
    print(coordss)
    T,T_op12,logTotal=forward_observable_tensors(T0,[T0_op]*len(coordss),coordss,\
                               layers=layers1,checkerboard=checkerboard,\
                               cached_Ts=Ts)
    correlation=_toN(trace_tensor(T_op12)/trace_tensor(T))
    newRow={**params,
        **options,
        'correlation':correlation,}
    for i,coords in enumerate(coordss):
        newRow['x'+str(i)]=coords[0]
        newRow['y'+str(i)]=coords[1]
    
    data.append(newRow)
    
data=pd.DataFrame(data)

data.to_pickle(filename)