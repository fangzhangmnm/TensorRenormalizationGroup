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


# filename='data/hotrg_gilt_X24_correlation_00.pkl'
# tensors_filename='data/hotrg_gilt_X24.pth'
# fix_x0y0=True

# filename='data/hotrg_X24_correlation_00.pkl'
# tensors_filename='data/hotrg_X24.pth'
# fix_x0y0=True

# filename='data/hotrg_X24_correlation.pkl'
# tensors_filename='data/hotrg_X24.pth'
# fix_x0y0=False

# filename='data/hotrg_gilt_X24_correlation.pkl'
# tensors_filename='data/hotrg_gilt_X24.pth'
# fix_x0y0=False

# filename='data/hotrg_gilt_X24_lowB_correlation.pkl'
# tensors_filename='data/hotrg_gilt_X24_lowB.pth'
# fix_x0y0=False

# filename='data/hotrg_gilt_X24_highB_correlation.pkl'
# tensors_filename='data/hotrg_gilt_X24_highB.pth'
# fix_x0y0=False

# log2Size=30
# data_count=100


options,params,layers,Ts,logTotals=torch.load(tensors_filename, map_location=device)
# params,options=torch.load(tensors_filename+'_options.pkl',map_location=device)
# layers,Ts,logTotals=torch.load(tensors_filename+'_tensors.pkl',map_location=device)


model=Ising2D(params)
T0=model.get_T0()
T0_op1,T0_op2,checkerboard=model.get_SZT0(),model.get_SZT0(),False

layers1=layers[:2*log2Size]
lattice_size=get_lattice_size(len(layers1),spacial_dim=len(T0.shape)//2)

coordsss=pickle.load(open(points_filename,'rb'))
print('coordsss:',coordsss)
print('lattice_size:',lattice_size)

# coordsss=[]
# for i in range(data_count):
#     lattice_size=get_lattice_size(2*log2Size,spacial_dim=2)
#     th=np.random.uniform(0,np.pi/2)
#     r=np.exp(np.random.uniform(np.log(1),np.log(min(lattice_size))))
#     x,y=int(np.abs(r*np.cos(th))),int(np.abs(r*np.sin(th)))
#     if x==0 and y==0:
#         x,y=(1,0) if np.random.uniform()<0.5 else (0,1)
#     x0,y0=np.random.randint(0,lattice_size[0]-x),np.random.randint(0,lattice_size[1]-y)
#     if fix_x0y0:
#         x0,y0=(0,0)
#     x1,y1=x0+x,y0+y
#     coordsss.append(((x0,y0),(x1,y1)))
# coordsss=list(sorted(set(coordsss)))

data=[]
for coordss in tqdm(coordsss):
    assert all(isinstance(c,int) and 0<=c and c<s for coords in coordss for c,s in zip(coords,lattice_size))
    assert coordss[0]!=coordss[1]
        
    print(coordss)
    T,T_op12,logTotal=forward_observable_tensors(T0,[T0_op1,T0_op2],coordss,\
                               layers=layers1,checkerboard=checkerboard,\
                               cached_Ts=Ts)
    correlation=_toN(trace_tensor(T_op12)/trace_tensor(T))
    newRow={**params,
        **options,
        'x0':coordss[0][0],'y0':coordss[0][1],
        'x1':coordss[1][0],'y1':coordss[1][1],
        'correlation':correlation,}
    
    data.append(newRow)
    
data=pd.DataFrame(data)

data.to_pickle(filename)