import torch
torch.set_default_tensor_type(torch.cuda.DoubleTensor)
torch.cuda.set_device(0)
device=torch.tensor(1).device

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from TNModels import Ising2D,AKLT2D,AKLT2DStrange
from HOTRGZ2 import forward_observable_tensor,forward_observable_tensors,trace_tensor,trace_two_tensors,get_lattice_size,get_dist_torus_2D
def _toN(t):
    return t.detach().cpu().tolist()


# filename='data/hotrg_X24_torus_correlation_1024'
# tensors_filename='data/hotrg_X24'
# log2Size=10


# filename='data/hotrg_gilt_X24_torus_correlation_1024'
# tensors_filename='data/hotrg_gilt_X24'
# log2Size=10

# filename='data/hotrg_X24_torus_correlation_30'
# tensors_filename='data/hotrg_X24'
# log2Size=30

filename='data/hotrg_gilt_X24_torus_correlation_30'
tensors_filename='data/hotrg_gilt_X24'
log2Size=30

data_count=100


params,options=torch.load(tensors_filename+'_options.pkl',map_location=device)
layers,Ts,logTotals=torch.load(tensors_filename+'_tensors.pkl',map_location=device)


model=Ising2D(params)

model=Ising2D(params)
T0=model.get_T0()
T0_op1,T0_op2,checkerboard=model.get_SZT0(),model.get_SZT0(),False


layers1=layers[:2*log2Size]
coordss=[]

lattice_size=get_lattice_size(len(layers1),spacial_dim=len(T0.shape)//2)
#for r in [0,1,2,3,4,10,30,100,300,1000,3000,10000,30000,100000,300000]:
for r in list(range(10))+list(np.geomspace(10,2**log2Size-1,max(2,data_count//2-10))):
    r=int(r)
    assert r<lattice_size[0]
    x,y=int(r),0
    coordss.append((x,y))
    x,y=int(lattice_size[0]-1-r),0
    coordss.append((x,y))

coordss=list(sorted(set(coordss)))

print('coordss:',coordss)

data=[]
for coords in tqdm(coordss):
    if not(all(isinstance(c,int) and 0<=c and c<s for c,s in zip(coords,lattice_size))):continue
    if all(c==0 for c in coords):continue
        

    T,T_op12,logTotal=forward_observable_tensors(T0,[T0_op1,T0_op2],[(0,)*len(coords),coords],\
                               layers=layers1,checkerboard=checkerboard,\
                               cached_Ts=Ts)
    correlation=_toN(trace_tensor(T_op12)/trace_tensor(T))
    newRow={**params,
        **options,
        'x':coords[0],'y':coords[1],
        'correlation':correlation,}
    
    data.append(newRow)
    
data=pd.DataFrame(data)

data.to_pickle(filename+'.pkl')