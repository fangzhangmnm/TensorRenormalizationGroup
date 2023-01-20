if __name__ != '__main__':
    assert False, 'This file is not meant to be imported'

import argparse
parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, required=True) # data/tnr_X16_L10
parser.add_argument('--tensor_path', type=str, required=True) # data/tnr_X16_tensors.pkl
parser.add_argument('--iLayer', type=int, required=True) # 10
parser.add_argument('--svd_num_eigvecs', type=int, default=16)
parser.add_argument('--version', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')

args=parser.parse_args()
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

from scipy.sparse.linalg import eigs,eigsh
from linearized import mysvd, myeigh, get_linearized_cylinder,get_linearized_cylinder_np, verify_linear_operator, check_hermicity
from ScalingDimensions import get_scaling_dimensions
from HOTRGZ2 import HOTRG_layers


options1,layers,Ts,logTotals=torch.load(options['tensor_path'],map_location=device)

iLayer=options['iLayer']
T=Ts[iLayer]

Mc_torch=get_linearized_cylinder(T)
# Mc_np=get_linearized_cylinder_np(_toP(T))

# u=np.random.randn(16**4)
# v_np=Mc_np.matvec(u)
# v_torch=Mc_torch.matvec(u)
# print('MV, np',v_np[:10])
# print('MV, torch',v_torch[:10])

print('check Mc_torch')
check_hermicity(Mc_torch,nTests=1)
verify_linear_operator(Mc_torch,nTests=1)

print('svd of Mc_torch')


# uc,sc,_=mysvd(Mc_np,k=options['svd_num_eigvecs'],tol=options['svd_tol'],maxiter=options['svd_max_iter'])
# -0.0000, 0.1147, 1.0017, 1.1369, 1.1369, 2.0013, 2.0014, 2.0018, 2.0018, 
# 2.1167, 2.1364, 2.1371, 3.0015, 3.0011, 3.0026, 3.0009

# sc,uc=eigs(Mc_np,k=options['svd_num_eigvecs'])
# -0.0000, 0.1147, 1.0017, 1.1369, 1.1369, 2.0012, 2.0014, 2.0018, 2.0018,
# 2.1167, 2.1363, 2.1371, 3.0007, 3.0007, 3.0010, 3.0032
sc,uc=eigsh(Mc_torch,k=options['svd_num_eigvecs'])
# -0.0000, 0.1147, 1.0017, 1.1369, 1.1369, 2.0012, 2.0014, 2.0018, 2.0018,
# 2.1167, 2.1363, 2.1371, 3.0007, 3.0007, 3.0010, 3.0032

print('eigenvalues',sc)
uc,sc=torch.tensor(uc),torch.tensor(sc)
sc,uc=sc.abs()[sc.abs().argsort(descending=True)],uc[:,sc.abs().argsort(descending=True)]

print(options)
print('scaling dimensions from Transfer Matrix on a Cylinder')
print(get_scaling_dimensions(sc,scaling=np.exp(2*np.pi/4)))

filename=options['filename']
if filename[-4:]!='.pkl':
    filename+='.pkl'

torch.save((options,sc,uc),filename)
print('file saved: ',filename)








