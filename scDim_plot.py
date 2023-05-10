if __name__ != '__main__':
    assert False, 'This file is not meant to be imported'

import argparse
parser = argparse.ArgumentParser()

#def show_scaling_dimensions(Ts,loop_length=2,num_scaling_dims=8,volume_scaling=2,is_HOTRG=False,reference_scaling_dimensions=None, reference_center_charge=None,filename=None):

parser.add_argument('--filename', type=str, required=True) # data/tnr_X16_L10
parser.add_argument('--tensor_path', type=str, required=True) # data/tnr_X16_tensors.pkl
parser.add_argument('--loop_length', type=int, default=2)
parser.add_argument('--is_HOTRG', action='store_true')
parser.add_argument('--num_scaling_dims', type=int, default=16)

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

from ScalingDimensions import show_scaling_dimensions,show_diff,show_effective_rank
from HOTRGZ2 import HOTRGLayer

print('loading tensors...')
options1, params, layers, Ts, logTotals = torch.load(options['tensor_path'],map_location=device)

if isinstance(layers[0],HOTRGLayer):
    assert options['is_HOTRG']==True

print(options)
print(options1)

reference_scaling_dimensions=[0,0.125,1,1.125,2,2.125,3,3.125,4,4.125,5,5.125]
reference_central_charge=.5

stride=2 if options['is_HOTRG'] else 1
diff_curve=show_diff(Ts,stride=stride,filename=options['filename'])
effective_rank_curve=show_effective_rank(Ts,filename=options['filename'])

scaling_dimensions_curve=show_scaling_dimensions(Ts,
                        loop_length=options['loop_length'],
                        num_scaling_dims=options['num_scaling_dims'],
                        is_HOTRG=options['is_HOTRG'],
                        filename=options['filename'],
                        reference_scaling_dimensions=reference_scaling_dimensions,
                        reference_center_charge=reference_central_charge)
torch.save((diff_curve,effective_rank_curve,scaling_dimensions_curve),options['filename']+'_curves.pth')
print('saved to',options['filename']+'_curves.pth')