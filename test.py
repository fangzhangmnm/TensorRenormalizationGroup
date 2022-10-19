from GILT import Gilt_Cube
from TNModels import Ising3D
import torch
import code

A=Ising3D().get_T0()
B=Gilt_Cube(*[A]*8)
code.interact(local=locals())
torch.cos