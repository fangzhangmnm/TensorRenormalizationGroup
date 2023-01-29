if __name__ != '__main__':
    assert False, 'This file is not meant to be imported'

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--filename', type=str, required=True) # data/hotrg_gilt_X24_Tc

import torch

data=torch.load(parser.parse_args().filename)
print(data)