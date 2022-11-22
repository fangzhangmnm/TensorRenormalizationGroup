import argparse
from tensors.tensor import Tensor
from tensors.symmetrytensors import TensorZ2, TensorZ3, TensorU1

# Functions for parsing different types
def parse_list_of_ints(string):
    if string[0:6] == "range(":
        t = string[6:-1].replace(" ", "").split(" ,") 
        t = tuple(map(int, t))
        return list(range(*t))
    if string:
        if string[0] == "[" or string[0] == "(":
            del(string[0])
        if string[-1] == "]" or string[0] == ")":
            del(string[-1])
    return list(int(s) for s in string.replace(" ", "").split(","))

def parse_list_of_floats(string):
    return list(float(s) for s in string.replace(" ","").split(","))

def parse_linspace(string):
    split = string.split()
    i = 0
    return_value = []
    while i+2 < len(split):
        return_value.append((float(split[i]), float(split[i+1]), int(split[i+2])))
        i += 3
    return return_value

def parse_bool(string):
    string = string.lower().strip()
    return not (string == "false" or string == "0")

def parse_dtype(string):
    string = string.lower().strip()
    if string=='complex' or string=='complex_' or string=='np.complex_':
        return np.complex_
    elif string=='float' or string=='float_' or string=='np.float_':
        return np.float_

def parse_tensor_class_list(string):
    word_list = parse_word_list(string)
    classes = []
    for w in word_list:
        if w.strip().lower() == "tensor":
            classes.append(Tensor)
        elif w.strip().lower() == "tensorz2":
            classes.append(TensorZ2)
        elif w.strip().lower() == "tensorz3":
            classes.append(TensorZ3)
        elif w.strip().lower() == "tensoru1":
            classes.append(TensorU1)
    return classes


parse_word_list = str.split

# Dictionary that maps a string that names a data type to a function for
# parsing it.
parser_dict = {'int_list': parse_list_of_ints,
               'float_list': parse_list_of_floats,
               'float': float,
               'int': int,
               'str': str,
               'linspace': parse_linspace,
               'bool': parse_bool,
               'dtype': parse_dtype,
               'tensor_class_list': parse_tensor_class_list,
               'word_list': parse_word_list}

def parse_argv(argv, *args):
    parser = argparse.ArgumentParser()
    for t in args:
        parser.add_argument('-' + t[0], type=parser_dict[t[1]], default=t[2])
    args = parser.parse_args(argv[1:])
    return args

