import torch
import numpy as np
import os.path
import shutil
import itertools
import json
import inspect
import argparse

def make_dir(path, overwrite=False, sub_dirs=False, verbose=True):  
    Directory = path
    if overwrite:
        shutil.rmtree(Directory, ignore_errors=True)
        os.mkdir(Directory)
    else:
        for I in itertools.count():
            Directory = path + '__' + str(I+1)
            if not os.path.isdir(Directory):
                os.mkdir(Directory)
                break
            else:
                continue
    if sub_dirs:
        for d in sub_dirs: 
            os.mkdir(Directory+'/'+d)
    if verbose:
        print("#================================================")
        print("INFO: created directory: {}".format(Directory))
        print("#================================================")
    return Directory

def save_configs(configs, filename):
    class_attributes = {}
    for cls in configs:
        attributes = {}
        for base in reversed(cls.__mro__):
            attributes.update(vars(base))
        attributes = {k: v for k, v in attributes.items() if not (k.startswith('__') or callable(v))}
        class_attributes[cls.__name__] = attributes
    with open(filename, 'w') as f:
        json.dump(class_attributes, f, indent=4)

def savefig(filename, extension="png"):
    counter = 1
    base_filename, ext = os.path.splitext(filename)
    if ext == "":
        ext = f".{extension}"
    unique_filename = f"{base_filename}{ext}"
    while os.path.exists(unique_filename):
        unique_filename = f"{base_filename}_{counter}{ext}"
        counter += 1
    return unique_filename        

def savetensor(tensor, filename, save_dir, extension="npy", use_seed=None, verbose=True):
    
    if use_seed is None:
        counter = 1
        filename = save_dir+'/'+filename 
        base_filename, ext = os.path.splitext(filename)
        if ext == "":
            ext = f".{extension}"
        unique_filename = f"{base_filename}{ext}"
        while os.path.exists(unique_filename):
            unique_filename = f"{base_filename}_{counter}{ext}"
            counter += 1
        file = '/{}'.format(unique_filename)
        np.save(save_dir + file, tensor.cpu().detach().numpy())
    else:
        cuda_seed = torch.cuda.initial_seed()
        file = '/{}_{}.npy'.format(filename, cuda_seed)
        np.save(save_dir + file, tensor.cpu().detach().numpy())
    if verbose:
        print("INFO: saving {} to file -> {}".format(tensor.shape, file))


def copy_parser(original_parser, description, modifications=False):
    new_parser = argparse.ArgumentParser(description=description)
    for action in original_parser._actions:
        if action.dest == 'help':
            continue
        kwargs = {'dest':action.dest, 'type':action.type, 'help':action.help, 'default':action.default, 'required':action.required}
        if modifications:
            if action.dest in modifications:
                kwargs.update(modifications[action.dest])
        new_parser.add_argument(action.option_strings[0], **kwargs)
    return new_parser


def serialize(obj, name_only=True):
    if callable(obj):
        if name_only:
            return obj.__name__
        else:
            return inspect.getsource(obj).strip()
    return obj

def args_to_json(args, name):
    args_dict = {a: serialize(b) for a, b in vars(args).items()}
    with open(args.workdir+'/'+name, 'w') as file: json.dump(args_dict, file, indent=4)


def get_gpu_memory():
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    free = total - reserved
    return {
        'total': total,
        'reserved': reserved,
        'allocated': allocated,
        'free': free
    }

def shuffle(tensor) -> torch.Tensor:
    indices = torch.randperm(tensor.size(0))
    return tensor[indices]

def fix_global_seed(seed):
    if seed is not None:
        torch.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False
    else:
        seed=torch.seed()
        torch.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed // 10**10)
        torch.backends.cudnn.deterministic = False 
        torch.backends.cudnn.benchmark = True
