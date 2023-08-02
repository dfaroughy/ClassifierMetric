import numpy as np
import os.path
import shutil
import itertools
import json

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
        info = "INFO: created directory: {}".format(Directory)
        print("#"+"="*len(info))
        print(info)
        print("#"+"="*len(info))
    return Directory


class GetConfigs:
    def __init__(self, path):
        with open(path, 'r') as f:
            params = json.load(f)
        config_key, config = list(params.items())[0]
        self.config_key = config_key
        for key, value in config.items():
            setattr(self, key, value)

def save_data(samples: dict, name: str, workdir : str, verbose: bool = True):
    for key in samples.keys():
        sample = samples[key].numpy()
        path = '{}/results/{}_{}.npy'.format(workdir, name, key) 
        np.save(path, sample)
    if verbose:
        print("INFO: saved {} data in {}".format(name, workdir))

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