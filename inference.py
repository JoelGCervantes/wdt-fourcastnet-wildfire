# libraries
import os, sys, time
import numpy as np
import h5py
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.YParams import YParams
from networks.afnonet import AFNONet
from collections import OrderedDict



def get_default_model_params():
    # We are going to use a default config. Please see github repo for other config examples
    config_file = "./FourCastNet/config/AFNO.yaml"
    config_name = "afno_backbone"
    params = YParams(config_file, config_name)
    return params
    
 

def visualize_2D_fields():    
    '''
    The ordering of atmospheric variables along the channel dimension is as follows:
    '''
    variables = ['u10', 'v10', 't2m', 'sp',
                 'msl', 't850', 'u1000',
                 'v1000', 'z1000', 'u850',
                 'v850', 'z850', 'u500',
                 'v500', 'z500', 't500',
                 'z50' , 'r500', 'r850', 'tcwv']
    
    sample_data = h5py.File(data_file, 'r')['fields']
    print('Total data shape:', sample_data.shape)
    timestep_idx = 0
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    for i, varname in enumerate(['u10', 't2m', 'z500', 'tcwv']):
        cm = 'bwr' if varname == 'u10' or varname == 'z500' else 'viridis'
        varidx = variables.index(varname)
        ax[i//2][i%2].imshow(sample_data[timestep_idx, varidx], cmap=cm)
        ax[i//2][i%2].set_title(varname)
    fig.tight_layout()

def load_model(model, params, checkpoint_file):
    ''' helper function to load model weights '''
    checkpoint_fname = checkpoint_file
    # Set weights_only=False to allow loading of trusted checkpoints from older PyTorch versions
    checkpoint = torch.load(checkpoint_fname, weights_only=False)
    try:
        ''' FourCastNet is trained with distributed data parallel
            (DDP) which prepends 'module' to all keys. Non-DDP
            models need to strip this prefix '''
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval() # set to inference mode
    return model


if __name__ == "__main__":
    
    # data and model paths
    data_path = "./ccai_demo/data/FCN_ERA5_data_v0/out_of_sample"
    data_file = os.path.join(data_path, "2018.h5")
    model_path = "./ccai_demo/model_weights/FCN_weights_v0/backbone.ckpt"
    global_means_path = "./ccai_demo/additional/stats_v0/global_means.npy"
    global_stds_path = "./ccai_demo/additional/stats_v0/global_stds.npy"
    time_means_path = "./ccai_demo/additional/stats_v0/time_means.npy"
    land_sea_mask_path = "./ccai_demo/additional/stats_v0/land_sea_mask.npy"
    
    # Get the model config from default configs
    sys.path.insert(1, './FourCastNet/') # insert code repo into path
    
    params = get_default_model_params()
    print("Model architecture used = {}".format(params["nettype"]))
    
    
    visualize_2D_fields()
    
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    # in and out channels: FourCastNet uses 20 input channels corresponding to 20 prognostic variables
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    params['N_in_channels'] = len(in_channels)
    params['N_out_channels'] = len(out_channels)
    params.means = np.load(global_means_path)[0, out_channels] # for normalizing data with precomputed train stats
    params.stds = np.load(global_stds_path)[0, out_channels]
    params.time_means = np.load(time_means_path)[0, out_channels]
    
    # load the model
    if params.nettype == 'afno':
        model = AFNONet(params).to(device)  # AFNO model
    else:
        raise Exception("not implemented")
    # load saved model weights
    model = load_model(model, params, model_path)
    model = model.to(device)
    
    
    
    
    
    