import utility
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam
import torch
import network_models
import tqdm as tqdm
import numpy as np
from sklearn import manifold
from sklearn import preprocessing
import data_generators





F_SAMPLING_HF=1000

#configs for testing
config_testing = {

             'down_sample_factor_testing_hf':2,
             'frame_length_testing_hf': 4*F_SAMPLING_HF,
             'stride_testing_hf':F_SAMPLING_HF // 10,
             'batch_size':32,


}

file_name_pre = 'mdl_Unet4l_axyz_09_Oct_02'
directory = '/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials'
fileObject = open(directory + '/Code Output/'+file_name_pre+'_pickle','rb')
(config, _ , _, _, _ ) = pickle.load(fileObject)

#get testing HF subjects
all_recording_instances = utility.load_testing_hf_subjects(directory + '/Testing Data Wearable HF', store_in_ram=True )

#get training configs
eps =  config['eps']
directory= config['directory']
sig_type_source=config['sig_type_source']
sig_type_target=config['sig_type_target']
model_type=config['model_type']
normalized = True if model_type!='Unet_multiple_signal_in_not_normalized' else False

#get testing configs
down_sample_factor_testing_hf=config_testing['down_sample_factor_testing_hf']
frame_length_testing_hf = config_testing['frame_length_testing_hf']
batch_size=config_testing['batch_size']
stride_testing_hf = config_testing['stride_testing_hf']
input_size = frame_length_testing_hf//down_sample_factor_testing_hf


test_gen_hf=data_generators.make_generator_multiple_signal_hf(list_of_recordings=all_recording_instances, batch_size=batch_size, eps=eps, frame_length=frame_length_testing_hf
                                  ,stride=stride_testing_hf, list_sig_type_source=sig_type_source+['ecg'], sig_type_target=sig_type_target
                                  , down_sample_factor=2, normalized=normalized, store_in_ram=True)

#check generator !!
#data_generators.diagnose_generator_multiple_signal_hf(test_gen_hf , sig_type_source+['ecg'])

#load model
model_path = directory + '/Code Output/'+ file_name_pre + '.pt'
model_type=config['model_type']
kernel_size=config['kernel_size']
filter_number=config['filter_number']
sig_model= network_models.load_saved_model(model_path , model_type , input_size , kernel_size  , filter_number=filter_number, signal_number=len(sig_type_source))




