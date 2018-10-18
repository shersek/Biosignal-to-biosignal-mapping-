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

file_name_pre = 'mdl_Unetx1_axyz_11_Oct_07'
directory =   '/my_directory'
fileObject = open(directory + '/Code Output/'+file_name_pre+'_pickle','rb')
(config, train_history , valid_history, train_subject_instances, val_subject_instances ) = pickle.load(fileObject)


def visualize_signals(gen , sig_type_source ):

    torch.cuda.empty_cache()
    X_batch, tt, freq_vector = next(gen)

    X_batch = torch.from_numpy(X_batch)
    X_batch = network_models.cuda(X_batch)
    X_batch = X_batch.type(torch.cuda.FloatTensor)

    Y_batch_predicted= sig_model.forward(X_batch).squeeze().detach().cpu().numpy()
    X_batch = X_batch.detach().cpu().numpy()

    offset = len(tt)//2

    no_sigs = X_batch.shape[1]
    for v in range(X_batch.shape[0]):
        plt.figure()
        for k in range(1,no_sigs+1):
            inp = X_batch[v,k-2,:]
            out = (Y_batch_predicted[v, :] - np.min(Y_batch_predicted[v, :])) / (
                        np.max(Y_batch_predicted[v, :]) - np.min(Y_batch_predicted[v, :]))
            time_ax=tt
            time_ax = time_ax if freq_vector[v]<3 else time_ax[ offset:offset+  int(len(tt)//freq_vector[v])   ]
            inp = inp if freq_vector[v]<3 else inp[ offset:offset+  int(len(tt)//freq_vector[v])   ]
            out = out if freq_vector[v] < 3 else out[offset:offset + int(len(tt) // freq_vector[v])]

            plt.subplot( no_sigs , 1 , k )
            plt.plot(time_ax,inp , '-g')
            plt.plot(time_ax,out, '-b')
            #plt.plot(Y_batch_predicted[v,:] , '-b')
            plt.title(sig_type_source[k-2] + ' , ' + str(freq_vector[v]))

        plt.tight_layout()
        plt.show()


    del X_batch
    del Y_batch_predicted



cycle_per_batch = config['cycle_per_batch']
mode= config['mode']
eps =  config['eps']
# sig_type= config['sig_type']
directory= config['directory']
# representation= config['representation']
sig_type_source=config['sig_type_source']
sig_type_target=config['sig_type_target']
down_sample_factor = config['down_sample_factor']
frame_length=config['frame_length']
input_size = frame_length//down_sample_factor
model_type=config['model_type']
normalized = True if model_type!='Unet_multiple_signal_in_not_normalized' else False
no_layers= config['no_layers']


#make a train and val generator
sine_gen = data_generators.make_generator_multiple_signal_sine_wave(batch_size=32,
                                                    freq_lo=1,
                                                    freq_hi=100,
                                                    eps=eps,
                                                    frame_length=frame_length,
                                                    list_sig_type_source= sig_type_source,
                                                    down_sample_factor=down_sample_factor )


# X_batch ,tt , freq_vector =next(sine_gen)
# for u in range(0,X_batch.shape[0],4):
#     plt.figure()
#     plt.plot(tt,X_batch[u,0,:])
#     plt.title(str(freq_vector[u]))

#check !
#data_generators.diagnose_generator_multiple_signal(train_gen , sig_type_source)
#data_generators.diagnose_generator_multiple_signal(val_gen , sig_type_source)



#load model
model_path = directory + '/Code Output/' + file_name_pre + '.pt'
model_type=config['model_type']
kernel_size=config['kernel_size']
filter_number=config['filter_number']
sig_model= network_models.load_saved_model(model_path , model_type , input_size , kernel_size  , filter_number=filter_number, signal_number=len(sig_type_source),
                                           no_layers=no_layers)


#visualize embeddings on training set
visualize_signals(sine_gen ,sig_type_source)

