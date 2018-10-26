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

#load model and workspace items
file_name_pre = 'mdl_Unetx5_axyz_23_Oct_01'
directory = '/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials'
fileObject = open(directory + '/Code Output/'+file_name_pre+'_pickle','rb')
(config, train_history , valid_history, train_subject_instances, val_subject_instances , best_val) = pickle.load(fileObject)
save_true = False

def visualize_signals(gen , sig_type_source ):
    '''
    plots target signal segments, source signal segments and the estimated target signal segments together
    :param gen: data generator
    :param sig_type_source: list of source signals e.g. ['aX', 'aY', 'aZ'] for all three axes of the accelerometer as source signals
    :return: -
    '''
    torch.cuda.empty_cache()
    X_batch, Y_batch, list_subjects = next(gen)

    X_batch = torch.from_numpy(X_batch)
    X_batch = network_models.cuda(X_batch)
    X_batch = X_batch.type(torch.cuda.FloatTensor)

    Y_batch_predicted= sig_model.forward(X_batch).squeeze().detach().cpu().numpy()
    X_batch = X_batch.detach().cpu().numpy()

    # Y_batch_predicted=-Y_batch_predicted

    no_sigs = X_batch.shape[1]
    for v in range(Y_batch.shape[0]):
        fig=plt.figure(figsize=(12,8))
        plt.subplot(no_sigs+1 , 1 , 1)
        #plt.plot(Y_batch[v,0,:]  , '-r')
        plt.plot( (Y_batch[v, 0, :] - np.mean(Y_batch[v, 0, :]) )/( np.sqrt(np.sum(np.power(  Y_batch[v, 0, :] - np.mean(Y_batch[v, 0, :])  ,2))))  , '-r')
        #plt.plot((Y_batch_predicted[v,:]-np.min(Y_batch_predicted[v,:]) )/(np.max(Y_batch_predicted[v,:])-np.min(Y_batch_predicted[v,:])) , '-b')
        plt.plot(( Y_batch_predicted[v,:]-np.mean(Y_batch_predicted[v,:]) )/(np.sqrt(np.sum(np.power(  Y_batch_predicted[v,:]-np.mean(Y_batch_predicted[v,:]) , 2  )))) ,
                 '-b')

        #plt.plot(Y_batch_predicted[v, :] , '-b')
        plt.title('Subject Number: ' + str(list_subjects[v]))

        for k in range(2,no_sigs+2):
            plt.subplot( no_sigs+1 , 1 , k )
            plt.plot(X_batch[v,k-2,:]  , '-g')
            plt.plot((Y_batch_predicted[v, :] - np.min(Y_batch_predicted[v, :])) / (
                        np.max(Y_batch_predicted[v, :]) - np.min(Y_batch_predicted[v, :])), '-b')

            #plt.plot(Y_batch_predicted[v,:] , '-b')
            plt.title(sig_type_source[k-2])

        plt.tight_layout()
        plt.show()
        if save_true:
            fig.savefig(directory + '/Code Output/' + file_name_pre + '_waveforms_' + str(v)+ '.png')

    del X_batch
    del Y_batch_predicted
    del Y_batch


#configs
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
store_in_ram = config['store_in_ram'] if 'store_in_ram' in config else False
no_layers = config['no_layers']

#make a train and val generator
train_gen = data_generators.make_generator_multiple_signal(list_of_subjects=train_subject_instances, cycle_per_batch=cycle_per_batch, eps=eps,frame_length=frame_length,
                                    mode=mode, list_sig_type_source= sig_type_source, sig_type_target= sig_type_target , down_sample_factor =down_sample_factor,
                                                           normalized=normalized, store_in_ram=store_in_ram)

val_gen = data_generators.make_generator_multiple_signal(list_of_subjects=val_subject_instances, cycle_per_batch=cycle_per_batch, eps=eps, frame_length=frame_length,
                                    mode=mode, list_sig_type_source= sig_type_source, sig_type_target= sig_type_target, down_sample_factor=down_sample_factor,
                                                         normalized=normalized, store_in_ram=store_in_ram)



#check !
#data_generators.diagnose_generator_multiple_signal(train_gen , sig_type_source)
#data_generators.diagnose_generator_multiple_signal(val_gen , sig_type_source)



#load model
model_path = directory + '/Code Output/'+ file_name_pre + '.pt'
model_type=config['model_type']
kernel_size=config['kernel_size']
filter_number=config['filter_number']
sig_model= network_models.load_saved_model(model_path , model_type , input_size , kernel_size  , filter_number=filter_number, signal_number=len(sig_type_source),
                                           no_layers = no_layers)

#visualize embeddings on training set
visualize_signals(train_gen ,sig_type_source)

#visualize embeddings on validation set
visualize_signals(val_gen  , sig_type_source)

