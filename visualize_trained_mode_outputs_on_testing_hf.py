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
             'frame_length_testing_hf': int(4.096*F_SAMPLING_HF),
             'stride_testing_hf': int(4.096*F_SAMPLING_HF ),
             'batch_size':32,


}

file_name_pre = 'mdl_Unetx8_axyz_11_Oct_10'
directory = '/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials'
fileObject = open(directory + '/Code Output/'+file_name_pre+'_pickle','rb')
(config, _ , _, _, _ ) = pickle.load(fileObject)
save_true = True


def visualize_signals_testing_hf(gen , sig_type_source ):

    torch.cuda.empty_cache()
    finished, X_batch, Y_batch, subject_id_list, recording_id_list= next(gen)

    if finished:
        print('Generator Finished')
    else:
        X_batch = torch.from_numpy(X_batch)
        X_batch = network_models.cuda(X_batch)
        X_batch = X_batch.type(torch.cuda.FloatTensor)

        Y_batch_predicted= sig_model.forward(X_batch).squeeze().detach().cpu().numpy()
        X_batch = X_batch.detach().cpu().numpy()


        no_sigs = X_batch.shape[1]
        for v in range(Y_batch.shape[0]):
            fig=plt.figure(figsize=(12,8))
            plt.subplot(no_sigs+1 , 1 , 1)
            plt.plot( (Y_batch[v, 0, :] - np.mean(Y_batch[v, 0, :]) )/( np.sqrt(np.sum(np.power(  Y_batch[v, 0, :] - np.mean(Y_batch[v, 0, :])  ,2))))  , '-r')
            plt.plot(( Y_batch_predicted[v,:]-np.mean(Y_batch_predicted[v,:]) )/(np.sqrt(np.sum(np.power(  Y_batch_predicted[v,:]-np.mean(Y_batch_predicted[v,:]) , 2  )))) ,
                     '-b')

            plt.title('Subject: ' + str(subject_id_list[v]) + ' Recording: ' + str(recording_id_list[v]) + ' Index in batch: ' + str(v))

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
                fig.savefig(directory + '/Code Output/' + file_name_pre + '_waveforms_testing_hf_' + str(v)+ '.png')

        del X_batch
        del Y_batch_predicted
        del Y_batch


#get testing HF subjects
all_recording_instances = utility.load_testing_hf_subjects(directory + '/Testing Data Wearable HF', store_in_ram=True )

#get training configs
eps =  config['eps']
directory= config['directory']
sig_type_source=config['sig_type_source']
sig_type_target=config['sig_type_target']
model_type=config['model_type']
normalized = True if model_type!='Unet_multiple_signal_in_not_normalized' else False
no_layers= config['no_layers']

#get testing configs
down_sample_factor_testing_hf=config_testing['down_sample_factor_testing_hf']
frame_length_testing_hf = config_testing['frame_length_testing_hf']
batch_size=config_testing['batch_size']
stride_testing_hf = config_testing['stride_testing_hf']
input_size = frame_length_testing_hf//down_sample_factor_testing_hf


test_gen_hf=data_generators.make_generator_multiple_signal_hf(list_of_recordings=all_recording_instances, batch_size=batch_size, eps=eps, frame_length=frame_length_testing_hf
                                  ,stride=stride_testing_hf, list_sig_type_source=sig_type_source, sig_type_target=sig_type_target
                                  , down_sample_factor=2, normalized=normalized, store_in_ram=True)

#check generator !!
#data_generators.diagnose_generator_multiple_signal_hf(test_gen_hf , sig_type_source+['ecg'])

#load model
model_path = directory + '/Code Output/'+ file_name_pre + '.pt'
model_type=config['model_type']
kernel_size=config['kernel_size']
filter_number=config['filter_number']
sig_model= network_models.load_saved_model(model_path , model_type , input_size , kernel_size  , filter_number=filter_number, signal_number=len(sig_type_source),
                                           no_layers=no_layers)


#visualize some predictions to check
visualize_signals_testing_hf(test_gen_hf ,sig_type_source)


