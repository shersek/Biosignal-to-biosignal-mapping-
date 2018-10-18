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

file_name_pre = 'mdl_Unet4l_axyz_09_Oct_02'
directory = '/my_directory'
fileObject = open(directory + '/Code Output/'+file_name_pre+'_pickle','rb')
(config, train_history , valid_history, train_subject_instances, val_subject_instances ) = pickle.load(fileObject)
save_true = True

def visualize_signals(gen , sig_type_source ):

    torch.cuda.empty_cache()
    X_batch, Y_batch, list_subjects = next(gen)
    ecg = X_batch[:,3,:]
    X_batch = X_batch[:,0:-1,:]

    X_batch = torch.from_numpy(X_batch)
    X_batch = network_models.cuda(X_batch.contiguous())
    X_batch = X_batch.type(torch.cuda.FloatTensor)


    Y_batch_predicted= sig_model.forward(X_batch).squeeze().detach().cpu().numpy()
    X_batch = X_batch.detach().cpu().numpy()

    # Y_batch_predicted=-Y_batch_predicted

    no_sigs = X_batch.shape[1]
    for v in range(Y_batch.shape[0]):
        fig=plt.figure(figsize=(12,8))
        plt.subplot(no_sigs+2 , 1 , 1)
        #plt.plot(Y_batch[v,0,:]  , '-r')
        plt.plot( (Y_batch[v, 0, :] - np.mean(Y_batch[v, 0, :]) )/( np.sqrt(np.sum(np.power(  Y_batch[v, 0, :] - np.mean(Y_batch[v, 0, :])  ,2))))  , '-r')
        #plt.plot((Y_batch_predicted[v,:]-np.min(Y_batch_predicted[v,:]) )/(np.max(Y_batch_predicted[v,:])-np.min(Y_batch_predicted[v,:])) , '-b')
        plt.plot(( Y_batch_predicted[v,:]-np.mean(Y_batch_predicted[v,:]) )/(np.sqrt(np.sum(np.power(  Y_batch_predicted[v,:]-np.mean(Y_batch_predicted[v,:]) , 2  )))) ,
                 '-b')

        #plt.plot(Y_batch_predicted[v, :] , '-b')
        plt.title('Subject Number: ' + str(list_subjects[v]) + ' ' + str(v) )

        for k in range(2,no_sigs+2):
            plt.subplot( no_sigs+2 , 1 , k )
            plt.plot(X_batch[v,k-2,:]  , '-g')
            plt.plot((Y_batch_predicted[v, :] - np.min(Y_batch_predicted[v, :])) / (
                        np.max(Y_batch_predicted[v, :]) - np.min(Y_batch_predicted[v, :])), '-b')

            #plt.plot(Y_batch_predicted[v,:] , '-b')
            plt.title(sig_type_source[k-2])

        plt.subplot(no_sigs + 2, 1, no_sigs + 2 )
        plt.plot(ecg[v])
        plt.tight_layout()
        plt.show()



    return X_batch, Y_batch , Y_batch_predicted , ecg




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

#make a train and val generator
train_gen = data_generators.make_generator_multiple_signal(list_of_subjects=train_subject_instances, cycle_per_batch=cycle_per_batch, eps=eps,frame_length=frame_length,
                                    mode=mode, list_sig_type_source= ['aX' , 'aY', 'aZ' , 'ecg'], sig_type_target= sig_type_target , down_sample_factor =down_sample_factor,
                                                           normalized=normalized, store_in_ram=store_in_ram)

val_gen = data_generators.make_generator_multiple_signal(list_of_subjects=val_subject_instances, cycle_per_batch=cycle_per_batch, eps=eps, frame_length=frame_length,
                                    mode=mode, list_sig_type_source= ['aX' , 'aY', 'aZ' , 'ecg' ], sig_type_target= sig_type_target, down_sample_factor=down_sample_factor,
                                                         normalized=normalized, store_in_ram=store_in_ram)



#check !
#data_generators.diagnose_generator_multiple_signal(train_gen , sig_type_source)
#data_generators.diagnose_generator_multiple_signal(val_gen , sig_type_source)



#load model
model_path = directory + '/Code Output/'+ file_name_pre + '.pt'
model_type=config['model_type']
kernel_size=config['kernel_size']
filter_number=config['filter_number']
sig_model= network_models.load_saved_model(model_path , model_type , input_size , kernel_size  , filter_number=filter_number, signal_number=len(sig_type_source))


#visualize embeddings on training set
# X_batch, Y_batch , Y_batch_predicted , ecg= visualize_signals(train_gen ,sig_type_source)
# np.save('/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/X_batch_3', X_batch)
# np.save('/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/Y_batch_3', Y_batch)
# np.save('/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/Y_batch_predicted_3', Y_batch_predicted)
# np.save('/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/ecg_3', ecg)



#draw figure
X_batch=np.load('/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/X_batch_3.npy')
Y_batch = np.load('/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/Y_batch_3.npy')
Y_batch_predicted=np.load('/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/Y_batch_predicted_3.npy')
ecg = np.load('/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/ecg_3.npy')


u=1
plt.figure(figsize=(12,8))

plt.subplot(4 , 1, 1)
plt.plot(ecg[u, :] , color='xkcd:cherry red' , linewidth=2)
plt.xlim([0,2000])
plt.ylim([0,1])
frame1=plt.gca()
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)

plt.subplot(4 , 1, 2)
plt.plot(Y_batch[u, 0, :] , color='xkcd:blue' , linewidth=2)
plt.xlim([0,2000])
plt.ylim([0,1])
frame1=plt.gca()
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)

plt.subplot(4 , 1, 3)
plt.plot((Y_batch_predicted[u, :]-np.min(Y_batch_predicted[u, :]))/(np.max(Y_batch_predicted[u, :]) - np.min(Y_batch_predicted[u, :])) , color='xkcd:pumpkin orange' , linewidth=2)
plt.xlim([0,2000])
plt.ylim([0,1])
frame1=plt.gca()
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)

plt.subplot(4 , 1, 4)
plt.plot(X_batch[u, 0, :] , color='xkcd:vibrant green' , linewidth=1.5)
plt.plot(X_batch[u, 1, :]-1, color='xkcd:vibrant green' , linewidth=1.5)
plt.plot(X_batch[u, 2, :]-2, color='xkcd:vibrant green' , linewidth=1.5)
plt.xlim([0,2000])
plt.ylim([-2,1])
frame1=plt.gca()
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)
plt.tight_layout()

plt.savefig("/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/Figure_3_waveforms.svg")
