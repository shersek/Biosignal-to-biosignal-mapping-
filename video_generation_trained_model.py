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
import os
import matplotlib.animation as animation


file_name_pre = 'mdl_Unetx1_axyz_11_Oct_07'
directory = '/my_directory'
fileObject = open(directory + '/Code Output/'+file_name_pre+'_pickle','rb')
(config, train_history , valid_history, train_subject_instances, val_subject_instances ) = pickle.load(fileObject)
number_of_videos=2
fps=5
dpi=400

def visualize_signals_video(gen ,
                  sig_type_source ,
                  directory,
                  model_path_for_video,
                  model_type ,
                   no_layers,
                  input_size  ,
                  kernel_size  ,
                  filter_number ,
                  video_name):
    list_all_files = os.listdir(directory+ '/Models for Video/')
    list_models= [file for file in list_all_files if file.startswith(model_path_for_video[-25::]) ]
    no_epochs = len(list_models)

    torch.cuda.empty_cache()
    X_batch, Y_batch, list_subjects = next(gen)

    X_batch = torch.from_numpy(X_batch)
    X_batch = network_models.cuda(X_batch)
    X_batch = X_batch.type(torch.cuda.FloatTensor)

    v = np.random.randint(0, Y_batch.shape[0])
    X_batch_detached = X_batch.detach().cpu().numpy()
    no_sigs = X_batch_detached.shape[1]
    fig = plt.figure(figsize=(12, 8))


    def animate(i):
    #for epoch in range(1,no_epochs+1):
        epoch=i+1
        if epoch < 10:
            epoch_str = '000' + str(epoch)
        elif epoch >= 10 and epoch < 100:
            epoch_str = '00' + str(epoch)
        elif epoch >= 100 and epoch < 1000:
            epoch_str = '0' + str(epoch)
        else:
            epoch_str = str(epoch)
        model_path = model_path_for_video+'_'+epoch_str+'.pt'
        print('Loading...'+model_path)
        sig_model = network_models.load_saved_model(model_path, model_type, input_size, kernel_size,
                                                    filter_number=filter_number, signal_number=len(sig_type_source), no_layers=no_layers)

        Y_batch_predicted= sig_model.forward(X_batch).squeeze().detach().cpu().numpy()

        #for v in range(Y_batch.shape[0]):
        plt.clf()
        plt.subplot(no_sigs+1 , 1 , 1)
        #plt.plot(Y_batch[v,0,:]  , '-r')
        plt.plot( (Y_batch[v, 0, :] - np.mean(Y_batch[v, 0, :]) )/( np.sqrt(np.sum(np.power(  Y_batch[v, 0, :] - np.mean(Y_batch[v, 0, :])  ,2))))  , '-r')
        #plt.plot((Y_batch_predicted[v,:]-np.min(Y_batch_predicted[v,:]) )/(np.max(Y_batch_predicted[v,:])-np.min(Y_batch_predicted[v,:])) , '-b')
        plt.plot(( Y_batch_predicted[v,:]-np.mean(Y_batch_predicted[v,:]) )/(np.sqrt(np.sum(np.power(  Y_batch_predicted[v,:]-np.mean(Y_batch_predicted[v,:]) , 2  )))) ,
                 '-b')
        #plt.plot(Y_batch_predicted[v, :] , '-b')
        # plt.title('Subject Number: ' + str(list_subjects[v]) +' , Epoch: ' + str(epoch)
        #           + ', Train Loss=' + str(train_history[epoch-1]['train_loss']) + ', Val Loss=' + str(valid_history[epoch-1]['valid_loss']) )

        plt.title(   'Subject Number: {}, Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(list_subjects[v], epoch,
                                                                                 train_history[epoch - 1]['train_loss'],
                                                                                 valid_history[epoch - 1]['valid_loss']))



        for k in range(2,no_sigs+2):
            plt.subplot( no_sigs+1 , 1 , k )
            plt.plot(X_batch_detached[v,k-2,:]  , '-g')
            plt.plot((Y_batch_predicted[v, :] - np.min(Y_batch_predicted[v, :])) / (
                        np.max(Y_batch_predicted[v, :]) - np.min(Y_batch_predicted[v, :])), '-b')
            #plt.plot(Y_batch_predicted[v,:] , '-b')
            plt.title(sig_type_source[k-2])

        # plt.tight_layout()
        # plt.show()
        # if save_true:
        #     fig.savefig(directory + '/' + file_name_pre + '_waveforms_' + str(v)+ '.png')
        del Y_batch_predicted

    ani = animation.FuncAnimation(fig, animate, frames=no_epochs, repeat=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(directory+'/Code Output/' + video_name+ '.mp4', writer=writer, dpi=dpi)
    del Y_batch
    del X_batch

cycle_per_batch = config['cycle_per_batch']
mode= config['mode']
eps =  config['eps']
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
                                                           normalized=normalized , store_in_ram=store_in_ram)

val_gen = data_generators.make_generator_multiple_signal(list_of_subjects=val_subject_instances, cycle_per_batch=cycle_per_batch, eps=eps, frame_length=frame_length,
                                    mode=mode, list_sig_type_source= sig_type_source, sig_type_target= sig_type_target, down_sample_factor=down_sample_factor,
                                                         normalized=normalized, store_in_ram=store_in_ram)



#check !
#data_generators.diagnose_generator_multiple_signal(train_gen , sig_type_source)
#data_generators.diagnose_generator_multiple_signal(val_gen , sig_type_source)



#load model
model_path_for_video = config['model_path_for_video'] #directory + '/' + file_name_pre + '.pt'
model_type=config['model_type']
kernel_size=config['kernel_size']
filter_number=config['filter_number']

#generate videos
for u in range(number_of_videos):
    video_name= file_name_pre + '_' + str(u)


    #visualize embeddings on training set
    visualize_signals_video(gen=train_gen ,
                      sig_type_source=sig_type_source ,
                      directory=directory,
                      model_path_for_video=model_path_for_video,
                      model_type=model_type ,
                      no_layers=no_layers,
                      input_size = input_size ,
                      kernel_size=kernel_size  ,
                      filter_number=filter_number,
                      video_name = video_name + '_train')

    #visualize embeddings on validation set
    visualize_signals_video(gen=val_gen ,
                      sig_type_source=sig_type_source ,
                      directory=directory,
                      model_path_for_video=model_path_for_video,
                       model_type=model_type ,
                        no_layers=no_layers,
                        input_size = input_size ,
                      kernel_size=kernel_size  ,
                      filter_number=filter_number,
                       video_name=video_name + '_val')

    #TODO: mutli-signal in multi-signal out
