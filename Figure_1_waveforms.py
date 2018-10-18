import utility
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam
import torch
import data_generators
import network_models
import time
import numpy as np


F_SAMPLING=2000

config = {
             'file_name_pre': 'mdl_Unet4l_axyz_09_Oct_02',  #should be 25 letters
             'mode':'both',
             'eps': 1e-3,
             'model_type': 'Unet_4l',
             'down_sample_factor':4,
            'frame_length' : 4*F_SAMPLING,
             'kernel_size': 7 , #(3,5) , #5,#(3,5), #5, #(3, 5),#5
             'directory':'/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials',
             'cycle_per_batch':2,
             'filter_number':64,
            'sig_type_source': [ 'aX' , 'aY' , 'aZ'],
            'sig_type_target': 'bcg',
             'loss_func':'pearson_r',
             'produce_video' : True,
             'store_in_ram':True

}
#
file_name_pre = config['file_name_pre']
cycle_per_batch = config['cycle_per_batch']
mode= config['mode']
eps =  config['eps']
kernel_size = config['kernel_size']
directory= config['directory']
model_type = config['model_type']
filter_number=config['filter_number']
sig_type_source=config['sig_type_source']
sig_type_target=config['sig_type_target']
down_sample_factor = config['down_sample_factor']
frame_length=config['frame_length']
input_size = frame_length//down_sample_factor
normalized = True if model_type!='Unet_multiple_signal_in_not_normalized' else False
loss_func = config['loss_func']
produce_video= config['produce_video']
store_in_ram=config['store_in_ram']

#get all subject data
all_subject_instances = utility.load_subjects(directory + '/Training Data Analog Acc', store_in_ram )

#train test split
train_subject_instances, val_subject_instances = train_test_split( all_subject_instances  , test_size=0.2, random_state=42 )

#make a train and val generator
train_gen = data_generators.make_generator_multiple_signal(list_of_subjects=train_subject_instances, cycle_per_batch=cycle_per_batch, eps=eps,frame_length=frame_length,
                                    mode=mode, list_sig_type_source= sig_type_source, sig_type_target= sig_type_target , down_sample_factor =down_sample_factor,
                                                           normalized=normalized , store_in_ram=store_in_ram)

val_gen = data_generators.make_generator_multiple_signal(list_of_subjects=val_subject_instances, cycle_per_batch=cycle_per_batch, eps=eps, frame_length=frame_length,
                                    mode=mode, list_sig_type_source= sig_type_source, sig_type_target= sig_type_target, down_sample_factor=down_sample_factor,
                                                         normalized=normalized, store_in_ram=store_in_ram)


#pick and choose subject from dataset
def generate_and_plot(gen , list_sig_type_source):
    print('Diagnosing Generator..')
    X_batch, Y_batch, subject_id_list = next(gen)
    no_sigs = X_batch.shape[1]
    batch_size = X_batch.shape[0]
    print(X_batch.shape)
    for u in range(0, batch_size, 2):

        plt.figure()

        plt.subplot(no_sigs + 1, 1, 1)
        plt.plot(Y_batch[u, 0, :])
        plt.title(str(subject_id_list[u]) + ' ' + str(u) )

        for v in range(2, no_sigs + 2):
            plt.subplot(no_sigs + 1, 1, v)
            plt.plot(X_batch[u, v - 2, :])
            plt.title(list_sig_type_source[v - 2])

        plt.tight_layout()
        plt.show()

    return X_batch, Y_batch

#generate a batch
# X_batch, Y_batch = generate_and_plot(train_gen, sig_type_source)
# np.save('/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/X_batch', X_batch)
# np.save('/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/Y_batch', Y_batch)



#draw figure
X_batch=np.load('/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/X_batch.npy')
Y_batch = np.load('/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/Y_batch.npy')
u=10
plt.figure()
plt.subplot(2 , 1, 2)
plt.plot(Y_batch[u, 0, :] , color='xkcd:blue' , linewidth=2)
plt.xlim([0,2000])
plt.ylim([0,1])
frame1=plt.gca()
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)

plt.subplot(2 , 1, 1)
plt.plot(X_batch[u, 0, :] , color='xkcd:vibrant green' , linewidth=1.5)
plt.plot(X_batch[u, 1, :]-1, color='xkcd:vibrant green' , linewidth=1.5)
plt.plot(X_batch[u, 2, :]-2, color='xkcd:vibrant green' , linewidth=1.5)
plt.xlim([0,2000])
plt.ylim([-2,1])
frame1=plt.gca()
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)
plt.tight_layout()

plt.savefig("/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/Figure_1_waveforms.svg")
