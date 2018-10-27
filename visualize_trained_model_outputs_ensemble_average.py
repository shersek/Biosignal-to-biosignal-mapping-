import matplotlib.pyplot as plt
import pickle
import torch
import network_models
import numpy as np
import data_generators
import signal_processing_modules
from scipy import signal

#load model and workspace items
file_name_pre = 'mdl_Unetx5_axyz_23_Oct_01'
directory = '/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials'
fileObject = open(directory + '/Code Output/'+file_name_pre+'_pickle','rb')
(config, train_history , valid_history, train_subject_instances, val_subject_instances , best_val) = pickle.load(fileObject)
save_true = False
upsample_factor = 1

def visualize_ijk(gen  , upsample_factor, no_cycles):
    '''
    plots target signal segments and estimated target signal segments, ensemble averaged with the i,j,k points
    :param gen: data generator
    :param sig_type_source: list of source signals e.g. ['aX', 'aY', 'aZ'] for all three axes of the accelerometer as source signals
    :return: -
    '''
    torch.cuda.empty_cache()
    X_batch, Y_batch, list_subjects = next(gen)
    ecg = X_batch[:,-1,:]

    X_batch = torch.from_numpy(X_batch[:,:-1,:]).contiguous()
    X_batch = network_models.cuda(X_batch)
    X_batch = X_batch.type(torch.cuda.FloatTensor)

    Y_batch_predicted= sig_model.forward(X_batch).squeeze().detach().cpu().numpy()
    X_batch = X_batch.detach().cpu().numpy()

    list_i_errors = []
    list_j_errors = []
    list_k_errors = []
    for _ in range(no_cycles):
        for v in range(Y_batch.shape[0]):

            r_peaks = signal_processing_modules.get_R_peaks(ecg[v,:])
            ensemble_avg_target, ensemble_beats_target = signal_processing_modules.get_ensemble_avg(r_peaks, Y_batch[v, 0, :], n_samples=500 , upsample_factor=upsample_factor)
            i_point_target, j_point_target, k_point_target = signal_processing_modules.get_IJK_peaks(ensemble_avg_target, upsample_factor=upsample_factor)

            ensemble_avg_estimate, ensemble_beats_estimate = signal_processing_modules.get_ensemble_avg(r_peaks, Y_batch_predicted[v, :], n_samples=500 , upsample_factor=upsample_factor)
            i_point_estimate, j_point_estimate, k_point_estimate = signal_processing_modules.get_IJK_peaks(ensemble_avg_estimate, upsample_factor=upsample_factor)

            i_error = 1000*np.abs(i_point_target-i_point_estimate)/(upsample_factor*500) if i_point_target!=-1 and i_point_estimate!=-1 else -1 #500 is the sampling rate of the signal segments, 1000* for miliseconds
            j_error = 1000*np.abs(j_point_target-j_point_estimate)/(upsample_factor*500) if j_point_target!=-1 and j_point_estimate!=-1 else -1
            k_error = 1000*np.abs(k_point_target-k_point_estimate)/(upsample_factor*500) if k_point_target!=-1 and k_point_estimate!=-1 else -1



            fig=plt.figure(figsize=(12,8))
            plt.subplot(2 , 1 , 1)

            plt.plot(ensemble_beats_target.T, 'r', alpha=0.3)
            plt.plot(ensemble_avg_target, '-r', linewidth=4)
            plt.plot(i_point_target, ensemble_avg_target[i_point_target], 'ok')
            plt.text(i_point_target, ensemble_avg_target[i_point_target] +0.05, 'I', color='red')
            plt.plot(j_point_target, ensemble_avg_target[j_point_target], 'ok')
            plt.text(j_point_target, ensemble_avg_target[j_point_target] +0.05, 'J', color='red')
            plt.plot(k_point_target, ensemble_avg_target[k_point_target], 'ok')
            plt.text(k_point_target, ensemble_avg_target[k_point_target] +0.05, 'K', color='red')

            plt.title('Subject Number: ' + str(list_subjects[v]))

            plt.subplot(2, 1, 2)

            plt.plot(ensemble_beats_estimate.T, 'b', alpha=0.3)
            plt.plot(ensemble_avg_estimate, '-b', linewidth=4)
            plt.plot(i_point_estimate, ensemble_avg_estimate[i_point_estimate], 'ok')
            plt.text(i_point_estimate, ensemble_avg_estimate[i_point_estimate] +0.05, 'I', color='blue')
            plt.plot(j_point_estimate, ensemble_avg_estimate[j_point_estimate], 'ok')
            plt.text(j_point_estimate, ensemble_avg_estimate[j_point_estimate] +0.05, 'J', color='blue')
            plt.plot(k_point_estimate, ensemble_avg_estimate[k_point_estimate], 'ok')
            plt.text(k_point_estimate, ensemble_avg_estimate[k_point_estimate] +0.05, 'K', color='blue')

            plt.title('I-error=' + str(i_error) +  ' | J-error=' + str(j_error) + ' | K-error=' + str(k_error) + ' in (ms)')

            plt.tight_layout()
            plt.show()
            if save_true:
                fig.savefig(directory + '/Code Output/' + file_name_pre + '_waveforms_ensemble_averaged' + str(v)+ '.png')

            if i_error != -1:
                list_i_errors.append(i_error)
            if j_error != -1:
                list_j_errors.append(j_error)
            if k_error!=-1:
                list_k_errors.append(k_error)

    print('MAE I-Point (ms): ' + str(np.mean(np.array(list_i_errors))))
    print('MAE J-Point (ms): ' + str(np.mean(np.array(list_j_errors))))
    print('MAE K-Point (ms): ' + str(np.mean(np.array(list_k_errors))))


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
                                    mode=mode, list_sig_type_source= sig_type_source+['ecg'], sig_type_target= sig_type_target , down_sample_factor =down_sample_factor,
                                                           normalized=normalized, store_in_ram=store_in_ram)

val_gen = data_generators.make_generator_multiple_signal(list_of_subjects=val_subject_instances, cycle_per_batch=cycle_per_batch, eps=eps, frame_length=frame_length,
                                    mode=mode, list_sig_type_source= sig_type_source+['ecg'], sig_type_target= sig_type_target, down_sample_factor=down_sample_factor,
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
visualize_ijk(train_gen  , upsample_factor , 10)

#visualize embeddings on validation set
visualize_ijk(val_gen  , upsample_factor , 10)
