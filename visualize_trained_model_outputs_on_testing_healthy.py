import utility
import matplotlib.pyplot as plt
import pickle
import torch
import network_models
import numpy as np
from sklearn import manifold
from sklearn import preprocessing
import data_generators
import signal_processing_modules

F_SAMPLING_TESTING=2000

#configs for testing
config_testing = {

             'frame_length_testing': int(4.096*F_SAMPLING_TESTING),
             'stride_testing': int(4.096*F_SAMPLING_TESTING ),
             'batch_size':32,

}

file_name_pre = 'mdl_Unetx6_axyz_29_Oct_01'
directory = '/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials'
fileObject = open(directory + '/Code Output/'+file_name_pre+'_pickle','rb')
(config, _ , _, _, _ , _) = pickle.load(fileObject)
save_true = False


def visualize_signals_testing(gen , sig_type_source ):
    '''
    visualize the estimated and target signals on a minibatch of testing data
    :param gen: test generator
    :param sig_type_source: list of str of source signals: e.g. ['aX', 'aY', 'aZ'] for all three axes of the acceleroemter
    :return: -
    '''
    torch.cuda.empty_cache()
    finished, X_batch, Y_batch, subject_id_list= next(gen)

    if finished:
        print('Generator Finished')
    else:
        X_batch = torch.from_numpy(X_batch)
        X_batch = network_models.cuda(X_batch)
        X_batch = X_batch.type(torch.cuda.FloatTensor)

        Y_batch_predicted= sig_model.forward(X_batch).squeeze().detach().cpu().numpy()
        X_batch = X_batch.detach().cpu().numpy()


        no_sigs = X_batch.shape[1]
        for v in range(0,Y_batch.shape[0],4):
            fig=plt.figure(figsize=(12,8))
            plt.subplot(no_sigs+1 , 1 , 1)
            plt.plot( (Y_batch[v, 0, :] - np.mean(Y_batch[v, 0, :]) )/( np.sqrt(np.sum(np.power(  Y_batch[v, 0, :] - np.mean(Y_batch[v, 0, :])  ,2))))  , '-r')
            plt.plot( ( Y_batch_predicted[v,:]-np.mean(Y_batch_predicted[v,:]) )/(np.sqrt(np.sum(np.power(  Y_batch_predicted[v,:]-np.mean(Y_batch_predicted[v,:]) , 2  )))) ,
                     '-b')

            plt.title('Subject: ' + str(subject_id_list[v]) + ' Index in batch: ' + str(v))

            for k in range(2,no_sigs+2):
                plt.subplot( no_sigs+1 , 1 , k )
                plt.plot(X_batch[v,k-2,:]  , '-g')
                plt.plot( (Y_batch_predicted[v, :] - np.min(Y_batch_predicted[v, :])) / (
                            np.max(Y_batch_predicted[v, :]) - np.min(Y_batch_predicted[v, :])) , '-b')

                #plt.plot(Y_batch_predicted[v,:] , '-b')
                plt.title(sig_type_source[k-2])

            plt.tight_layout()
            plt.show()
            if save_true:
                fig.savefig(directory + '/Code Output/' + file_name_pre + '_waveforms_testing_hf_' + str(v)+ '.png')

        del X_batch
        del Y_batch_predicted
        del Y_batch



def visualize_ijk_test(gen  , upsample_factor, no_cycles):
    '''
    plots target signal segments and estimated target signal segments, ensemble averaged with the i,j,k points
    :param gen: data generator
    :param sig_type_source: list of source signals e.g. ['aX', 'aY', 'aZ'] for all three axes of the accelerometer as source signals
    :return: -
    '''
    torch.cuda.empty_cache()
    finished, X_batch, Y_batch, subject_id_list = next(gen)
    ecg = X_batch[:,-1,:]

    X_batch = torch.from_numpy(X_batch[:,:-1,:]).contiguous()
    X_batch = network_models.cuda(X_batch)
    X_batch = X_batch.type(torch.cuda.FloatTensor)

    Y_batch_predicted= sig_model.forward(X_batch).squeeze().detach().cpu().numpy()
    X_batch = X_batch.detach().cpu().numpy()

    list_i_errors = []
    list_j_errors = []
    list_k_errors = []
    list_noise_var_target  = []
    list_noise_var_estimate = []
    for _ in range(no_cycles):
        for v in range(Y_batch.shape[0]):

            r_peaks = signal_processing_modules.get_R_peaks(ecg[v,:])
            ensemble_avg_target, ensemble_beats_target = signal_processing_modules.get_ensemble_avg(r_peaks,
                                                                                (Y_batch[v, 0, :] - np.mean(Y_batch[v, 0, :]) )/( np.sqrt(np.sum(np.power(  Y_batch[v, 0, :] - np.mean(Y_batch[v, 0, :])  ,2)))) ,
                                                                                n_samples=500 , upsample_factor=upsample_factor)
            i_point_target, j_point_target, k_point_target = signal_processing_modules.get_IJK_peaks(ensemble_avg_target, upsample_factor=upsample_factor)

            ensemble_avg_estimate, ensemble_beats_estimate = signal_processing_modules.get_ensemble_avg(r_peaks,
                                                                                                ( Y_batch_predicted[v,:]-np.mean(Y_batch_predicted[v,:]) )/(np.sqrt(np.sum(np.power(  Y_batch_predicted[v,:]-np.mean(Y_batch_predicted[v,:]) , 2  )))) ,
                                                                                                n_samples=500 , upsample_factor=upsample_factor)
            i_point_estimate, j_point_estimate, k_point_estimate = signal_processing_modules.get_IJK_peaks(ensemble_avg_estimate, upsample_factor=upsample_factor)

            i_error = 1000*np.abs(i_point_target-i_point_estimate)/(upsample_factor*500) if i_point_target!=-1 and i_point_estimate!=-1 else -1 #500 is the sampling rate of the signal segments, 1000* for miliseconds
            j_error = 1000*np.abs(j_point_target-j_point_estimate)/(upsample_factor*500) if j_point_target!=-1 and j_point_estimate!=-1 else -1
            k_error = 1000*np.abs(k_point_target-k_point_estimate)/(upsample_factor*500) if k_point_target!=-1 and k_point_estimate!=-1 else -1

            fig=plt.figure(figsize=(12,8))
            plt.subplot(2 , 1 , 1)

            plt.plot(ensemble_beats_target.T, 'r', alpha=0.3)
            plt.plot(ensemble_avg_target, '-r', linewidth=4)
            plt.plot(i_point_target, ensemble_avg_target[i_point_target], 'ok')
            plt.text(i_point_target, ensemble_avg_target[i_point_target] +0.01, 'I', color='red')
            plt.plot(j_point_target, ensemble_avg_target[j_point_target], 'ok')
            plt.text(j_point_target, ensemble_avg_target[j_point_target] +0.01, 'J', color='red')
            plt.plot(k_point_target, ensemble_avg_target[k_point_target], 'ok')
            plt.text(k_point_target, ensemble_avg_target[k_point_target] +0.01, 'K', color='red')

            plt.title('Subject Number: ' + str(subject_id_list[v]) + ' | Noise Variance =  '
                      + str( signal_processing_modules.get_noise_variance(ensemble_avg_target, ensemble_beats_target) ) )

            plt.subplot(2, 1, 2)

            plt.plot(ensemble_beats_estimate.T, 'b', alpha=0.3)
            plt.plot(ensemble_avg_estimate, '-b', linewidth=4)
            plt.plot(i_point_estimate, ensemble_avg_estimate[i_point_estimate], 'ok')
            plt.text(i_point_estimate, ensemble_avg_estimate[i_point_estimate] +0.01, 'I', color='blue')
            plt.plot(j_point_estimate, ensemble_avg_estimate[j_point_estimate], 'ok')
            plt.text(j_point_estimate, ensemble_avg_estimate[j_point_estimate] +0.01, 'J', color='blue')
            plt.plot(k_point_estimate, ensemble_avg_estimate[k_point_estimate], 'ok')
            plt.text(k_point_estimate, ensemble_avg_estimate[k_point_estimate] +0.01, 'K', color='blue')

            plt.title('I-error=' + str(i_error) +  ' | J-error=' + str(j_error) + ' | K-error=' + str(k_error) + ' in (ms)' + ' | Noise Variance =  '
                      + str( signal_processing_modules.get_noise_variance(ensemble_avg_estimate, ensemble_beats_estimate) ))

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

            list_noise_var_target.append(signal_processing_modules.get_noise_variance(ensemble_avg_target, ensemble_beats_target))
            list_noise_var_estimate.append(signal_processing_modules.get_noise_variance(ensemble_avg_estimate, ensemble_beats_estimate))


    print('MAE I-Point (ms): ' + str(np.mean(np.array(list_i_errors))))
    print('MAE J-Point (ms): ' + str(np.mean(np.array(list_j_errors))))
    print('MAE K-Point (ms): ' + str(np.mean(np.array(list_k_errors))))

    del X_batch
    del Y_batch_predicted
    del Y_batch

#get testing subjects
all_testing_subject_instances = utility.load_subjects(directory + '/Testing Data Healthy', store_in_ram=True, subject_type='testing' )

#check !
# print([u.subject_id for u in all_testing_subject_instances])
#utility.diagnose_training_subjects(all_testing_subject_instances, subject_type='testing')

#get training configs
eps =  config['eps']
directory= config['directory']
sig_type_source=config['sig_type_source']
sig_type_target=config['sig_type_target']
down_sample_factor_testing=config['down_sample_factor'] #same downsampling factor for training and testing
model_type=config['model_type']
normalized = True if model_type!='Unet_multiple_signal_in_not_normalized' else False
no_layers= config['no_layers']

#get testing configs
frame_length_testing = config_testing['frame_length_testing']
batch_size=config_testing['batch_size']
stride_testing = config_testing['stride_testing']
input_size = frame_length_testing//down_sample_factor_testing

test_gen=data_generators.make_generator_multiple_signal_testing(list_of_subjects=all_testing_subject_instances, batch_size=batch_size, eps=eps, frame_length=frame_length_testing
                                  ,stride=stride_testing, list_sig_type_source=sig_type_source, sig_type_target=sig_type_target
                                  , down_sample_factor=down_sample_factor_testing, normalized=normalized, store_in_ram=True)

#check generator !!
#data_generators.diagnose_generator_multiple_signal_testing(test_gen , sig_type_source)

#load model
model_path = directory + '/Code Output/'+ file_name_pre + '.pt'
model_type=config['model_type']
kernel_size=config['kernel_size']
filter_number=config['filter_number']
sig_model= network_models.load_saved_model(model_path , model_type , input_size , kernel_size  , filter_number=filter_number, signal_number=len(sig_type_source),
                                           no_layers=no_layers)


#visualize some predictions to check
visualize_signals_testing(test_gen ,sig_type_source)


#visualize ijk point detection
test_gen_ijk=data_generators.make_generator_multiple_signal_testing(list_of_subjects=all_testing_subject_instances, batch_size=batch_size, eps=eps, frame_length=frame_length_testing
                                  ,stride=stride_testing, list_sig_type_source=sig_type_source+['ecg'], sig_type_target=sig_type_target
                                  , down_sample_factor=down_sample_factor_testing, normalized=normalized, store_in_ram=True)

visualize_ijk_test(test_gen_ijk  , upsample_factor=1, no_cycles=1)

