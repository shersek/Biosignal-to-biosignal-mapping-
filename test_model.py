import utility
import pickle
import torch
import network_models
import numpy as np
import data_generators
import signal_processing_modules
import matplotlib.pyplot as plt

F_SAMPLING_TESTING=2000

#configs for testing
config_testing = {
             'frame_length_testing': int(4.096*F_SAMPLING_TESTING),
             'stride_testing': 4*int(4.096*F_SAMPLING_TESTING ) ,
             'batch_size':32,
}

file_name_pre = 'mdl_Unetx6_axyz_29_Oct_01'
directory = '/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials'
fileObject = open(directory + '/Code Output/'+file_name_pre+'_pickle','rb')
(config, _ , _, _, _ , _) = pickle.load(fileObject)
save_true = False


#get testing HF subjects
all_testing_subject_instances = utility.load_subjects(directory + '/Testing Data Healthy', store_in_ram=True, subject_type='testing' )

#check !
# print([u.subject_id for u in all_testing_subject_instances])
#utility.diagnose_training_subjects(all_testing_subject_instances, subject_type='testing')

#get training configs
eps =  config['eps']
directory= config['directory']
sig_type_source=config['sig_type_source']
sig_type_target=config['sig_type_target']
model_type=config['model_type']
normalized = True if model_type!='Unet_multiple_signal_in_not_normalized' else False
no_layers= config['no_layers']

#get testing configs
frame_length_testing = config_testing['frame_length_testing']
batch_size=config_testing['batch_size']
stride_testing = config_testing['stride_testing']
down_sample_factor_testing=config['down_sample_factor'] #same downsampling factor for training and testing
input_size = frame_length_testing//down_sample_factor_testing

#create the testing generator
test_gen=data_generators.make_generator_multiple_signal_testing(list_of_subjects=all_testing_subject_instances, batch_size=batch_size, eps=eps, frame_length=frame_length_testing
                                  ,stride=stride_testing, list_sig_type_source=sig_type_source+['ecg'], sig_type_target=sig_type_target
                                  , down_sample_factor=down_sample_factor_testing, normalized=normalized, store_in_ram=True)

#check generator !!
#data_generators.diagnose_generator_multiple_signal_testing(test_gen , sig_type_source+['ecg'])

#load model
model_path = directory + '/Code Output/'+ file_name_pre + '.pt'
model_type=config['model_type']
kernel_size=config['kernel_size']
filter_number=config['filter_number']
sig_model= network_models.load_saved_model(model_path , model_type , input_size , kernel_size  , filter_number=filter_number, signal_number=len(sig_type_source),
                                           no_layers=no_layers)

def test_model(gen, sig_model):
    '''
    function tests a model by running it over the test data and calculating error metrics
    :param gen: test generator
    :param sig_model: model
    :return: list_pearson_r_loss: list of pearson correlations between target and estimated segments
    :return: list_subject_ids: subject ID for each segment
    :return: list_i_errors: list of R-I interval errors between each target and estimated signal segment
    :return: list_j_errors: list of R-J interval errors between each target and estimated signal segment
    :return: list_k_errors: list of R-K interval errors between each target and estimated signal segment
    '''
    criterion = network_models.PearsonRLoss()
    with torch.no_grad():

        finished = False
        list_pearson_r_loss = []
        list_i_errors = []
        list_j_errors = []
        list_k_errors = []
        list_subject_ids = []
        list_noise_var_target = []
        list_noise_var_estimate = []
        while not finished:

            print('Running Testing')
            torch.cuda.empty_cache()
            finished, X_batch, Y_batch, subject_id_list= next(gen)

            if finished:
                print('Generator Finished')
            else:

                #get ecg
                ecg = X_batch[:, -1, :]

                #calculate pearson correlation
                X_batch = torch.from_numpy(X_batch[:, :-1, :]).contiguous()
                X_batch = network_models.cuda(X_batch)
                X_batch = X_batch.type(torch.cuda.FloatTensor)

                Y_batch_predicted= sig_model.forward(X_batch).squeeze()

                Y_batch = torch.from_numpy(Y_batch)
                Y_batch = network_models.cuda(Y_batch)
                Y_batch = Y_batch.type(torch.cuda.FloatTensor)
                Y_batch = Y_batch.squeeze()

                loss = criterion.get_induvidual_losses(Y_batch_predicted, Y_batch)

                list_pearson_r_loss+=loss.cpu().numpy().reshape(-1).tolist()
                list_subject_ids+=subject_id_list

                #ijk points
                Y_batch_predicted = Y_batch_predicted.detach().cpu().numpy()
                Y_batch = Y_batch.detach().cpu().numpy()

                for v in range(Y_batch.shape[0]):
                    r_peaks = signal_processing_modules.get_R_peaks(ecg[v, :])
                    ensemble_avg_target, ensemble_beats_target = signal_processing_modules.get_ensemble_avg(r_peaks,
                                                                                                            (Y_batch[v, 0, :] - np.mean(Y_batch[v, 0, :]) )/( np.sqrt(np.sum(np.power(  Y_batch[v, 0, :] - np.mean(Y_batch[v, 0, :])  ,2)))) ,
                                                                                                            n_samples=500,upsample_factor=1)

                    i_point_target, j_point_target, k_point_target = signal_processing_modules.get_IJK_peaks(ensemble_avg_target, upsample_factor=1)

                    ensemble_avg_estimate, ensemble_beats_estimate = signal_processing_modules.get_ensemble_avg(r_peaks,
                                                                                                                ( Y_batch_predicted[v,:]-np.mean(Y_batch_predicted[v,:]) )/(np.sqrt(np.sum(np.power(  Y_batch_predicted[v,:]-np.mean(Y_batch_predicted[v,:]) , 2  )))) ,
                                                                                                                n_samples=500, upsample_factor=1)
                    i_point_estimate, j_point_estimate, k_point_estimate = signal_processing_modules.get_IJK_peaks(ensemble_avg_estimate, upsample_factor=1)

                    i_error = 1000 * np.abs(i_point_target - i_point_estimate) / (500) if i_point_target != -1 and i_point_estimate != -1 else -1  # 500 is the sampling rate of the signal segments, 1000* for miliseconds
                    j_error = 1000 * np.abs(j_point_target - j_point_estimate) / (500) if j_point_target != -1 and j_point_estimate != -1 else -1
                    k_error = 1000 * np.abs(k_point_target - k_point_estimate) / (500) if k_point_target != -1 and k_point_estimate != -1 else -1

                    list_i_errors.append(i_error)
                    list_j_errors.append(j_error)
                    list_k_errors.append(k_error)
                    list_noise_var_target.append( signal_processing_modules.get_noise_variance(ensemble_avg_target, ensemble_beats_target) )
                    list_noise_var_estimate.append( signal_processing_modules.get_noise_variance(ensemble_avg_estimate, ensemble_beats_estimate ) )

                # fig = plt.figure(figsize=(12, 8))
                # plt.subplot(2, 1, 1)
                #
                # plt.plot(ensemble_beats_target.T, 'r', alpha=0.3)
                # plt.plot(ensemble_avg_target, '-r', linewidth=4)
                # plt.plot(i_point_target, ensemble_avg_target[i_point_target], 'ok')
                # plt.text(i_point_target, ensemble_avg_target[i_point_target] + 0.05, 'I', color='red')
                # plt.plot(j_point_target, ensemble_avg_target[j_point_target], 'ok')
                # plt.text(j_point_target, ensemble_avg_target[j_point_target] + 0.05, 'J', color='red')
                # plt.plot(k_point_target, ensemble_avg_target[k_point_target], 'ok')
                # plt.text(k_point_target, ensemble_avg_target[k_point_target] + 0.05, 'K', color='red')
                #
                # plt.title('Subject Number: ' + str(subject_id_list[v]))
                #
                # plt.subplot(2, 1, 2)
                #
                # plt.plot(ensemble_beats_estimate.T, 'b', alpha=0.3)
                # plt.plot(ensemble_avg_estimate, '-b', linewidth=4)
                # plt.plot(i_point_estimate, ensemble_avg_estimate[i_point_estimate], 'ok')
                # plt.text(i_point_estimate, ensemble_avg_estimate[i_point_estimate] + 0.05, 'I', color='blue')
                # plt.plot(j_point_estimate, ensemble_avg_estimate[j_point_estimate], 'ok')
                # plt.text(j_point_estimate, ensemble_avg_estimate[j_point_estimate] + 0.05, 'J', color='blue')
                # plt.plot(k_point_estimate, ensemble_avg_estimate[k_point_estimate], 'ok')
                # plt.text(k_point_estimate, ensemble_avg_estimate[k_point_estimate] + 0.05, 'K', color='blue')
                #
                # plt.title('I-error=' + str(i_error) + ' | J-error=' + str(j_error) + ' | K-error=' + str(
                #     k_error) + ' in (ms)')
                #
                # plt.tight_layout()
                # plt.show()

    del X_batch
    del Y_batch_predicted
    del Y_batch

    return list_pearson_r_loss, list_subject_ids, list_i_errors, list_j_errors, list_k_errors, list_noise_var_target, list_noise_var_estimate


#run testing
list_pearson_r_loss, list_subject_ids, list_i_errors, list_j_errors, list_k_errors , list_noise_var_target , list_noise_var_estimate= test_model(test_gen, sig_model)

#print results
print('Mean Pearson Correlation Coefficient: ' + str(np.mean(np.array(list_pearson_r_loss))) + '+/-' + str(np.std(np.array(list_pearson_r_loss))))
print('R-I MAEin (ms): ' + str(np.mean( np.array( [u for u in list_i_errors if u>0] ) ))  + '+/-' + str(np.std( np.array( [u for u in list_i_errors if u>0] ))))
print('R-J MAEin (ms): ' + str(np.mean( np.array( [u for u in list_j_errors if u>0] ) ))+ '+/-' + str(np.std( np.array( [u for u in list_j_errors if u>0] ))))
print('R-K MAEin (ms): ' + str(np.mean( np.array( [u for u in list_k_errors if u>0] ) )) + '+/-' + str(np.std( np.array( [u for u in list_k_errors if u>0] ))))

print('Noise Variance for Targets: ' + str(np.mean( np.array( list_noise_var_target ) )) + '+/-' + str(np.std( np.array( list_noise_var_target ))))
print('Noise Variance for Estimates: ' + str(np.mean( np.array( list_noise_var_estimate ) )) + '+/-' + str(np.std( np.array( list_noise_var_estimate ))))


plt.figure()
plt.scatter(list_noise_var_target, list_i_errors)
plt.title('R-I Error vs Noise Variance Target')

plt.figure()
plt.scatter(list_noise_var_target, list_j_errors)
plt.title('R-J Error vs Noise Variance Target')

plt.figure()
plt.scatter(list_noise_var_target, list_k_errors)
plt.title('R-K Error vs Noise Variance Target')


plt.figure()
plt.scatter(list_noise_var_estimate, list_i_errors)
plt.title('R-I Error vs Noise Variance Target')

plt.figure()
plt.scatter(list_noise_var_estimate, list_j_errors)
plt.title('R-J Error vs Noise Variance Target')

plt.figure()
plt.scatter(list_noise_var_estimate, list_k_errors)
plt.title('R-K Error vs Noise Variance Target')

plt.figure()
plt.scatter(list_noise_var_target , list_noise_var_estimate)
plt.title('Noise Variance of Target vs Estimate')
