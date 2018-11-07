import utility
import pickle
import torch
import network_models
import numpy as np
import data_generators
import signal_processing_modules
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels import robust
import os
import pickle

F_SAMPLING_TESTING=2000

#configs for testing
config_testing = {
             'frame_length_testing': int(4.096*F_SAMPLING_TESTING),
             'stride_testing': 16*int(4.096*F_SAMPLING_TESTING ) ,
             'batch_size':32
}

directory = '/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials'
models_directory = directory + '/Models to Test/'
all_models = os.listdir(models_directory)
model_names_pre = [u[:-3] for u in all_models]

list_model_names = []
list_models_pearson = []
list_models_i = []
list_models_j = []
list_models_k = []

for file_name_pre in model_names_pre:
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
            list_target_i_points = []
            list_target_j_points = []
            list_target_k_points = []
            list_sdr = []

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


                    if len(Y_batch.size())==1:
                        Y_batch=Y_batch.view(1,-1)

                    if len(Y_batch_predicted.size())==1:
                        Y_batch_predicted=Y_batch_predicted.view(1,-1)

                    loss = criterion.get_induvidual_losses(Y_batch_predicted, Y_batch)

                    list_pearson_r_loss+=loss.cpu().numpy().reshape(-1).tolist()
                    list_subject_ids+=subject_id_list

                    #ijk points
                    Y_batch_predicted = Y_batch_predicted.detach().cpu().numpy()
                    Y_batch = Y_batch.detach().cpu().numpy()

                    for v in range(Y_batch.shape[0]):
                        r_peaks = signal_processing_modules.get_R_peaks(ecg[v, :])
                        ensemble_avg_target, ensemble_beats_target = signal_processing_modules.get_ensemble_avg(r_peaks,
                                                                                                                (Y_batch[v, :] - np.mean(Y_batch[v,  :]) )/( np.sqrt(np.sum(np.power(  Y_batch[v, :] - np.mean(Y_batch[v, :])  ,2)))) ,
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
                        list_target_i_points.append(i_point_target)
                        list_target_j_points.append(j_point_target)
                        list_target_k_points.append(k_point_target)
                        list_sdr.append(signal_processing_modules.get_sdr(( Y_batch_predicted[v,:]-np.mean(Y_batch_predicted[v,:]) )/(np.sqrt(np.sum(np.power(  Y_batch_predicted[v,:]-np.mean(Y_batch_predicted[v,:]) , 2  ))))
                                    , (Y_batch[v, :] - np.mean(Y_batch[v,  :]) )/( np.sqrt(np.sum(np.power(  Y_batch[v, :] - np.mean(Y_batch[v, :])  ,2)))) ))



        del X_batch
        del Y_batch_predicted
        del Y_batch

        return list_pearson_r_loss, list_subject_ids, list_i_errors, list_j_errors, list_k_errors, list_noise_var_target, list_noise_var_estimate, list_target_i_points, list_target_j_points, list_target_k_points, list_sdr


    #run testing
    list_pearson_r_loss, list_subject_ids, list_i_errors, list_j_errors, list_k_errors , list_noise_var_target , list_noise_var_estimate, \
    list_target_i_points, list_target_j_points, list_target_k_points , list_sdr = test_model(test_gen, sig_model)

    #print results mean +/- std
    print('results mean +/- std')
    print('Mean Pearson Correlation Coefficient: ' + str(np.mean(np.array(list_pearson_r_loss))) + '+/-' + str(np.std(np.array(list_pearson_r_loss))))
    print('R-I MAEin (ms): ' + str(np.mean( np.array( [u for u in list_i_errors if u>0] ) ))  + '+/-' + str(np.std( np.array( [u for u in list_i_errors if u>0] ))))
    print('R-J MAEin (ms): ' + str(np.mean( np.array( [u for u in list_j_errors if u>0] ) ))+ '+/-' + str(np.std( np.array( [u for u in list_j_errors if u>0] ))))
    print('R-K MAEin (ms): ' + str(np.mean( np.array( [u for u in list_k_errors if u>0] ) )) + '+/-' + str(np.std( np.array( [u for u in list_k_errors if u>0] ))) )
    print('SDR (dB): ' + str(np.mean( np.array( list_sdr ) )) + '+/-' + str(np.std( np.array( list_sdr )))  )
    print('Average R-I interval (ms): ' + str (np.mean( np.array( [u for u in list_target_i_points if u>0] ) )) + ' +/- ' + str (np.std( np.array( [u for u in list_target_i_points if u>0] ) )) )
    print('Average R-J interval (ms): ' + str (np.mean( np.array( [u for u in list_target_j_points if u>0] ) )) + ' +/- ' + str (np.std( np.array( [u for u in list_target_j_points if u>0] ) )))
    print('Average R-K interval (ms): ' + str (np.mean( np.array( [u for u in list_target_k_points if u>0] ) )) + ' +/- ' + str (np.std( np.array( [u for u in list_target_k_points if u>0] ) )))
    print('Noise Variance for Targets: ' + str(np.mean( np.array( list_noise_var_target ) )) + '+/-' + str(np.std( np.array( list_noise_var_target ))))
    print('Noise Variance for Estimates: ' + str(np.mean( np.array( list_noise_var_estimate ) )) + '+/-' + str(np.std( np.array( list_noise_var_estimate ))))

    #print results median +/- mad
    print('results median +/- mad')
    print('Median Pearson Correlation Coefficient: ' + str(np.median(np.array(list_pearson_r_loss))) + '+/-' + str(robust.mad(np.array(list_pearson_r_loss))))
    print('R-I MAEin (ms): ' + str(np.median( np.array( [u for u in list_i_errors if u>0] ) ))  + '+/-' + str(robust.mad( np.array( [u for u in list_i_errors if u>0] ))))
    print('R-J MAEin (ms): ' + str(np.median( np.array( [u for u in list_j_errors if u>0] ) ))+ '+/-' + str(robust.mad( np.array( [u for u in list_j_errors if u>0] ))))
    print('R-K MAEin (ms): ' + str(np.median( np.array( [u for u in list_k_errors if u>0] ) )) + '+/-' + str(robust.mad( np.array( [u for u in list_k_errors if u>0] ))))
    print('SDR (dB): ' + str(np.median( np.array( list_sdr ) )) + '+/-' + str(robust.mad( np.array( list_sdr )))  )
    print('Average R-I interval (ms): ' + str (np.median( np.array( [u for u in list_target_i_points if u>0] ) )) + ' +/- ' + str (robust.mad( np.array( [u for u in list_target_i_points if u>0] ) )) )
    print('Average R-J interval (ms): ' + str (np.median( np.array( [u for u in list_target_j_points if u>0] ) )) + ' +/- ' + str (robust.mad( np.array( [u for u in list_target_j_points if u>0] ) )))
    print('Average R-K interval (ms): ' + str (np.median( np.array( [u for u in list_target_k_points if u>0] ) )) + ' +/- ' + str (robust.mad( np.array( [u for u in list_target_k_points if u>0] ) )))
    print('Noise Variance for Targets: ' + str(np.median( np.array( list_noise_var_target ) )) + '+/-' + str(robust.mad( np.array( list_noise_var_target ))))
    print('Noise Variance for Estimates: ' + str(np.median( np.array( list_noise_var_estimate ) )) + '+/-' + str(robust.mad( np.array( list_noise_var_estimate ))))

    list_model_names.append(file_name_pre)
    list_models_pearson.append(list_pearson_r_loss)
    list_models_i.append(list_i_errors)
    list_models_j.append(list_j_errors)
    list_models_k.append(list_k_errors)

    # print configs to a test file
    with open(directory + '/Code Output/' +  'model_test_results.txt', "a") as text_file:
        print(' ' , file=text_file)
        print(file_name_pre, file=text_file)
        print('Median Pearson Correlation Coefficient: ' + str(np.median(np.array(list_pearson_r_loss))) + '+/-' + str(
            robust.mad(np.array(list_pearson_r_loss))) , file=text_file)
        print('R-I MAEin (ms): ' + str(np.median(np.array([u for u in list_i_errors if u > 0]))) + '+/-' + str(
            robust.mad(np.array([u for u in list_i_errors if u > 0]))), file=text_file)
        print('R-J MAEin (ms): ' + str(np.median(np.array([u for u in list_j_errors if u > 0]))) + '+/-' + str(
            robust.mad(np.array([u for u in list_j_errors if u > 0]))), file=text_file)
        print('R-K MAEin (ms): ' + str(np.median(np.array([u for u in list_k_errors if u > 0]))) + '+/-' + str(
            robust.mad(np.array([u for u in list_k_errors if u > 0]))), file=text_file)
        print('SDR (dB): ' + str(np.median(np.array(list_sdr))) + '+/-' + str(robust.mad(np.array(list_sdr))), file=text_file)
        print('Average R-I interval (ms): ' + str(
            np.median(np.array([u for u in list_target_i_points if u > 0]))) + ' +/- ' + str(
            robust.mad(np.array([u for u in list_target_i_points if u > 0]))), file=text_file)
        print('Average R-J interval (ms): ' + str(
            np.median(np.array([u for u in list_target_j_points if u > 0]))) + ' +/- ' + str(
            robust.mad(np.array([u for u in list_target_j_points if u > 0]))), file=text_file)
        print('Average R-K interval (ms): ' + str(
            np.median(np.array([u for u in list_target_k_points if u > 0]))) + ' +/- ' + str(
            robust.mad(np.array([u for u in list_target_k_points if u > 0]))), file=text_file)
        print('Noise Variance for Targets: ' + str(np.median(np.array(list_noise_var_target))) + '+/-' + str(
            robust.mad(np.array(list_noise_var_target))), file=text_file)
        print('Noise Variance for Estimates: ' + str(np.median(np.array(list_noise_var_estimate))) + '+/-' + str(
            robust.mad(np.array(list_noise_var_estimate))), file=text_file)



#save workspace
pickle_list = [list_model_names, list_models_pearson , list_models_i, list_models_j, list_models_k]
fileObject = open(directory+'/Code Output/' +'model_test_results_pickle','wb')
pickle.dump(pickle_list,fileObject)
fileObject.close()

fig=plt.figure()
plt.scatter([u[-6::] for u in list_model_names] , [np.median(np.array(u)) for u in list_models_pearson ] )
plt.title('Median Pearson Correlation Coefficient')
fig.savefig(directory + '/Code Output/' + 'test_models_results_1.png')

fig=plt.figure()
list_y = []
for u in range(len(list_models_i)):
    list_y.append( [v for v in list_models_i[u] if v > 0] )
plt.scatter([u[-6::] for u in list_model_names], [np.median(np.array(u)) for u in list_y ]  )
plt.title('Median R-I Error')
fig.savefig(directory + '/Code Output/' + 'test_models_results_2.png')

fig=plt.figure()
list_y = []
for u in range(len(list_models_j)):
    list_y.append( [v for v in list_models_j[u] if v > 0] )
plt.scatter([u[-6::] for u in list_model_names], [np.median(np.array(u)) for u in list_y ]  )
plt.title('Median R-J Error')
fig.savefig(directory + '/Code Output/' + 'test_models_results_3.png')

fig=plt.figure()
list_y = []
for u in range(len(list_models_k)):
    list_y.append( [v for v in list_models_k[u] if v > 0] )
plt.scatter([u[-6::] for u in list_model_names], [np.median(np.array(u)) for u in list_y ]  )
plt.title('Median R-K Error')
fig.savefig(directory + '/Code Output/' + 'test_models_results_4.png')


