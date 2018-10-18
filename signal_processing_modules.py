import utility
import data_generators
import matplotlib.pyplot as plt
import numpy as np


F_SAMPLING=2000


def get_R_peaks(sig_ecg):
    '''
    function to get ECG R-peaks
    :param sig_ecg: ECG signal
    :return: sample indices of the ECG R-peaks
    '''

    ##write code below


    return np.array([500, 1000 , 1500, 1800])

def get_ensemble_avg(r_peaks, sig, n_samples):
    '''
    take the ensemble average of a given signal
    :param r_peaks: ECG R-peaks
    :param sig: signal to ensemble average
    :param n_samples: number of samples per beat
    :return: ensemble average waveform and a matrix contatining all beats of the ensemble
    '''

    ensemble_beats = np.zeros((r_peaks.shape[0] , n_samples))
    for u in range(r_peaks.shape[0] ):
        beat = sig[ r_peaks[u] : min(r_peaks[u] + n_samples , sig.shape[0]) ]
        beat = beat-np.mean(beat)
        ensemble_beats[ u, 0:beat.shape[0] ] = beat


    ensemble_avg = np.mean(ensemble_beats , 0)

    return ensemble_avg , ensemble_beats

def get_IJK_peaks(sig_ensemble_bcg):
    '''
    find the BCG I, J and K points from the ensemble averaged waveform
    :param sig_ensemble_bcg: ensemble averaged BCG waveform
    :return: i,j,k points from the ensmeble averaged waveform
    '''

    ##write code below
    i_point = 5
    j_point = 100
    k_point = 300

    return  i_point, j_point, k_point

def diagnose_peak_finding(signal_generator , n_samples):
    '''
    generates a series of plots to visualize if the peak finding is working well
    :param signal_generator:
    :return: -
    '''
    print('Diagnosing')
    X_batch, Y_batch, subject_id_list = next(signal_generator)
    no_sigs = X_batch.shape[1]
    batch_size = X_batch.shape[0]
    print(X_batch.shape)
    for u in range(0, batch_size):
        bcg = Y_batch[u, 0, :]
        ecg = X_batch[u, 0, :]
        r_peaks = get_R_peaks(ecg)
        ensemble_avg, ensemble_beats = get_ensemble_avg(r_peaks, bcg, n_samples)
        i_point, j_point, k_point = get_IJK_peaks(ensemble_avg)

        plt.figure()

        plt.subplot(3, 1, 1)
        plt.plot(bcg, '-b')
        plt.title('BCG ' + str(subject_id_list[u]) + ' ' + str(u))

        plt.subplot(3, 1, 2)
        plt.plot(ecg, '-r')
        plt.plot(r_peaks, ecg[r_peaks], 'ok')
        plt.title('ECG')
        plt.plot()

        plt.subplot(3, 1, 3)
        plt.plot(ensemble_beats.T, 'b', alpha=0.5)
        plt.plot(ensemble_avg, '-g', linewidth=3)
        plt.plot(i_point, ensemble_avg[i_point], 'ok')
        plt.text(i_point, ensemble_avg[i_point] + 0.1, 'I', color='red')
        plt.plot(j_point, ensemble_avg[j_point], 'ok')
        plt.text(j_point, ensemble_avg[j_point] + 0.1, 'J', color='red')
        plt.plot(k_point, ensemble_avg[k_point], 'ok')
        plt.text(k_point, ensemble_avg[k_point] + 0.1, 'K', color='red')

        plt.title('Ensemble Average')
        plt.plot()

        plt.tight_layout()
        plt.show()


#get all subject data
directory_for_subjects ='/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Training Data Analog Acc'
all_subject_instances = utility.load_subjects(directory_for_subjects, store_in_ram=False )

#make a train and val generator
signal_generator = data_generators.make_generator_multiple_signal(list_of_subjects=all_subject_instances, cycle_per_batch=1, eps=1e-3 ,frame_length=int(4.096*F_SAMPLING),
                                    mode='both', list_sig_type_source= [ 'ecg' ], sig_type_target= 'bcg' , down_sample_factor =4,
                                                           normalized=True , store_in_ram=False ,
                                                           augment_accel = False , augment_theta_lim = 10  , augment_prob=0.5)

#run diagnostics
diagnose_peak_finding(signal_generator , n_samples=500)


##TODO:
# this script is written to detect ECG R-peaks and BCG I-J-K peaks
# change the 'directory_for_subjects' variable to the directory on your computer where all the data is
# in the 'get_R_peaks(sig_ecg)' function, you will see a portion where it says '##write code below'
# design an algorithm to detect the ECG R-peaks inside this function
# the return of this function is 'np.array([500, 1000 , 1500, 1800])', which is a dummy return value so that should be changed
# design an algorithm to detect the I,J and K peaks of the ensemble averaged BCG inside the function 'get_IJK_peaks(sig_ensemble_bcg)'
# you can do this where it says 'write code below'
# you should erase the dummy return values i_point = 5, j_point = 100 ,k_point = 300
# also, play around with the n_samples variable in 'diagnose_peak_finding(signal_generator , n_samples=500)', this controls the length
# of the ensemble averaged waveform in samples. I set it to 500 randomly as a start, but you guys can figure out what the value of this should be
#More info:
# All signal segments are 4.096 seconds long. All ecg and bcg signal segments are normalized to [0,1]
# You shouldn't need to change anything other than the functions 'get_R_peaks', 'get_IJK_peaks' and the variable 'n_samples' in 'diagnose_peak_finding(signal_generator , n_samples=500)'
# and ofcourse the directory_for_subjects
