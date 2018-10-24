import numpy as np
import matplotlib.pyplot as plt
import math

F_SAMPLING = 2000
F_SAMPLING_HF=1000


def get_raw_sig(sig, offset, frame_length, down_sample_factor ,eps):
    raw_sig = sig[offset:(frame_length + offset)]
    raw_sig = raw_sig[::down_sample_factor]
    raw_sig = (raw_sig - np.min(raw_sig)) / (np.max(raw_sig) - np.min(raw_sig) + eps)
    return raw_sig

def get_raw_sig_not_normalized(sig, offset, frame_length, down_sample_factor ,eps):
    raw_sig = sig[offset:(frame_length + offset)]
    raw_sig = raw_sig[::down_sample_factor]
    return raw_sig

def get_sine(frame_length , down_sample_factor , freq , eps):
    tt = np.arange(0,frame_length,1)/F_SAMPLING
    x= np.sin(2*math.pi*freq*tt )
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + eps)
    x = x[::down_sample_factor]
    tt = tt[::down_sample_factor]

    return tt,x


def get_rot_matrix(rot_angle , axis):

    if axis=='x':
        R = np.array([[1,0,0] , [0, math.cos(rot_angle) , math.sin(rot_angle) ] , [0 , -math.sin(rot_angle) , math.cos(rot_angle) ] ])
    elif axis=='y':
        R = np.array([[math.cos(rot_angle) , 0,  -math.sin(rot_angle) ] , [0,1,0] , [math.sin(rot_angle) , 0 , math.cos(rot_angle)] ])
    elif axis=='z':
        R = np.array([[math.cos(rot_angle) , math.sin(rot_angle) , 0 ] , [-math.sin(rot_angle) , math.cos(rot_angle), 0] , [0,0,1] ])

    return R

def augment_accel_signals(subject, rot_angle_x, rot_angle_y , rot_angle_z , store_in_ram=False):

    R = np.matmul(np.matmul (get_rot_matrix(rot_angle_x , 'x')   , get_rot_matrix(rot_angle_y , 'y')),  get_rot_matrix(rot_angle_z , 'z'))

    if store_in_ram:
        accel_vector = np.array((subject.aX  , subject.aY , subject.aZ))

    else:
        accel_vector = np.array((subject.get_aX()  ,subject.get_aY() , subject.get_aZ()))

    accel_vector_rotated = np.matmul(R, accel_vector)
    aX_tilde = accel_vector_rotated[0,:].reshape(-1)
    aY_tilde = accel_vector_rotated[1,:].reshape(-1)
    aZ_tilde = accel_vector_rotated[2,:].reshape(-1)


    return aX_tilde, aY_tilde , aZ_tilde



def make_generator_multiple_signal(list_of_subjects, cycle_per_batch, eps, frame_length=4*F_SAMPLING, mode='rest', list_sig_type_source= ['icg'], sig_type_target= 'bp'
                   ,down_sample_factor=1 , normalized = True, store_in_ram=False ,
                    augment_accel = False, augment_theta_lim = 10 , augment_prob=0.2):

    while True:

        no_of_subjects = len(list_of_subjects)
        batch_size = cycle_per_batch * no_of_subjects

        X_batch = np.zeros(shape=(batch_size, len(list_sig_type_source), frame_length//down_sample_factor))
        Y_batch = np.zeros(shape=(batch_size, 1, frame_length//down_sample_factor))
        count = 0
        subject_id_list = []

        for c in range(cycle_per_batch):
            for subject in list_of_subjects:

                if mode=='rest':
                    # get random segment
                    max_offset = subject.rest_interval[1] - frame_length
                    min_offset = subject.rest_interval[0]
                elif mode=='recovery':
                    # get random segment
                    max_offset = subject.recovery_interval[1] - frame_length
                    min_offset = subject.recovery_interval[0]
                elif mode=='both':
                    rand_selection = np.random.uniform(0,1)
                    if rand_selection<0.5:
                        max_offset = subject.rest_interval[1] - frame_length
                        min_offset = subject.rest_interval[0]
                    else:
                        max_offset = subject.recovery_interval[1] - frame_length
                        min_offset = subject.recovery_interval[0]

                offset = np.random.randint(min_offset, max_offset) #THIS IS SUPPOSED TO BE HERE ????? !!!!! HUGE BUG FIX


                if augment_accel:
                    aug_selection = np.random.uniform(0, 1)
                    if aug_selection < augment_prob:  # if augment
                        augment_angle_x = np.random.uniform(-augment_theta_lim, augment_theta_lim) * math.pi / 180
                        augment_angle_y = np.random.uniform(-augment_theta_lim, augment_theta_lim) * math.pi / 180
                        augment_angle_z = np.random.uniform(-augment_theta_lim, augment_theta_lim) * math.pi / 180
                    else:
                        augment_angle_x = 0
                        augment_angle_y = 0
                        augment_angle_z = 0

                for i,sig_type_source in enumerate(list_sig_type_source):

                    #get signal
                    if sig_type_source=='aZ':


                        if augment_accel:
                             _,_,sig_source = augment_accel_signals(subject, augment_angle_x, augment_angle_y ,augment_angle_z , store_in_ram=store_in_ram)

                        else:
                            if store_in_ram:
                                sig_source = subject.aZ
                            else:
                                sig_source = subject.get_aZ()

                        # N = 1500
                        # sig_filt = np.convolve(sig_source, np.ones((N,)) / N, mode='same')
                        # sig_source = sig_source - sig_filt


                    elif sig_type_source=='aY':

                        if augment_accel:
                             _, sig_source , _ = augment_accel_signals(subject, augment_angle_x, augment_angle_y ,augment_angle_z , store_in_ram=store_in_ram)

                        else:
                            if store_in_ram:
                                sig_source = subject.aY
                            else:
                                sig_source = subject.get_aY()


                    elif sig_type_source == 'aX':

                        if augment_accel:
                            sig_source,_,_ = augment_accel_signals(subject, augment_angle_x, augment_angle_y ,augment_angle_z , store_in_ram=store_in_ram)

                        else:
                            if store_in_ram:
                                sig_source = subject.aX
                            else:
                                sig_source = subject.get_aX()


                    elif sig_type_source == 'ecg':
                        if store_in_ram:
                            sig_source = subject.ecg
                        else:
                            sig_source = subject.get_ecg()
                    elif sig_type_source == 'bcg':
                        if store_in_ram:
                            sig_source = subject.bcg
                        else:
                            sig_source = subject.get_bcg()
                        N = 1500
                        sig_filt = np.convolve(sig_source, np.ones((N,)) / N, mode='same')
                        sig_source = sig_source - sig_filt
                    elif sig_type_source == 'icg':
                        if store_in_ram:
                            sig_source = subject.icg
                        else:
                            sig_source = subject.get_icg()
                    elif sig_type_source == 'bp':
                        if store_in_ram:
                            sig_source = subject.bp
                        else:
                            sig_source = subject.get_bp()
                    elif sig_type_source=='gZ':
                        if store_in_ram:
                            sig_source = subject.gZ
                        else:
                            sig_source = subject.get_gZ()
                    elif sig_type_source=='gY':
                        if store_in_ram:
                            sig_source = subject.gY
                        else:
                            sig_source = subject.get_gY()
                    elif sig_type_source == 'gX':
                        if store_in_ram:
                            sig_source = subject.gX
                        else:
                            sig_source = subject.get_gX()
                        N = 1500
                        sig_filt = np.convolve(sig_source, np.ones((N,)) / N, mode='same')
                        sig_source = sig_source - sig_filt


                    if normalized:
                        X_batch[count, i , :] = get_raw_sig(sig_source , offset , frame_length , down_sample_factor , eps)
                    else:
                        X_batch[count, i , :] = get_raw_sig_not_normalized(sig_source , offset , frame_length , down_sample_factor , eps)




                # get signal
                if sig_type_target == 'aZ':
                    if store_in_ram:
                        sig_target = subject.aZ
                    else:
                        sig_target = subject.get_aZ()
                    N = 1500
                    sig_filt = np.convolve(sig_target, np.ones((N,)) / N, mode='same')
                    sig_target = sig_target - sig_filt
                elif sig_type_target == 'aY':
                    if store_in_ram:
                        sig_target = subject.aY
                    else:
                        sig_target = subject.get_aY()
                elif sig_type_target == 'aX':
                    if store_in_ram:
                        sig_target = subject.aX
                    else:
                        sig_target = subject.get_aX()
                elif sig_type_target == 'ecg':
                    if store_in_ram:
                        sig_target = subject.ecg
                    else:
                        sig_target = subject.get_ecg()
                elif sig_type_target == 'bcg':
                    if store_in_ram:
                        sig_target = subject.bcg
                    else:
                        sig_target = subject.get_bcg()
                    N = 1500
                    sig_filt = np.convolve(sig_target, np.ones((N,)) / N, mode='same')
                    sig_target = sig_target - sig_filt
                elif sig_type_target == 'icg':
                    if store_in_ram:
                        sig_target = subject.icg
                    else:
                        sig_target = subject.get_icg()
                elif sig_type_target == 'bp':
                    if store_in_ram:
                        sig_target = subject.bp
                    else:
                        sig_target = subject.get_bp()
                elif sig_type_target == 'gZ':
                    if store_in_ram:
                        sig_target = subject.gZ
                    else:
                        sig_target = subject.get_gZ()
                elif sig_type_target == 'gY':
                    if store_in_ram:
                        sig_target = subject.gY
                    else:
                        sig_target = subject.get_gY()
                elif sig_type_target == 'gX':
                    if store_in_ram:
                        sig_target = subject.gX
                    else:
                        sig_target = subject.get_gX()
                    N = 1500
                    sig_filt = np.convolve(sig_target, np.ones((N,)) / N, mode='same')
                    sig_target = sig_target - sig_filt


                if normalized:
                    Y_batch[count, 0, :] = get_raw_sig(sig_target, offset, frame_length, down_sample_factor, eps)
                else:
                    Y_batch[count, 0, :] = get_raw_sig_not_normalized(sig_target, offset, frame_length, down_sample_factor, eps)

                count += 1
                subject_id_list.append(subject.subject_id)
                # print(count)

        yield X_batch, Y_batch , subject_id_list






def make_generator_multiple_signal_sine_wave(batch_size, freq_lo, freq_hi,  eps,frame_length=4*F_SAMPLING,list_sig_type_source= ['icg'],down_sample_factor=1 ):

    while True:

        X_batch = np.zeros(shape=(batch_size, len(list_sig_type_source), frame_length//down_sample_factor))
        count = 0
        freq_vector = np.arange(freq_lo,freq_hi , step=(freq_hi-freq_lo)/batch_size )

        for c in range(batch_size):

            for i,sig_type_source in enumerate(list_sig_type_source):

                tt, X_batch[count, i , :] = get_sine(frame_length, down_sample_factor, freq_vector[c], eps)

            count += 1

        yield X_batch ,tt , freq_vector


def diagnose_generator_multiple_signal(gen , list_sig_type_source):
    print('Diagnosing Generator..')
    X_batch, Y_batch, subject_id_list = next(gen)
    no_sigs = X_batch.shape[1]
    batch_size = X_batch.shape[0]
    print(X_batch.shape)
    for u in range(0,batch_size):

        plt.figure()

        plt.subplot(no_sigs+1, 1, 1)
        plt.plot(Y_batch[u, 0, :])
        plt.title(str(subject_id_list[u]) + ' ' + str(u))

        for v in range(2,no_sigs+2):
            plt.subplot(no_sigs+1, 1, v)
            plt.plot(X_batch[u, v-2, :])
            plt.title(list_sig_type_source[v-2])

        plt.tight_layout()
        plt.show()






def make_generator_multiple_signal_hf(list_of_recordings, batch_size , eps, frame_length=4*F_SAMPLING_HF, stride= F_SAMPLING_HF//10, list_sig_type_source= ['aX'], sig_type_target= 'bcg'
                   ,down_sample_factor=2 , normalized = True, store_in_ram=False):


    X_batch = np.zeros(shape=(batch_size, len(list_sig_type_source), frame_length // down_sample_factor))
    Y_batch = np.zeros(shape=(batch_size, 1, frame_length // down_sample_factor))
    count = 0
    subject_id_list = []
    recording_id_list = []

    for recording in list_of_recordings:


        sig_length = recording.ecg.shape[0]

        for offset in range(0,sig_length - frame_length , stride):

            for i,sig_type_source in enumerate(list_sig_type_source):
                #get signal
                if sig_type_source=='aZ':
                    if store_in_ram:
                        sig_source = recording.aZ
                    else:
                        sig_source = recording.get_aZ()
                    # N=750
                    # sig_filt = np.convolve(sig_source, np.ones((N,)) / N, mode='same')
                    # sig_source = sig_source - sig_filt

                elif sig_type_source=='aY':
                    if store_in_ram:
                        sig_source = recording.aY
                    else:
                        sig_source = recording.get_aY()

                elif sig_type_source == 'aX':
                    if store_in_ram:
                        sig_source = recording.aX
                    else:
                        sig_source = recording.get_aX()
                elif sig_type_source == 'ecg':
                    if store_in_ram:
                        sig_source = recording.ecg
                    else:
                        sig_source = recording.get_ecg()
                elif sig_type_source == 'bcg':
                    if store_in_ram:
                        sig_source = recording.bcg
                    else:
                        sig_source = recording.get_bcg()
                    N=750
                    sig_filt = np.convolve(sig_source, np.ones((N,)) / N, mode='same')
                    sig_source = sig_source - sig_filt


                if normalized:
                    X_batch[count, i , :] = get_raw_sig(sig_source , offset , frame_length , down_sample_factor , eps)
                else:
                    X_batch[count, i , :] = get_raw_sig_not_normalized(sig_source , offset , frame_length , down_sample_factor , eps)




            # get signal
            if sig_type_target == 'aZ':
                if store_in_ram:
                    sig_target = recording.aZ
                else:
                    sig_target = recording.get_aZ()
                N=750
                sig_filt = np.convolve(sig_target, np.ones((N,)) / N, mode='same')
                sig_target = sig_target - sig_filt
            elif sig_type_target == 'aY':
                if store_in_ram:
                    sig_target = recording.aY
                else:
                    sig_target = recording.get_aY()
            elif sig_type_target == 'aX':
                if store_in_ram:
                    sig_target = recording.aX
                else:
                    sig_target = recording.get_aX()
            elif sig_type_target == 'ecg':
                if store_in_ram:
                    sig_target = recording.ecg
                else:
                    sig_target = recording.get_ecg()
            elif sig_type_target == 'bcg':
                if store_in_ram:
                    sig_target = recording.bcg
                else:
                    sig_target = recording.get_bcg()
                N=750
                sig_filt = np.convolve(sig_target, np.ones((N,)) / N, mode='same')
                sig_target = sig_target - sig_filt


            if normalized:
                Y_batch[count, 0, :] = get_raw_sig(sig_target, offset, frame_length, down_sample_factor, eps)
            else:
                Y_batch[count, 0, :] = get_raw_sig_not_normalized(sig_target, offset, frame_length, down_sample_factor, eps)

            count += 1
            subject_id_list.append(recording.subject_id)
            recording_id_list.append(recording.recording_id)

            if count==batch_size:

                yield False , X_batch, Y_batch , subject_id_list , recording_id_list

                X_batch = np.zeros(shape=(batch_size, len(list_sig_type_source), frame_length // down_sample_factor))
                Y_batch = np.zeros(shape=(batch_size, 1, frame_length // down_sample_factor))
                count = 0
                subject_id_list = []
                recording_id_list = []

    if count<batch_size and count>0:
        X_batch= X_batch[0:count ,:,:]
        Y_batch= Y_batch[0:count ,:,:]
        subject_id_list=subject_id_list[0:count]
        recording_id_list=recording_id_list[0:count]
        yield False, X_batch, Y_batch, subject_id_list, recording_id_list

    while True:
        yield True, None, None, None , None


def diagnose_generator_multiple_signal_hf(gen , list_sig_type_source):
    print('Diagnosing Generator..')
    finished, X_batch, Y_batch, subject_id_list, recording_id_list = next(gen) #left here
    no_sigs = X_batch.shape[1]
    batch_size = X_batch.shape[0]
    print(X_batch.shape)
    for u in range(0,batch_size,2):

        plt.figure(figsize=(12,8))

        plt.subplot(no_sigs+1, 1, 1)
        plt.plot(Y_batch[u, 0, :])
        plt.title('Subject: ' + str(subject_id_list[u]) + ' Recording: ' + str(recording_id_list[u]) + ' Index in batch: ' + str(u))

        for v in range(2,no_sigs+2):
            plt.subplot(no_sigs+1, 1, v)
            plt.plot(X_batch[u, v-2, :])
            plt.title(list_sig_type_source[v-2])

        #plt.tight_layout()
        plt.show()
