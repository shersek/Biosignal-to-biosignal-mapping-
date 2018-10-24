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


F_SAMPLING=2000

config = {
             'file_name_pre': 'mdl_00000_a000_23_Oct_02',  #should be 25 letters
             'mode':'both',
             'eps': 1e-3,
             'model_type': 'Unetxl',
             'no_layers': 5 ,
             'down_sample_factor':4,
            'frame_length' : int(4.096*F_SAMPLING),
             'kernel_size': 7 , #(3,5) , #5,#(3,5), #5, #(3, 5),#5
             'directory':'/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials',
             'cycle_per_batch':2,
             'filter_number': 64,
            'sig_type_source': [ 'aX' , 'aY' , 'aZ'],
            'sig_type_target': 'bcg',
             'loss_func':'pearson_r',
             'produce_video' : False,
             'store_in_ram':True,
             'augment_accel':True,
             'augment_theta_lim': 10,
             'augment_prob':0.5

}

#
cycle_per_batch = config['cycle_per_batch']
mode= config['mode']
eps =  config['eps']
kernel_size = config['kernel_size']
directory= config['directory']
model_type = config['model_type']
no_layers=config['no_layers']
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
augment_accel= config['augment_accel']
augment_theta_lim = config['augment_theta_lim']
augment_prob=config['augment_prob']
file_name_pre = config['file_name_pre']
file_name_pre = file_name_pre[0:4] + model_type[:-1] + str(no_layers) +file_name_pre[9::]
axis_string = ''.join(['x' if 'aX' in sig_type_source else '0' , 'y' if 'aY' in sig_type_source else '0' , 'z' if 'aZ' in sig_type_source else '0' ])
file_name_pre = file_name_pre[:12] + axis_string + file_name_pre[15::]
print('Model Name: ' + file_name_pre)

#get all subject data
all_subject_instances = utility.load_subjects(directory + '/Training Data Analog Acc', store_in_ram )

#train test split
train_subject_instances, val_subject_instances = train_test_split( all_subject_instances  , test_size=0.2, random_state=300 ) #random_state=50

#make a train and val generator
train_gen = data_generators.make_generator_multiple_signal(list_of_subjects=train_subject_instances, cycle_per_batch=cycle_per_batch, eps=eps,frame_length=frame_length,
                                    mode=mode, list_sig_type_source= sig_type_source, sig_type_target= sig_type_target , down_sample_factor =down_sample_factor,
                                                           normalized=normalized , store_in_ram=store_in_ram ,
                                                           augment_accel = augment_accel , augment_theta_lim = augment_theta_lim , augment_prob=augment_prob)

val_gen = data_generators.make_generator_multiple_signal(list_of_subjects=val_subject_instances, cycle_per_batch=cycle_per_batch, eps=eps, frame_length=frame_length,
                                    mode=mode, list_sig_type_source= sig_type_source, sig_type_target= sig_type_target, down_sample_factor=down_sample_factor,
                                                         normalized=normalized, store_in_ram=store_in_ram)




#check !
#utility.diagnose_training_subjects(all_subject_instances)
#data_generators.diagnose_generator_multiple_signal(train_gen , sig_type_source)
#data_generators.diagnose_generator_multiple_signal(val_gen , sig_type_source)

#create model
if model_type=='Unetxl':
    sig_model = network_models.Unet_xl(input_size, kernel_size, filter_number,  len(sig_type_source) , no_layers )
# elif model_type=='Unet_5l':
#     sig_model = network_models.Unet_5l(input_size, kernel_size, filter_number,  len(sig_type_source) )
# elif model_type=='Unet_4l':
#     sig_model = network_models.Unet_4l(input_size, kernel_size, filter_number,  len(sig_type_source) )


sig_model = nn.DataParallel(sig_model.cuda(), device_ids=[0,1])
loss = network_models.PearsonRLoss()
cudnn.benchmark = True


#train model
config['n_epochs'] = 400
config['scheduler_milestones'] = [30,60,120] #[50,100,200]
config['train_steps'] = 10
config['val_steps'] = 10
config['initial_lr'] = 0.001
config['model_path'] = directory + '/Code Output/best_so_far.pt'
config['model_path_for_video'] = directory + '/Models for Video/' + file_name_pre


args = {'lr': config['initial_lr'],
        'n_epochs':config['n_epochs'],
        'model_path': config['model_path'],
        'step_count':config['train_steps'],
        'val_steps': config['val_steps'],
        'scheduler_milestones': config['scheduler_milestones'],
        'model_path_for_video': config['model_path_for_video']
}

start_model_train = time.time()
train_history, valid_history , best_val= network_models.train_torch_generator_with_video(args=args,
                                         sig_model=sig_model,
                                         criterion=loss,
                                         train_gen=train_gen,
                                         val_gen=val_gen,
                                         init_optimizer = lambda lr: Adam(sig_model.parameters(), lr=lr),
                                        init_schedule = lambda optimizer,milestones: torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5) ,
                                         produce_video=produce_video)

end_model_train = time.time()
print('Model Training Duration In Seconds: ' + str(end_model_train - start_model_train))
print('Best Validation Loss: ' + str(best_val))

#save model
sig_model = network_models.load_saved_model(model_path=config['model_path'],
                                        model_type= model_type,
                                        input_size=input_size ,
                                        kernel_size=kernel_size  ,
                                        filter_number=filter_number,
                                        signal_number=len(sig_type_source),
                                        no_layers = no_layers)

torch.save({
    'model': sig_model.state_dict(),
}, directory+'/Code Output/' + file_name_pre + '.pt')

pickle_list = [config, train_history , valid_history, train_subject_instances, val_subject_instances , best_val]
fileObject = open(directory+'/Code Output/' +file_name_pre+'_pickle','wb')
pickle.dump(pickle_list,fileObject)
fileObject.close()

#print results to text file
with open(directory + '/Code Output/' + file_name_pre + '_config.txt', "w") as text_file:
    print(config, file=text_file)
    print('Model Training Duration In Seconds: ' + str(end_model_train - start_model_train), file=text_file)

#model diagnosis
network_models.show_loss_torch_model(train_history, valid_history, file_name_pre , directory+'/Code Output')

#TODO
#try head to foot to bcg again train overnight !
#increase number of filters and maybe add dropout later
#GO EVEN DEEPER !!!
#try on HF data
# lms/rls compare, linear compare
#multi signal in multi signal out
#other metrics: correlation, dot product etc
#rnn

#head to foot -> bcg was not bad at all after 200 epocsh
