import torch.nn as nn
import torch.nn.functional as F
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt


class ConvElu(nn.Module):
    '''
    Conv + BatchNorm (optional) + ELU, used in encoder and decoder blocks
    '''
    def __init__(self,filter_number_in, filter_number_out , kernel_size ,batch_norm=False ):
        super(ConvElu, self).__init__()
        self.batch_norm =batch_norm
        self.conv = nn.Conv1d(filter_number_in, filter_number_out, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(filter_number_out)
        self.elu = nn.ELU(inplace=False)

    def forward(self, x):
        if self.batch_norm:
            return self.elu(self.bn(self.conv(x)))
        else:
            return self.elu(self.conv(x))


class ConvTransposeElu(nn.Module):
    '''
    Transpose-Conv + BatchNorm (optional) + ELU, used in decoder blocks
    '''
    def __init__(self,filter_number_in, filter_number_out ,batch_norm=False  ):
        super(ConvTransposeElu, self).__init__()
        self.batch_norm =batch_norm
        self.conv_transpose = nn.ConvTranspose1d(filter_number_in, filter_number_out, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm1d(filter_number_out)
        self.elu = nn.ELU(inplace=False)

    def forward(self, x):

        if self.batch_norm:
            return self.elu(self.bn(self.conv_transpose(x)))
        else:
            return self.elu(self.conv_transpose(x))



class EncoderBlock(nn.Module):

    def __init__(self, filter_number_in, filter_number_out , kernel_size):
        super(EncoderBlock, self).__init__()

        self.block = nn.Sequential(

            ConvElu(filter_number_in, filter_number_out , kernel_size , batch_norm=True ),
            #nn.Dropout(),
            #ConvElu(filter_number_out, filter_number_out, kernel_size, batch_norm=True),
            #nn.Dropout(),
            #ConvElu(filter_number_out, filter_number_out, kernel_size, batch_norm=True),
        )

    def forward(self, x):
        return self.block(x)



class DecoderBlock(nn.Module):

    def __init__(self, filter_number_in, filter_number_out , kernel_size):
        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(

            ConvElu(filter_number_in, filter_number_out , kernel_size , batch_norm=True ),
            #nn.Dropout(),
            #ConvElu(filter_number_out, filter_number_out, kernel_size, batch_norm=True),
            # nn.Dropout(),
            #ConvElu(filter_number_out, filter_number_out, kernel_size, batch_norm=True),

        )

    def forward(self, x):
        return self.block(x)




class Unet_xl(nn.Module):
    '''
    UNet-like neural network architecture to map source signals to a target signal
    '''
    def __init__(self,input_size, kernel_size  , filter_number , sig_number, layer_number):

        super().__init__()

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.filter_number=filter_number
        self.layer_number=layer_number

        self.encoder = nn.ModuleList()
        self.drops_encoder = nn.ModuleList()

        for u in range(layer_number):
            if u ==0:
                self.encoder.append(EncoderBlock(sig_number, filter_number, kernel_size))
            else:
                self.encoder.append(EncoderBlock(u*filter_number, (u+1)*filter_number, kernel_size))
            self.drops_encoder.append(nn.Dropout())

        self.conv_middle = nn.Conv1d((u+1)*filter_number, (u+2)*filter_number , kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.bn_middle = nn.BatchNorm1d((u+2)*filter_number )
        self.elu = nn.ELU(inplace=False)
        self.drop_middle = nn.Dropout()

        self.decoder = nn.ModuleList()
        self.conv_tr = nn.ModuleList()
        self.drops_decoder = nn.ModuleList()

        for u in range(layer_number):
            self.conv_tr.append(ConvTransposeElu((u+2)*filter_number,(u+1)*filter_number , batch_norm=True ) )
            self.drops_decoder.append(nn.Dropout())
            self.decoder.append(DecoderBlock((2*(u+1))* filter_number, (u+1)* filter_number, kernel_size))

        self.conv_final = nn.Conv1d(filter_number, 1 , kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

        self.final_sigmoid = nn.Sigmoid()


    def forward(self, x):

        encoded = []


        for u in range(self.layer_number):

            x = self.encoder[u](x)


            encoded.append(x)
            x=F.max_pool1d(self.drops_encoder[u](x),2)


        x= self.conv_middle(x)
        x=self.bn_middle(x)
        x=self.elu(x)
        x=self.drop_middle(x)

        for u in reversed(range(self.layer_number)):

            x = self.conv_tr[u](x)


            x = self.drops_decoder[u](x)

            x = self.decoder[u](torch.cat([encoded[u], x], 1))


        x = self.conv_final(x)
        x = self.final_sigmoid(x)

        return x




class PearsonRLoss(nn.Module):
    '''
    Loss function to minimize, negative of the Pearson Correlation Coefficient
    '''
    def __init__(self):
        super(PearsonRLoss, self).__init__()

    def forward(self, outputs, targets ):

        outputs_hat = outputs-torch.mean(outputs,dim=1).view(-1,1)
        targets_hat = targets - torch.mean(targets, dim=1).view(-1,1)


        outputs_norm = torch.sqrt(outputs_hat.pow(2).sum(dim=1).view(-1,1))
        targets_norm = torch.sqrt(targets_hat.pow(2).sum(dim=1).view(-1,1))


        outputs_0_mean_unit_norm = outputs_hat/outputs_norm
        targets_0_mean_unit_norm = targets_hat / targets_norm


        #elementwise multiply
        pearsonr_r_batch = (outputs_0_mean_unit_norm*targets_0_mean_unit_norm).sum(dim=1)

        return -pearsonr_r_batch.mean()


def cuda(x):
    '''
    send module tensor to cuda
    :param x: pytorch tensor
    :return: pytorch tensor sent to cuda
    '''
    return x.cuda(async=True) if torch.cuda.is_available() else x

def get_learning_rate(optimizer):
    '''
    get current learning rate of optimizer
    :param optimizer: optimizer being used
    :return: learning rate
    '''
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]
    return lr[0]


def train_torch_generator_with_video(args, sig_model, criterion, train_gen, val_gen, init_optimizer, init_schedule ,produce_video):
    '''
    trains the pytorch model, the model with the lowest validation error is saved
    :param args: training configs
    :param sig_model: pytorch model
    :param criterion: loss functions
    :param train_gen: training data generator
    :param val_gen: validation data generator
    :param init_optimizer: initial optimizer
    :param init_schedule: optimization schedule
    :param produce_video: produce a video ? if so saves the models from each epoch
    :return: dict of training and validation history (losses at each epoch) and the best validation score
    '''

    lr = args['lr']
    n_epochs = args['n_epochs']
    model_path= args['model_path']
    step_count = args['step_count']
    val_steps = args['val_steps']
    scheduler_milestones = args['scheduler_milestones']
    model_path_for_video= args['model_path_for_video']
    optimizer = init_optimizer(lr)
    scheduler = init_schedule( optimizer , scheduler_milestones)
    torch.cuda.empty_cache()


    epoch = 1
    step = 0


    valid_history = []
    train_history = []
    best_val = float('inf')

    for epoch in range(epoch, n_epochs + 1):
        scheduler.step()
        sig_model.train()
        tq = tqdm.tqdm(total=(step_count))
        tq.set_description('Epoch {}, lr {:.5f}'.format(epoch, get_learning_rate(optimizer)))
        losses = []
        try:
            for u in range(step_count):

                torch.cuda.empty_cache()
                inputs, targets , _ = next(train_gen)
                inputs = torch.from_numpy(inputs)
                inputs = cuda(inputs)
                inputs = inputs.type(torch.cuda.FloatTensor)

                outputs = sig_model(inputs).squeeze()

                del inputs

                targets = torch.from_numpy(targets)
                with torch.no_grad():
                    targets = cuda(targets)
                targets = targets.type(torch.cuda.FloatTensor)
                targets = targets.squeeze()
                torch.cuda.empty_cache()

                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(1)

                losses.append(loss.item())


                mean_loss = losses[-1]

                tq.set_postfix(loss= '{:.5f}'.format(mean_loss)  )


            tq.close()


            train_loss = np.mean(losses)


            print('')
            print('Train loss: {:.5f}'.format(train_loss)  )
            train_history.append({'train_loss': train_loss })


            valid_metrics = validation_binary(sig_model, criterion, val_gen , val_steps)
            valid_history.append(valid_metrics)

            #save if validation metric improved
            if valid_history[-1]['valid_loss']<best_val:
                print('valid_loss improved from ' + str(best_val) + ' to ' + str(valid_history[-1]['valid_loss']) + ' saving model...')
                best_val =valid_history[-1]['valid_loss']
                torch.save({
                    'model': sig_model.state_dict(),
                }, str(model_path))

            if produce_video:
                if epoch<10:
                    epoch_str = '000' + str(epoch)
                elif epoch>=10 and epoch<100:
                    epoch_str = '00' + str(epoch)
                elif epoch >= 100 and epoch<1000:
                    epoch_str = '0' + str(epoch)
                else:
                    epoch_str = str(epoch)

                torch.save({
                    'model': sig_model.state_dict(),
                }, model_path_for_video+'_' + str(epoch_str) +'.pt')

        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            print('done.')
            return train_history, valid_history, best_val

    return train_history, valid_history, best_val




def validation_binary(sig_model, criterion, val_gen , val_steps):
    '''
    predict on validation data with the current state of the model being trained
    :param sig_model: current model
    :param criterion: loss function
    :param val_gen: validation data generator
    :param val_steps: number of times the model tested on validation data, validation loss is the mean of the losses from each repetition
    :return: dict validation metrics
    '''
    with torch.no_grad():
        sig_model.eval()
        losses = []
        for u in range(val_steps):

            torch.cuda.empty_cache()
            inputs, targets , _= next(val_gen)
            inputs = torch.from_numpy(inputs)
            inputs = cuda(inputs)
            inputs = inputs.type(torch.cuda.FloatTensor)

            outputs = sig_model(inputs).squeeze()
            del inputs

            targets = torch.from_numpy(targets)
            with torch.no_grad():
                targets = cuda(targets)
            targets = targets.type(torch.cuda.FloatTensor)
            targets = targets.view(targets.size(0), -1)
            torch.cuda.empty_cache()

            loss = criterion(outputs, targets)
            losses.append(loss.item())


            del targets


        valid_loss = np.mean(losses)  # type: float

        print('')
        print('Valid loss: {:.5f}'.format(valid_loss))
        metrics = {'valid_loss': valid_loss}

        return metrics



def show_loss_torch_model(train_history, valid_history, file_name_pre , directory):
    '''
    plot the training loss and validation loss over the epochs of training
    :param train_history: dictionary of training history
    :param valid_history: dictionary of validation history
    :param file_name_pre: str file name prefix
    :param directory: str directory where the figure will be saved
    :return: -
    '''
    fig=plt.figure()
    train_loss = [ x['train_loss'] for x in train_history]
    val_loss = [ x['valid_loss'] for x in valid_history]
    plt.plot(range(1, len(train_loss) + 1), train_loss, 'bo', label='train loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, 'r', label='val loss')
    plt.legend()
    fig.savefig(directory + '/' + file_name_pre + ' results2.png')


def load_saved_model(model_path ,model_type, input_size , kernel_size  , filter_number=64 , signal_number = 1 , no_layers = 1):
    '''
    load saved pytorch model
    :param model_path: model directory
    :param model_type: str model architecture
    :param input_size: int number of samples that each signal segment input to the model has
    :param kernel_size: size of filter in conv layers
    :param filter_number: int n_f parameter
    :param signal_number: int how many signals are input to the network
    :param no_layers: int how many "levels" deep the network is
    :return: pytorch model
    '''
    torch.cuda.empty_cache()
    if model_type == 'Unetxl':
        model = Unet_xl(input_size, kernel_size, filter_number,signal_number,no_layers)



    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)


    model.eval()
    if torch.cuda.is_available():
        return model.cuda()

    return model

