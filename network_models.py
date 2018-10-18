import torch.nn as nn
import torch.nn.functional as F
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt


class ConvElu(nn.Module):
    def __init__(self,filter_number_in, filter_number_out , kernel_size ,batch_norm=False ):
        super(ConvElu, self).__init__()
        self.batch_norm =batch_norm
        self.conv = nn.Conv1d(filter_number_in, filter_number_out, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(filter_number_out)
        self.elu = nn.ELU(inplace=False)

    def forward(self, x):
        #y = self.conv(x)
        #y =self.elu(y)
        if self.batch_norm:
            return self.elu(self.bn(self.conv(x)))
        else:
            return self.elu(self.conv(x))


class ConvTransposeElu(nn.Module):
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

#
# class Unet_4l(nn.Module):
#     def __init__(self,input_size, kernel_size  , filter_number , sig_number):
#
#         super().__init__()
#
#         self.input_size = input_size
#         self.kernel_size = kernel_size
#         self.filter_number=filter_number
#
#         self.enc_1 = EncoderBlock(sig_number, filter_number, kernel_size)
#         self.dropout_1 = nn.Dropout()
#
#
#         self.enc_2 = EncoderBlock(1*filter_number, 2*filter_number, kernel_size)
#         self.dropout_2 = nn.Dropout()
#
#
#
#         self.enc_3 = EncoderBlock(2*filter_number, 4*filter_number, kernel_size)
#         self.dropout_3 = nn.Dropout()
#
#
#         self.enc_4 = EncoderBlock(4*filter_number, 8*filter_number, kernel_size)
#         self.dropout_4 = nn.Dropout()
#
#         self.enc_5 = EncoderBlock(8*filter_number, 16*filter_number, kernel_size)
#         self.dropout_5 = nn.Dropout()
#
#
#         self.conv_t_1 = ConvTransposeElu(16*filter_number,8*filter_number)
#         self.dec_1 = DecoderBlock(16*filter_number, 8*filter_number, kernel_size)
#
#         self.conv_t_2 = ConvTransposeElu(8 * filter_number, 4 * filter_number)
#         self.dec_2 = DecoderBlock(8*filter_number, 4*filter_number, kernel_size)
#
#         self.conv_t_3 = ConvTransposeElu(4 * filter_number, 2 * filter_number)
#         self.dec_3 = DecoderBlock(4*filter_number, 2*filter_number, kernel_size)
#
#         self.conv_t_4 = ConvTransposeElu(2 * filter_number, 1 * filter_number)
#         self.dec_4 = DecoderBlock(2*filter_number, 1*filter_number, kernel_size)
#
#         self.conv_1d = nn.Conv1d(filter_number, 1 , kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
#
#         self.final_sigmoid = nn.Sigmoid()
#
#
#     def forward(self, x0):
#
#         x1= self.enc_1(x0)
#         x2= F.max_pool1d(self.dropout_1(x1),2)
#
#         x3= self.enc_2(x2)
#         x4= F.max_pool1d(self.dropout_2(x3),2)
#
#
#         x5= self.enc_3(x4)
#         x6= F.max_pool1d(self.dropout_3(x5),2)
#
#         x7= self.enc_4(x6)
#         x8= F.max_pool1d(self.dropout_4(x7),2)
#
#         x9= self.enc_5(x8)
#         x10=self.dropout_5(x9)
#
#
#         x11 = self.conv_t_1(x10)
#         x12 = self.dec_1(torch.cat([x7, x11], 1))
#
#         x13 = self.conv_t_2(x12)
#         x14 = self.dec_2(torch.cat([x5, x13], 1))
#
#         x15 = self.conv_t_3(x14)
#         x16 = self.dec_3(torch.cat([x3, x15], 1))
#
#         x17 = self.conv_t_4(x16)
#         x18 = self.dec_4(torch.cat([x1, x17], 1))
#
#         x19 = self.conv_1d(x18)
#
#         x20 = self.final_sigmoid(x19)
#
#         return x20
#
#
#
#
#
# class Unet_5l(nn.Module):
#     def __init__(self,input_size, kernel_size  , filter_number , sig_number):
#
#         super().__init__()
#
#         self.input_size = input_size
#         self.kernel_size = kernel_size
#         self.filter_number=filter_number
#
#         self.enc_1 = EncoderBlock(sig_number, filter_number, kernel_size)
#         self.dropout_1 = nn.Dropout()
#
#
#         self.enc_2 = EncoderBlock(1*filter_number, 2*filter_number, kernel_size)
#         self.dropout_2 = nn.Dropout()
#
#
#
#         self.enc_3 = EncoderBlock(2*filter_number, 4*filter_number, kernel_size)
#         self.dropout_3 = nn.Dropout()
#
#
#         self.enc_4 = EncoderBlock(4*filter_number, 8*filter_number, kernel_size)
#         self.dropout_4 = nn.Dropout()
#
#         self.enc_5 = EncoderBlock(8*filter_number, 16*filter_number, kernel_size)
#         self.dropout_5 = nn.Dropout()
#
#         self.enc_6 = EncoderBlock(16*filter_number, 32*filter_number, kernel_size)
#         self.dropout_6 = nn.Dropout()
#
#
#         self.conv_t_1 = ConvTransposeElu(32*filter_number,16*filter_number)
#         self.dec_1 = DecoderBlock(32*filter_number, 16*filter_number, kernel_size)
#
#         self.conv_t_2 = ConvTransposeElu(16 * filter_number, 8 * filter_number)
#         self.dec_2 = DecoderBlock(16*filter_number, 8*filter_number, kernel_size)
#
#         self.conv_t_3 = ConvTransposeElu(8 * filter_number, 4 * filter_number)
#         self.dec_3 = DecoderBlock(8*filter_number, 4*filter_number, kernel_size)
#
#         self.conv_t_4 = ConvTransposeElu(4 * filter_number, 2 * filter_number)
#         self.dec_4 = DecoderBlock(4*filter_number, 2*filter_number, kernel_size)
#
#         self.conv_t_5 = ConvTransposeElu(2 * filter_number, 1 * filter_number)
#         self.dec_5 = DecoderBlock(2*filter_number, 1*filter_number, kernel_size)
#
#         self.conv_1d = nn.Conv1d(filter_number, 1 , kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
#
#         self.final_sigmoid = nn.Sigmoid()
#
#
#     def forward(self, x0):
#
#         x1= self.enc_1(x0)
#         x2= F.max_pool1d(self.dropout_1(x1),2)
#
#         x3= self.enc_2(x2)
#         x4= F.max_pool1d(self.dropout_2(x3),2)
#
#         x5= self.enc_3(x4)
#         x6= F.max_pool1d(self.dropout_3(x5),2)
#
#         x7= self.enc_4(x6)
#         x8= F.max_pool1d(self.dropout_4(x7),2)
#
#         x9= self.enc_5(x8)
#         x10= F.max_pool1d(self.dropout_5(x9),2)
#
#         x11= self.enc_6(x10)
#         x12=self.dropout_6(x11)
#
#         x13 = self.conv_t_1(x12)
#         x14 = self.dec_1(torch.cat([x9, x13], 1))
#
#         x15 = self.conv_t_2(x14)
#         x16 = self.dec_2(torch.cat([x7, x15], 1))
#
#         x17 = self.conv_t_3(x16)
#         x18 = self.dec_3(torch.cat([x5, x17], 1))
#
#         x19 = self.conv_t_4(x18)
#         x20 = self.dec_4(torch.cat([x3, x19], 1))
#
#         x21 = self.conv_t_5(x20)
#         x22 = self.dec_5(torch.cat([x1, x21], 1))
#
#         x23 = self.conv_1d(x22)
#
#         x24 = self.final_sigmoid(x23)
#
#         return x24



# class Unet_xl(nn.Module):
#     def __init__(self,input_size, kernel_size  , filter_number , sig_number, layer_number):
#
#         super().__init__()
#
#         self.input_size = input_size
#         self.kernel_size = kernel_size
#         self.filter_number=filter_number
#         self.layer_number=layer_number
#
#         self.encoder = nn.ModuleList()
#         self.drops_encoder = nn.ModuleList()
#
#         for u in range(layer_number):
#             if u ==0:
#                 self.encoder.append(EncoderBlock(sig_number, filter_number, kernel_size))
#             else:
#                 self.encoder.append(EncoderBlock((2**(u-1))*filter_number, (2**(u))*filter_number, kernel_size))
#             self.drops_encoder.append(nn.Dropout())
#
#         self.conv_middle = nn.Conv1d((2**(u))*filter_number, (2**(u+1))*filter_number , kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
#         self.bn_middle = nn.BatchNorm1d((2**(u+1))*filter_number )
#         self.elu = nn.ELU(inplace=False)
#         self.drop_middle = nn.Dropout()
#
#         self.decoder = nn.ModuleList()
#         self.conv_tr = nn.ModuleList()
#         self.drops_decoder = nn.ModuleList()
#
#         for u in range(layer_number):
#             self.decoder.append(DecoderBlock((2**(u+1))* filter_number, (2**u)* filter_number, kernel_size))
#             self.conv_tr.append(ConvTransposeElu((2**(u+1))*filter_number,(2**u)*filter_number , batch_norm=True ) )
#             self.drops_decoder.append(nn.Dropout())
#
#         self.conv_final = nn.Conv1d(filter_number, 1 , kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
#
#         self.final_sigmoid = nn.Sigmoid()
#
#
#     def forward(self, x):
#
#         encoded = []
#         #print(x.shape)
#         for u in range(self.layer_number):
#             x = self.encoder[u](x)
#             encoded.append(x)
#             x=F.max_pool1d(self.drops_encoder[u](x),2)
#             #print(x.shape)
#
#         x= self.conv_middle(x)
#         x=self.bn_middle(x)
#         x=self.elu(x)
#         x=self.drop_middle(x)
#         #print(x.shape)
#
#         for u in reversed(range(self.layer_number)):
#             x = self.conv_tr[u](x)
#             x = self.drops_decoder[u](x)
#             # print(self.conv_tr[u](x).shape)
#             x = self.decoder[u](torch.cat([encoded[u], x], 1))
#             #print(x.shape)
#
#         x = self.conv_final(x)
#         #print(x.shape)
#
#         x = self.final_sigmoid(x)
#         #print(x.shape)
#
#         return x
#
#
#


class Unet_xl(nn.Module):
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
        #print(x.shape)
        for u in range(self.layer_number):
            x = self.encoder[u](x)
            encoded.append(x)
            x=F.max_pool1d(self.drops_encoder[u](x),2)
            #print(x.shape)

        x= self.conv_middle(x)
        x=self.bn_middle(x)
        x=self.elu(x)
        x=self.drop_middle(x)
        #print(x.shape)

        for u in reversed(range(self.layer_number)):
            x = self.conv_tr[u](x)
            x = self.drops_decoder[u](x)
            # print(self.conv_tr[u](x).shape)
            x = self.decoder[u](torch.cat([encoded[u], x], 1))
            #print(x.shape)

        x = self.conv_final(x)
        #print(x.shape)

        x = self.final_sigmoid(x)
        #print(x.shape)

        return x






class PearsonRLoss(nn.Module):

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
        #pearsonr_r_batch = torch.abs((outputs_0_mean_unit_norm*targets_0_mean_unit_norm).sum(dim=1))
        pearsonr_r_batch = (outputs_0_mean_unit_norm*targets_0_mean_unit_norm).sum(dim=1)

        return -pearsonr_r_batch.mean()


def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]
    return lr[0]


def train_torch_generator_with_video(args, sig_model, criterion, train_gen, val_gen, init_optimizer, init_schedule ,produce_video):


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
        #acc = []
        #aucs = []
        try:
            for u in range(step_count):

                torch.cuda.empty_cache()
                inputs, targets , _ = next(train_gen)
                inputs = torch.from_numpy(inputs)
                inputs = cuda(inputs)
                inputs = inputs.type(torch.cuda.FloatTensor)


                outputs = sig_model(inputs).squeeze()
                #print(outputs)
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
                #acc+=[get_accuracy(targets.long(), (outputs > 0.5 ).long()) ]
                #aucs+=[get_auc(targets, outputs)]

                mean_loss = losses[-1]#np.mean(losses[-report_each:])
                #mean_acc = acc[-1]  # np.mean(losses[-report_each:])
                #mean_auc = aucs[-1]
                tq.set_postfix(loss= '{:.5f}'.format(mean_loss)  )


            tq.close()


            train_loss = np.mean(losses)
            #train_acc = np.mean(acc)
            #train_auc = np.mean(aucs)

            print('')
            print('Train loss: {:.5f}'.format(train_loss)  )
            train_history.append({'train_loss': train_loss })


            valid_metrics = validation_binary(sig_model, criterion, val_gen , val_steps)
            #valid_metrics = validation_score_per_file(sig_model, criterion, val_gen, val_steps, threshold=0.5)
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
            #save(epoch)
            print('done.')
            return train_history, valid_history

    return train_history, valid_history





def validation_binary(sig_model, criterion, val_gen , val_steps):
    with torch.no_grad():
        sig_model.eval()
        losses = []
        #acc = []
        #aucs=[]
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
            #acc+=[get_accuracy(targets.long(), (outputs > 0.5 ).long()) ]
            #aucs += [get_auc(targets, outputs)]

            del targets


        valid_loss = np.mean(losses)  # type: float
        #valid_acc = np.mean(acc)
        #valid_auc = np.mean(aucs)

        print('')
        print('Valid loss: {:.5f}'.format(valid_loss))
        metrics = {'valid_loss': valid_loss}

        return metrics



def show_loss_torch_model(train_history, valid_history, file_name_pre , directory):


    fig=plt.figure()
    train_loss = [ x['train_loss'] for x in train_history]
    val_loss = [ x['valid_loss'] for x in valid_history]
    plt.plot(range(1, len(train_loss) + 1), train_loss, 'bo', label='train loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, 'r', label='val loss')
    plt.legend()
    fig.savefig(directory + '/' + file_name_pre + ' results2.png')


def load_saved_model(model_path ,model_type, input_size , kernel_size  , filter_number=64 , signal_number = 1 , no_layers = 1):
    torch.cuda.empty_cache()
    if model_type == 'Unetxl':
        model = Unet_xl(input_size, kernel_size, filter_number,signal_number,no_layers)
    # elif model_type == 'Unet_5l':
    #     model = Unet_5l(input_size, kernel_size, filter_number,signal_number)
    # elif model_type == 'Unet_4l':
    #     model = Unet_4l(input_size, kernel_size, filter_number,signal_number)


    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)


    model.eval()
    if torch.cuda.is_available():
        return model.cuda()

    return model

