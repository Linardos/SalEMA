import cv2
import os
import datetime
import numpy as np
from model import SalGANmore, SalEMA
import pickle
import torch
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils import data
from torch.autograd import Variable
from data_loader import DHF1K_frames, Hollywood_frames



"""
Be sure to check the name of NEW_MODEL before running
"""
dataset_name = "UCF-sports"
dataset_name = "DHF1K"
dataset_name = "Hollywood-2"
frame_size = (192, 256) # original shape is (360, 640, 3)
learning_rate = 0.0000001 # Added another 0 for hollywood
decay_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
epochs = 7+1 #+3+2
plot_every = 1
clip_length = 10
starting_video = 1
number_of_videos = 600 #600 #Reset to 600

VAL_PERC = 0 #percentage of the given data to be used as validation on runtime. In our last experiments we validate after training so we don't use validation on runtime.
TEMPORAL = True
FREEZE = False
RESIDUAL = False
DROPOUT = True
SALGAN_WEIGHTS = 'model_weights/salgan_salicon.pt' #JuanJo's weights
#CONV_LSTM_WEIGHTS = './SalConvLSTM.pt' #These are not relevant in this problem after all, SalGAN was trained on a range of 0-255, the ConvLSTM was trained on a 0-1 range so they are incompatible.
ALPHA = 0.1
LEARN_ALPHA_ONLY = False
DOUBLE = False
EMA_LOC = 30     # 30 is the bottleneck
#EMA_LOC_2 = 54
PROCESS = 'parallel'
# Parameters
params = {'batch_size': 1, # number of videos / batch, I need to implement padding if I want to do more than 1, but with DataParallel it's quite messy
          'num_workers': 4,
          'pin_memory': True}

NEW_MODEL = 'SalEMA{}D_HU.pt'.format(EMA_LOC)
NEW_MODEL = 'SalGANmid_HU.pt'
NEW_MODEL = 'SalEMA{}A.pt'.format(EMA_LOC)
NEW_MODEL = 'SalEMA{}Afinal_H.pt'.format(EMA_LOC)
#NEW_MODEL = 'SalBCE.pt'
#NEW_MODEL = 'SalEMA{}&{}.pt'.format(EMA_LOC, EMA_LOC_2)
#pretrained_model = 'SalEMA{}.pt'.format(EMA_LOC)
pretrained_model = 'rawSalEMA{}D.pt'.format(EMA_LOC)
pretrained_model = 'SalEMA{}D_H.pt'.format(EMA_LOC)
pretrained_model = 'SalGANmid_H.pt'
pretrained_model = None
pretrained_model = 'SalEMA{}Afinal.pt'.format(EMA_LOC)
#NEW_MODEL = 'SalCLSTM30.pt'

dtype = torch.FloatTensor
if PROCESS == 'gpu' or PROCESS == 'parallel':
    assert(torch.cuda.is_available())
    dtype = torch.cuda.FloatTensor

def main(params = params):

    # =================================================
    # ================ Data Loading ===================

    #Expect Error if either validation size or train size is 1
    if dataset_name == "DHF1K":
        print("Commencing training on dataset {}".format(dataset_name))
        train_set = DHF1K_frames(
            number_of_videos = number_of_videos,
            starting_video = starting_video,
            clip_length = clip_length,
            resolution = frame_size,
            val_perc = VAL_PERC,
            split = "train")
        print("Size of train set is {}".format(len(train_set)))
        train_loader = data.DataLoader(train_set, **params)

        if VAL_PERC > 0:
            val_set = DHF1K_frames(
                number_of_videos = number_of_videos,
                starting_video = starting_video,
                clip_length = clip_length,
                resolution = frame_size,
                val_perc = VAL_PERC,
                split = "validation")
            print("Size of validation set is {}".format(len(val_set)))
            val_loader = data.DataLoader(val_set, **params)

    elif dataset_name == "Hollywood-2" or dataset_name == "UCF-sports":
        print("Commencing training on dataset {}".format(dataset_name))
        train_set = Hollywood_frames(
            root_path = "/imatge/lpanagiotis/work/{}/training".format(dataset_name),
            #root_path = "/home/linardosHollywood-2/training",
            clip_length = clip_length,
            resolution = frame_size,
            load_gt = True)
        video_name_list = train_set.video_names() #match an index to the sample video name
        train_loader = data.DataLoader(train_set, **params)

    else:
        print('Your model was not recognized. Check the name again.')
        exit()
    # =================================================
    # ================ Define Model ===================

    # The seed pertains to initializing the weights with a normal distribution
    # Using brute force for 100 seeds I found the number 65 to provide a good starting point (one that looks close to a saliency map predicted by the original SalGAN)
    if 'CLSTM56' in NEW_MODEL:
        model = SalGANmore.SalGANplus(seed_init=65, freeze=FREEZE)
        print("Initialized {}".format(NEW_MODEL))
    elif 'CLSTM30' in NEW_MODEL:
        model = SalGANmore.SalCLSTM30(seed_init=65, residual=RESIDUAL, freeze=FREEZE)
        print("Initialized {}".format(NEW_MODEL))
    elif NEW_MODEL == 'SalBCE.pt':
        model = SalGANmore.SalGAN()
        print("Initialized {}".format(NEW_MODEL))
    elif 'EMA' in NEW_MODEL:
        if DOUBLE:
            model = SalEMA.SalEMA2(alpha=ALPHA, ema_loc_1=EMA_LOC, ema_loc_2=EMA_LOC_2)
            print("Initialized {}".format(NEW_MODEL))
        else:
            model = SalEMA.SalEMA(alpha=ALPHA, residual=RESIDUAL, dropout= DROPOUT, ema_loc=EMA_LOC)
            print("Initialized {} with residual set to {} and dropout set to {}".format(NEW_MODEL, RESIDUAL, DROPOUT))
    else:
        print("Your model was not recognized, check the name of the model and try again.")
        exit()
    #criterion = nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and the BCELoss in one single class
    criterion = nn.BCELoss()
    #optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
    #optimizer = torch.optim.RMSprop(model.parameters(), learning_rate, alpha=0.99, eps=1e-08, momentum=momentum, weight_decay=weight_decay)
    #start

    if FREEZE:
        # Load only the unfrozen part to the optimizer

        if NEW_MODEL == 'SalGANplus.pt':
            optimizer = torch.optim.Adam([{'params': model.Gates.parameters()},{'params': model.final_convs.parameters()}], learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)

        elif 'SalCLSTM30' in NEW_MODEL:
            optimizer = torch.optim.Adam([{'params': model.Gates.parameters()}], learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)

    else:
        #optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
        optimizer = torch.optim.Adam([
            {'params':model.salgan.parameters() , 'lr': learning_rate, 'weight_decay':weight_decay},
            {'params':model.alpha, 'lr': 0.1}])
        if LEARN_ALPHA_ONLY:
            optimizer = torch.optim.Adam([{'params':[model.alpha]}], 0.1)



    if pretrained_model == None:
        # In truth it's not None, we default to SalGAN or SalBCE (JuanJo's)weights
        # By setting strict to False we allow the model to load only the matching layers' weights
        if SALGAN_WEIGHTS == 'model_weights/gen_model.pt':
            model.salgan.load_state_dict(torch.load(SALGAN_WEIGHTS), strict=False)
        else:
            model.salgan.load_state_dict(torch.load(SALGAN_WEIGHTS)['state_dict'], strict=False)


        start_epoch = 1

    else:

        # Load an entire pretrained model
        checkpoint = load_weights(model, pretrained_model)
        model.load_state_dict(checkpoint, strict=False)
        start_epoch = torch.load(pretrained_model, map_location='cpu')['epoch']
        #optimizer.load_state_dict(torch.load(pretrained_model, map_location='cpu')['optimizer'])

        print("Model loaded, commencing training from epoch {}".format(start_epoch))

    if PROCESS == 'parallel':
        model = nn.DataParallel(model).cuda()
    elif PROCESS == 'gpu':
        model = model.cuda()
    else:
        pass
    cudnn.benchmark = True #https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    criterion = criterion.cuda()
    # =================================================
    # ================== Training =====================


    train_losses = []
    val_losses = []
    starting_time = datetime.datetime.now().replace(microsecond=0)
    print("Training started at : {}".format(starting_time))

    n_iter = 0
    #if "EMA" in NEW_MODEL:
    #    print("Alpha value started at: {}".format(model.alpha))

    for epoch in range(start_epoch, epochs+1):

        try:
            #adjust_learning_rate(optimizer, epoch, decay_rate) #Didn't use this after all
            # train for one epoch
            train_loss, n_iter, optimizer = train(train_loader, model, criterion, optimizer, epoch, n_iter)

            print("Epoch {}/{} done with train loss {}\n".format(epoch, epochs, train_loss))

            if VAL_PERC > 0:
                print("Running validation..")
                val_loss = validate(val_loader, model, criterion, epoch)
                print("Validation loss: {}".format(val_loss))

            if epoch % plot_every == 0:
                train_losses.append(train_loss.cpu())
                if VAL_PERC > 0:
                    val_losses.append(val_loss.cpu())

            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.cpu().state_dict(),
                'optimizer' : optimizer.state_dict()
                }, NEW_MODEL)

            if PROCESS == 'parallel':
                model = nn.DataParallel(model).cuda()
            elif PROCESS == 'gpu':
                model = model.cuda()
            else:
                pass

            """
            else:

                print("Training on whole set")
                train_loss, n_iter, optimizer = train(whole_loader, model, criterion, optimizer, epoch, n_iter)
                print("Epoch {}/{} done with train loss {}".format(epoch, epochs, train_loss))
            """

        except RuntimeError:
            print("A memory error was encountered. Further training aborted.")
            epoch = epoch - 1
            break

    print("Training of {} started at {} and finished at : {} \n Now saving..".format(NEW_MODEL, starting_time, datetime.datetime.now().replace(microsecond=0)))
    #if "EMA" in NEW_MODEL:
    #    print("Alpha value tuned to: {}".format(model.alpha))
    # ===================== #
    # ======  Saving ====== #

    # If I try saving in regular intervals I have to move the model to CPU and back to GPU.
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.cpu().state_dict(),
        'optimizer' : optimizer.state_dict()
        }, NEW_MODEL)

    """
    hyperparameters = {
        'momentum' : momentum,
        'weight_decay' : weight_decay,
        'learning_rate' : learning_rate,
        'decay_rate' : decay_rate,
        'epochs' : epochs,
        'batch_size' : batch_size
    }
    """

    if VAL_PERC > 0:
        to_plot = {
            'epoch_ticks': list(range(start_epoch, epochs+1, plot_every)),
            'train_losses': train_losses,
            'val_losses': val_losses
            }
        with open('to_plot.pkl', 'wb') as handle:
            pickle.dump(to_plot, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ===================

mean = lambda x : sum(x)/len(x)


def adjust_learning_rate(optimizer, epoch, decay_rate=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (decay_rate ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_weights(model, pretrained_model, device='cpu'):
    # Load stored model:
    temp = torch.load(pretrained_model, map_location=device)['state_dict']
    # Because of dataparallel there is contradiction in the name of the keys so we need to remove part of the string in the keys:.
    from collections import OrderedDict
    checkpoint = OrderedDict()
    for key in temp.keys():
        new_key = key.replace("module.","")
        checkpoint[new_key]=temp[key]

    return checkpoint

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(train_loader, model, criterion, optimizer, epoch, n_iter):


    # Switch to train mode
    model.train()

    if TEMPORAL and FREEZE:
        if PROCESS == 'parallel':
            # Unfreeze layers depending on epoch number
            optimizer = model.module.thaw(epoch, optimizer) #When you wrap a model with DataParallel, the model.module can be seen as the model before it’s wrapped.

            # Confirm:
            model.module.print_layers()
        else:
            # Unfreeze layers depending on epoch number
            optimizer = model.thaw(epoch, optimizer)

            # Confirm:
            model.print_layers()

    video_losses = []
    print("Now commencing epoch {}".format(epoch))
    for i, video in enumerate(train_loader):
        """
        if i == 956 or i == 957:
            #some weird bug happens there
            continue
        """
        #print(type(video))
        accumulated_losses = []
        start = datetime.datetime.now().replace(microsecond=0)
        print("Number of clips for video {} : {}".format(i,len(video)))
        state = None # Initially no hidden state
        for j, (clip, gtruths) in enumerate(video):

            n_iter+=j

            # Reset Gradients
            optimizer.zero_grad()

            # Squeeze out the video dimension
            # [video_batch, clip_length, channels, height, width]
            # After transpose:
            # [clip_length, video_batch, channels, height, width]

            clip = Variable(clip.type(dtype).transpose(0,1))
            gtruths = Variable(gtruths.type(dtype).transpose(0,1))

            if TEMPORAL and not DOUBLE:
                #print(clip.size()) #works! torch.Size([5, 1, 1, 360, 640])
                loss = 0
                for idx in range(clip.size()[0]):
                    #print(clip[idx].size())

                    # Compute output
                    state, saliency_map = model.forward(input_ = clip[idx], prev_state = state) # Based on the number of epoch the model will unfreeze deeper layers moving on to shallow ones

                    saliency_map = saliency_map.squeeze(0) # Target is 3 dimensional (grayscale image)
                    if saliency_map.size() != gtruths[idx].size():
                        #print(saliency_map.size())
                        #print(gtruths[idx].size())
                        a, b, c, _ = saliency_map.size()
                        saliency_map = torch.cat([saliency_map, torch.zeros(a, b, c, 1).cuda()], 3) #because of upsampling we need to concatenate another column of zeroes. The original number is odd so it is impossible for upsampling to get an odd number as it scales by 2


                    # Apply sigmoid before visualization
                    # logits will be whatever you have to rescale this

                    # Compute loss
                    loss = loss + criterion(saliency_map, gtruths[idx])

                # Keep score
                accumulated_losses.append(loss.data)

                # Compute gradient
                loss.backward()


                # Clip gradient to avoid explosive gradients. Gradients are accumulated so I went for a threshold that depends on clip length. Note that the loss that is stored in the score for printing does not include this clipping.
                nn.utils.clip_grad_norm_(model.parameters(), 10*clip.size()[0])

                # Update parameters
                optimizer.step()

                # Repackage to avoid backpropagating further through time
                state = repackage_hidden(state)

            elif TEMPORAL and DOUBLE:
                if state == None:
                    state = (None, None)
                loss = 0
                for idx in range(clip.size()[0]):
                    #print(clip[idx].size())

                    # Compute output
                    state, saliency_map = model.forward(input_ = clip[idx], prev_state_1 = state[0], prev_state_2 = state[1]) # Based on the number of epoch the model will unfreeze deeper layers moving on to shallow ones

                    saliency_map = saliency_map.squeeze(0) # Target is 3 dimensional (grayscale image)
                    if saliency_map.size() != gtruths[idx].size():
                        print(saliency_map.size())
                        print(gtruths[idx].size())
                        a, b, c, _ = saliency_map.size()
                        saliency_map = torch.cat([saliency_map, torch.zeros(a, b, c, 1).cuda()], 3) #because of upsampling we need to concatenate another column of zeroes. The original number is odd so it is impossible for upsampling to get an odd number as it scales by 2


                    # Apply sigmoid before visualization
                    # logits will be whatever you have to rescale this

                    # Compute loss
                    loss = loss + criterion(saliency_map, gtruths[idx])

                # Keep score
                accumulated_losses.append(loss.data)

                # Compute gradient
                loss.backward()


                # Clip gradient to avoid explosive gradients. Gradients are accumulated so I went for a threshold that depends on clip length. Note that the loss that is stored in the score for printing does not include this clipping.
                nn.utils.clip_grad_norm_(model.parameters(), 10*clip.size()[0])

                # Update parameters
                optimizer.step()

                # Repackage to avoid backpropagating further through time
                state = repackage_hidden(state)

            else:
                #print(type(clip))
                #print(clip.size())
                for idx in range(clip.size()[0]):
                    saliency_map = model.forward(clip[idx])
                    saliency_map = saliency_map.squeeze(0)
                    loss = criterion(saliency_map, gtruths[idx])
                    loss.backward()
                    optimizer.step()

                    accumulated_losses.append(loss.data)

            # Visualize some of the data
            if i%100==0 and j == 5:

                #writer.add_image('Frame', clip[idx], n_iter)
                #writer.add_image('Gtruth', gtruths[idx], n_iter)

                post_process_saliency_map = (saliency_map-torch.min(saliency_map))/(torch.max(saliency_map)-torch.min(saliency_map))
                utils.save_image(post_process_saliency_map, "./log/smap{}_epoch{}.png".format(i, epoch))

                if epoch == 1:
                    print(saliency_map.max())
                    print(saliency_map.min())
                    print(gtruths[idx].max())
                    print(gtruths[idx].min())
                    print(post_process_saliency_map.max())
                    print(post_process_saliency_map.min())
                    utils.save_image(gtruths[idx], "./log/gt{}.png".format(i))
                #writer.add_image('Prediction', prediction, n_iter)


        end = datetime.datetime.now().replace(microsecond=0)
        print('Epoch: {}\tVideo: {}\t Training Loss: {}\t Time elapsed: {}\t'.format(epoch, i, mean(accumulated_losses), end-start))
        video_losses.append(mean(accumulated_losses))

    return (mean(video_losses), n_iter, optimizer)


def validate(val_loader, model, criterion, epoch):

    # switch to evaluate mode
    model.eval()

    video_losses = []
    print("Now running validation..")
    for i, video in enumerate(val_loader):
        accumulated_losses = []
        state = None # Initially no hidden state
        for j, (clip, gtruths) in enumerate(video):

            clip = Variable(clip.type(dtype).transpose(0,1), requires_grad=False)
            gtruths = Variable(gtruths.type(dtype).transpose(0,1), requires_grad=False)

            loss = 0
            for idx in range(clip.size()[0]):
                #print(clip[idx].size()) needs unsqueeze
                # Compute output
                if TEMPORAL:
                    state, saliency_map = model.forward(clip[idx], state)
                else:
                    saliency_map = model.forward(clip[idx])

                saliency_map = saliency_map.squeeze(0)

                if saliency_map.size() != gtruths[idx].size():
                    a, b, c, _ = saliency_map.size()
                    saliency_map = torch.cat([saliency_map, torch.zeros(a, b, c, 1).cuda()], 3) #because of upsampling we need to concatenate another column of zeroes. The original number is odd so it is impossible for upsampling to get an odd number as it scales by 2

                # Compute loss
                loss = loss + criterion(saliency_map, gtruths[idx])

            if TEMPORAL:
                state = repackage_hidden(state)

            # Keep score
            accumulated_losses.append(loss.data)

        video_losses.append(mean(accumulated_losses))

    return(mean(video_losses))

if __name__ == '__main__':
    main()

    #utils.save_image(saliency_map.data.cpu(), "test.png")


