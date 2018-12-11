import cv2
import os
import datetime
import numpy as np
from model import SalGANmore
import pickle
import torch
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils import data
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from data_loader import DHF1K_frames

dtype = torch.FloatTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor

frame_size = (192, 256) # original shape is (360, 640, 3)
learning_rate = 0.000001 #
decay_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
epochs = 5
plot_every = 1
pretrained_model = None #This refers to a pretrained model beyond SalGAN. In any case we are loading pretrained SalGAN weights.
new_model = 'SalGANmid.pt'
clip_length = 10
number_of_videos = 4 # DHF1K offers 700 labeled videos, the other 300 are held back by the authors

TEMPORAL = True
FREEZE = True
SALGAN_WEIGHTS = 'model_weights/gen_model.pt'
#CONV_LSTM_WEIGHTS = './SalConvLSTM.pt' #These are not relevant in this problem after all, SalGAN was trained on a range of 0-255, the ConvLSTM was trained on a 0-1 range so they are incompatible.
#writer = SummaryWriter('./log') #Tensorboard, uncomment all lines containing writer if you wish to use this visualization tool

# Parameters
params = {'batch_size': 1, # number of videos / batch, I need to implement padding if I want to do more than 1, but with DataParallel it's quite messy
          'num_workers': 4,
          'pin_memory': True}


def main(params = params):

    # =================================================
    # ================ Data Loading ===================

    #Expect Error if either validation size or train size is 1
    train_set = DHF1K_frames(
        number_of_videos = number_of_videos,
        clip_length = clip_length,
        resolution = frame_size,
        split = "train")
    print("Size of train set is {}".format(len(train_set)))

    val_set = DHF1K_frames(
        number_of_videos = number_of_videos,
        clip_length = clip_length,
        resolution = frame_size,
        split = "validation")
    print("Size of validation set is {}".format(len(val_set)))

    whole_set = DHF1K_frames(
        number_of_videos = number_of_videos,
        clip_length = clip_length,
        resolution = frame_size,
        split = None)
    print("Size of whole set is {}".format(len(whole_set)))


    #print(len(train_set[0]))
    #print(len(train_set[1]))

    train_loader = data.DataLoader(train_set, **params)
    val_loader = data.DataLoader(val_set, **params)
    whole_loader = data.DataLoader(whole_set, **params)

    # =================================================
    # ================ Define Model ===================

    # The seed pertains to initializing the weights with a normal distribution
    # Using brute force for 100 seeds I found the number 65 to provide a good starting point (one that looks close to a saliency map predicted by the original SalGAN)
    if new_model == 'SalGANplus.pt':
        model = SalGANmore.SalGANplus(seed_init=65, freeze=FREEZE)
        print("Initialized {}".format(new_model))
    elif new_model == 'SalGANmid.pt':
        model = SalGANmore.SalGANmid(seed_init=65, freeze=FREEZE)
        print("Initialized {}".format(new_model))
    elif new_model == 'SalGAN.pt':
        model = SalGANmore.SalGAN()
        print("Initialized {}".format(new_model))
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

        if new_model == 'SalGANplus.pt':
            optimizer = torch.optim.Adam([{'params': model.Gates.parameters()},{'params': model.final_convs.parameters()}], learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)

        elif new_model == 'SalGANmid.pt':
            optimizer = torch.optim.Adam([{'params': model.Gates.parameters()}], learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)

    else:
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)


    if pretrained_model == None:

        # Load the weights of salgan generator.
        # By setting strict to False we allow the model to load only the matching layers' weights
        model.salgan.load_state_dict(torch.load(SALGAN_WEIGHTS), strict=False)
        start_epoch = 1

    else:

        # Load an entire pretrained model
        checkpoint = load_weights(model, pretrained_model)
        model.load_state_dict(checkpoint, strict=True)
        start_epoch = torch.load(pretrained_model, map_location='cpu')['epoch']
        #optimizer.load_state_dict(torch.load(pretrained_model, map_location='cpu')['optimizer'])

        print("Model loaded, commencing training from epoch {}".format(start_epoch))




    model = nn.DataParallel(model).cuda()
    cudnn.benchmark = True #https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    criterion = criterion.cuda()
    # =================================================
    # ================== Training =====================


    train_losses = []
    val_losses = []
    starting_time = datetime.datetime.now().replace(microsecond=0)
    print("Training started at : {}".format(starting_time))

    n_iter = 0
    for epoch in range(start_epoch, epochs+1):
        try:
            if epoch != epochs:
                #adjust_learning_rate(optimizer, epoch, decay_rate) #Didn't use this after all
                # train for one epoch
                train_loss, n_iter, optimizer = train(train_loader, model, criterion, optimizer, epoch, n_iter)

                val_loss = validate(val_loader, model, criterion, epoch)

                if epoch % plot_every == 0:
                    train_losses.append(train_loss.cpu())
                    val_losses.append(val_loss.cpu())

                print("Epoch {}/{} done with train loss {} and validation loss {}\n".format(epoch, epochs, train_loss, val_loss))
            else:

                print("Training on whole set")
                train_loss, n_iter, optimizer = train(whole_loader, model, criterion, optimizer, epoch, n_iter)
                print("Epoch {}/{} done with train loss {}".format(epoch, epochs, train_loss))


        except RuntimeError:
            print("A memory error was encountered. Further training aborted.")
            epoch = epoch - 1
            break

    print("Training started at {} and finished at : {} \n Now saving..".format(starting_time, datetime.datetime.now().replace(microsecond=0)))

    # ===================== #
    # ======  Saving ====== #

    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.cpu().state_dict(),
        'optimizer' : optimizer.state_dict()
        }, new_model)

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
        # Unfreeze layers depending on epoch number
        optimizer = model.module.thaw(epoch, optimizer) #When you wrap a model with DataParallel, the model.module can be seen as the model before it’s wrapped.

        # Confirm:
        model.module.print_layers()

    video_losses = []
    print("Now commencing epoch {}".format(epoch))
    for i, video in enumerate(train_loader):
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

            if TEMPORAL:
                #print(clip.size()) #works! torch.Size([5, 1, 1, 360, 640])
                loss = 0
                for idx in range(clip.size()[0]):
                    #print(clip[idx].size())

                    # Compute output
                    state, saliency_map = model.forward(input_ = clip[idx], prev_state = state) # Based on the number of epoch the model will unfreeze deeper layers moving on to shallow ones

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
                print(type(clip))
                print(clip.size())
                for idx in range(clip.size()[0]):
                    saliency_map = model.forward(clip[idx])
                    saliency_map = saliency_map.squeeze(0)
                    loss = criterion(saliency_map, gtruths[idx])
                    loss.backward()
                    optimizer.step()

                    accumulated_losses.append(loss.data)

            # Visualize some of the data
            if i%50==0 and j == 5:

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


