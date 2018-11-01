import cv2
import os
import datetime
import numpy as np
from SalGANmore import SalGAN, SalGANplus, SalGANmid
import pickle
import torch
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils import data
from torch.autograd import Variable
#from tensorboardX import SummaryWriter
from data_loader import DHF1K_frames

dtype = torch.FloatTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor

"""
Based on the superior BCE-based loss compared with MSE,
we also explored the impact of computing the content loss over
downsampled versions of the saliency map. This technique re-
duces the required computational resources at both training and
test times and, as shown in Table 3, not only does it not decrease
performance, but it can actually improve it. Given this results,
we chose to train SalGAN on saliency maps downsampled by
a factor 1/4, which in our architecture corresponds to saliency
maps of 64 × 48. - Salgan paper
"""

GPU = 2
frame_size = 64 # original shape is (360, 640, 3)
learning_rate = 0.00001 #
decay_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
start_epoch = 1
epochs = 3
plot_every = 1
load_model = False
#pretrained_model = './SalGANmid.pt'
clip_length = 10
number_of_videos = 700 # DHF1K offers 700 labeled videos, the other 300 are held back by the authors

SALGAN_WEIGHTS = 'model_weights/gen_model.pt'
CONV_LSTM_WEIGHTS = './SalConvLSTM.pt'
#writer = SummaryWriter('./log') #Tensorboard

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
        split = "train",
        transforms = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ]))
    print("Size of train set is {}".format(len(train_set)))

    val_set = DHF1K_frames(
        number_of_videos = number_of_videos,
        clip_length = clip_length,
        split = "validation",
        transforms = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ]))
    print("Size of validation set is {}".format(len(val_set)))

    #print(len(train_set[0]))
    #print(len(train_set[1]))

    train_loader = data.DataLoader(train_set, **params)
    val_loader = data.DataLoader(val_set, **params)


    # =================================================
    # ================ Define Model ===================

    # The seed pertains to initializing the weights with a normal distribution
    # Using brute force for 100 seeds I found the number 65 to provide a good starting point (one that looks close to a saliency map predicted by the original SalGAN)
    model = SalGANplus(seed_init=65)

    # Load the weights of salgan generator.
    # By setting strict to False we allow the model to load only the matching layers' weights
    model.salgan.load_state_dict(torch.load(SALGAN_WEIGHTS), strict=False)

    checkpoint = load_weights(model, CONV_LSTM_WEIGHTS)
    model.Gates.load_state_dict(checkpoint, strict=False)
    model.conv1x1.load_state_dict(checkpoint, strict=False)

    if load_model:
        checkpoint = load_weights(model, pretrained_model)
        model.load_state_dict(checkpoint, strict=True)

    criterion = nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and the BCELoss in one single class

    #optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.RMSprop(model.parameters(), learning_rate, alpha=0.99, eps=1e-08, momentum=momentum, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=4, min_lr=0.0000001)

    if GPU >= 2:
        model = nn.DataParallel(model).cuda()
        cudnn.benchmark = True #https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    if GPU >= 1:
        model.cuda()
        criterion = criterion.cuda()

    # =================================================
    # ================== Training =====================


    train_losses = []
    val_losses = []
    starting_time = datetime.datetime.now().replace(microsecond=0)
    print("Training started at : {}".format(starting_time))
    for epoch in range(start_epoch, epochs+1):
        #adjust_learning_rate(optimizer, epoch, decay_rate)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        val_loss = validate(val_loader, model, criterion, epoch)
        # if validation has not improved for a certain amount of epochs, reduce the learning rate:
        scheduler.step(val_loss)

        if epoch % plot_every == 0:
            train_losses.append(train_loss.cpu())
            val_losses.append(val_loss.cpu())

        print("Epoch {}/{} done with train loss {} and validation loss {}\n".format(epoch, epochs, train_loss, val_loss))

    print("Training started at {} and finished at : {} \n Now saving..".format(starting_time, datetime.datetime.now().replace(microsecond=0)))

    # ===================== #
    # ======  Saving ====== #

    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.cpu().state_dict(),
        'optimizer' : optimizer.state_dict()
        }, 'SalGANplus.pt')
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
        'epoch_ticks': list(range(start_epoch, epochs, plot_every)),
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

def train(train_loader, model, criterion, optimizer, epoch):

    # Switch to train mode
    model.train()

    video_losses = []
    print("Now commencing epoch {}".format(epoch))
    for i, video in enumerate(train_loader):
        #print(type(video))
        frame_losses = []
        start = datetime.datetime.now().replace(microsecond=0)
        print("Number of clips for video {} : {}".format(i,len(video)))
        state = None # Initially no hidden state
        for j, (clip, gtruths) in enumerate(video):

            # Reset Gradients
            optimizer.zero_grad()

            # Squeeze out the video dimension
            # [video_batch, clip_length, channels, height, width]
            # After transpose:
            # [clip_length, video_batch, channels, height, width]

            clip = Variable(clip.type(dtype).transpose(0,1))
            gtruths = Variable(gtruths.type(dtype).transpose(0,1))

            #print(clip.size()) #works! torch.Size([5, 1, 1, 360, 640])
            loss = 0
            for idx in range(clip.size()[0]):
                #print(clip[idx].size())
                # Compute output
                state, saliency_map = model.forward(clip[idx], state)
                if saliency_map.size() != gtruths[idx].size():
                    #print(saliency_map.size())
                    #print(gtruths[idx].size())
                    a, b, c, _ = saliency_map.size()
                    saliency_map = torch.cat([saliency_map, torch.zeros(a, b, c, 1).cuda()], 3) #because of upsampling we need to concatenate another column of zeroes. The original number is odd so it is impossible for upsampling to get an odd number as it scales by 2

                # Compute loss
                loss = loss + criterion(saliency_map, gtruths[idx])
                # Keep score
                frame_losses.append(loss.data)

                """
                # Accumulate gradients
                if idx == (clip.size()[0]-1):
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
                """

            if j == 0:
                if not os.path.exists("./test"):
                    os.mkdir("./test")
                utils.save_image(clip[idx], "./test/clip{}.png".format(i))
                utils.save_image(gtruths[idx], "./test/gtruths{}.png".format(i))
                utils.save_image(saliency_map, "./test/smap{}.png".format(i))

            # Repackage to avoid backpropagating further through time
            (hidden, cell) = state
            hidden = Variable(hidden.data)
            cell = Variable(cell.data)
            state = (hidden, cell)

            #hidden = Variable(hidden.data)
            #cell = Variable(cell.data)


            # Compute gradient
            loss.backward()

            # Clip gradient to override explosive gradients. Gradients are accumulated so I went for a threshold that depends on clip length.
            nn.utils.clip_grad_norm(model.parameters(), 10*clip.size()[0])

            # Update parameters
            optimizer.step()

            #state = (hidden, cell)
            #hidden = Variable(hidden.data, requires_grad=True)
            #cell = Variable(cell.data, requires_grad=True)

            """
            if (j+1)%20==0:
                print('Training Loss: {} Batch/Clip: {}/{} '.format(loss.data, i, j+1))
            """

        #writer.add_scalar('Train/Loss', mean(frame_losses), i)
        end = datetime.datetime.now().replace(microsecond=0)
        print('Epoch: {}\tVideo: {}\t Training Loss: {}\t Time elapsed: {}\t'.format(epoch, i, mean(frame_losses), end-start))
        video_losses.append(mean(frame_losses))

    return (mean(video_losses))


def validate(val_loader, model, criterion, epoch):

    # switch to evaluate mode
    model.eval()

    video_losses = []
    print("Now running validation..")
    for i, video in enumerate(val_loader):
        frame_losses = []
        state = None # Initially no hidden state
        for j, (clip, gtruths) in enumerate(video):

            clip = Variable(clip.type(dtype).transpose(0,1), requires_grad=False)
            gtruths = Variable(gtruths.type(dtype).transpose(0,1), requires_grad=False)

            for idx in range(clip.size()[0]):
                #print(clip[idx].size()) needs unsqueeze
                # Compute output
                (hidden, cell), saliency_map = model.forward(clip[idx], state)

                hidden = Variable(hidden.data)
                cell = Variable(cell.data)
                state = (hidden, cell)

                if saliency_map.size() != gtruths[idx].size():
                    a, b, c, _ = saliency_map.size()
                    saliency_map = torch.cat([saliency_map, torch.zeros(a, b, c, 1).cuda()], 3) #because of upsampling we need to concatenate another column of zeroes. The original number is odd so it is impossible for upsampling to get an odd number as it scales by 2

                # Compute loss
                loss = criterion(saliency_map, gtruths[idx])
                # Keep score
                frame_losses.append(loss.data)

        video_losses.append(mean(frame_losses))
        #writer.add_scalar('Val/Loss', mean(frame_losses), i)

    return(mean(video_losses))

if __name__ == '__main__':
    main()

    #utils.save_image(saliency_map.data.cpu(), "test.png")


