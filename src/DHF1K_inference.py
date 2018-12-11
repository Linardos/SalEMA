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
from data_loader import DHF1K_frames

dtype = torch.FloatTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor

clip_length = 10 #with 10 clips the loss seems to reach zero very fast
number_of_videos = 700 # DHF1K offers 700 labeled videos, the other 300 are held back by the authors
#pretrained_model = './SalGANplus.pt'
#pretrained_model = '/imatge/lpanagiotis/work/SalGANmore/model_weights/gen_model.pt' # Vanilla SalGAN
pretrained_model = './SalGAN.pt'
frame_size = (192, 256)

#dst = "/imatge/lpanagiotis/work/DHF1K/SGplus_predictions"
#dst = "/imatge/lpanagiotis/work/DHF1K/SG_predictions"
dst = "/imatge/lpanagiotis/work/DHF1K/SGtuned_predictions"
# Parameters
params = {'batch_size': 1, # number of videos / batch, I need to implement padding if I want to do more than 1, but with DataParallel it's quite messy
          'num_workers': 4,
          'pin_memory': True}

def main():

    # =================================================
    # ================ Data Loading ===================

    #Expect Error if either validation size or train size is 1
    dataset = DHF1K_frames(
        number_of_videos = number_of_videos,
        clip_length = clip_length,
        split = None,
        resolution = frame_size)
         #add a parameter node = training or validation
    print("Size of test set is {}".format(len(dataset)))

    loader = data.DataLoader(dataset, **params)

    # =================================================
    # ================= Load Model ====================

    # Using same kernel size as they do in the DHF1K paper
    # Amaia uses default hidden size 128
    # input size is 1 since we have grayscale images
    if pretrained_model == './SalGANplus.pt':

        model = SalGANmore.SalGANplus(seed_init=65, freeze=False)

        temp = torch.load(pretrained_model)['state_dict']
        # Because of dataparallel there is contradiction in the name of the keys so we need to remove part of the string in the keys:.
        from collections import OrderedDict
        checkpoint = OrderedDict()
        for key in temp.keys():
            new_key = key.replace("module.","")
            checkpoint[new_key]=temp[key]

        model.load_state_dict(checkpoint, strict=True)
        print("Pre-trained model SalGANplus loaded succesfully")

        TEMPORAL = True

    elif pretrained_model == './SalGAN.pt':

        model = SalGANmore.SalGAN()

        temp = torch.load(pretrained_model)['state_dict']
        # Because of dataparallel there is contradiction in the name of the keys so we need to remove part of the string in the keys:.
        from collections import OrderedDict
        checkpoint = OrderedDict()
        for key in temp.keys():
            new_key = key.replace("module.","")
            checkpoint[new_key]=temp[key]

        model.load_state_dict(checkpoint, strict=True)
        print("Pre-trained model tuned SalGAN loaded succesfully")

        TEMPORAL = False

    else:
        model = SalGANmore.SalGAN()
        model.salgan.load_state_dict(torch.load(pretrained_model))
        print("Pre-trained model vanilla SalGAN loaded succesfully")

        TEMPORAL = False

    model = nn.DataParallel(model).cuda()
    cudnn.benchmark = True #https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    #model = model.cuda()
    # ==================================================
    # ================== Inference =====================

    if not os.path.exists(dst):
        os.mkdir(dst)
    else:
        print("Be warned, you are about to write on an existing folder {}. If this is not intentional cancel now.".format(dst))

    # switch to evaluate mode
    model.eval()

    for i, video in enumerate(loader):
        video_dst = os.path.join(dst, str(i+1))
        if not os.path.exists(video_dst):
            os.mkdir(video_dst)

        count = 0
        state = None # Initially no hidden state
        for j, (clip, gtruths) in enumerate(video):
            clip = Variable(clip.type(dtype).transpose(0,1), requires_grad=False)
            gtruths = Variable(gtruths.type(dtype).transpose(0,1), requires_grad=False)
            for idx in range(clip.size()[0]):
                # Compute output
                if TEMPORAL:
                    state, saliency_map = model.forward(input_ = clip[idx], prev_state = state)
                else:
                    saliency_map = model.forward(input_ = clip[idx])

                count+=1
                saliency_map = saliency_map.squeeze(0)

                post_process_saliency_map = (saliency_map-torch.min(saliency_map))/(torch.max(saliency_map)-torch.min(saliency_map))
                utils.save_image(post_process_saliency_map, os.path.join(video_dst, "{}.png".format(str(count))))

            if TEMPORAL:
                state = repackage_hidden(state)

        print("Video {} done".format(i+1))

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

if __name__ == '__main__':
    main()
