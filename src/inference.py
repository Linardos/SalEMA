import cv2
import os
import datetime
import numpy as np
from model import SalCLSTM, SalEMA
from args import get_inference_parser
import pickle
import torch
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils import data
from torch.autograd import Variable
from data_loader import DHF1K_frames, Hollywood_frames, DAVIS_frames


"""
Before inferring check:
EMA_LOC,
args.residual,
pt_model,
dst
"""
"""
dataset_name = "UCF-sports"
dataset_name = "Hollywood-2"
dataset_name = "DHF1K"
"""
CLIP_LENGTH = 10
EMA_LOC = 30 # 30 is the bottleneck
#pt_model = './SalEMA{}Afinal.pt'.format(EMA_LOC)
#pt_model = 'SalEMA{}&{}.pt'.format(EMA_LOC,EMA_LOC_2)
frame_size = (192, 256)
# Destination for predictions:

#dst = "/imatge/lpanagiotis/work/{}/SG_predictions".format(dataset_name)
frames_path = "/imatge/lpanagiotis/work/DHF1K/frames"
gt_path = "/imatge/lpanagiotis/work/DHF1K/maps"

params = {'batch_size': 1,
          'num_workers': 4,
          'pin_memory': True}

def main(args):

    if args.dataset == "Hollywood-2" or args.dataset == "UCF-sports":
        dst = os.path.join(args.dst, "{}/testing".format(args.dataset)) #Hollywood or UCF-sports
    else:
        dst = os.path.join(args.dst, "{}_predictions".format(args.pt_model.replace(".pt", "")))
    print("Output directory {}".format(dst))

    # =================================================
    # ================ Data Loading ===================

    #Expect Error if either validation size or train size is 1
    if args.dataset == "DHF1K":
        #print(args.start)
        #print(args.end)
        print("Commencing inference for dataset {}".format(args.dataset))
        dataset = DHF1K_frames(
            root_path = args.src,
            load_gt = False,
            starting_video = int(args.start),
            number_of_videos = int(args.end),
            clip_length = CLIP_LENGTH,
            split = None,
            resolution = frame_size)
             #add a parameter node = training or validation

    elif args.dataset == "Hollywood-2" or args.dataset == "UCF-sports":
        print("Commencing inference for dataset {}".format(args.dataset))
        dataset = Hollywood_frames(
            root_path = args.src,
            clip_length = CLIP_LENGTH,
            resolution = frame_size)
        video_name_list = dataset.video_names() #match an index to the sample video name
    elif args.dataset == "DAVIS" or args.dataset == "other":
        print("Commencing inference for dataset {}".format(args.dataset))
        dataset = DAVIS_frames(
            root_path = args.src,
            clip_length = CLIP_LENGTH,
            resolution = frame_size)
        video_name_list = dataset.video_names() #match an index to the sample video name


    print("Size of test set is {}".format(len(dataset)))

    loader = data.DataLoader(dataset, **params)

    # =================================================
    # ================= Load Model ====================

    # Using same kernel size as they do in the DHF1K paper
    # Amaia uses default hidden size 128
    # input size is 1 since we have grayscale images
    if 'SalCLSTM30' in args.pt_model:

        model = SalCLSTM.SalCLSTM30(seed_init=65, freeze=False, residual=False)

        load_model(args.pt_model, model)
        print("Pre-trained model SalCLSTM30 loaded succesfully")

        TEMPORAL = True

    elif 'SalGAN' in args.pt_model:

        model = SalCLSTM.SalGAN()

        load_model(args.pt_model, model)
        print("Pre-trained model tuned SalGAN loaded succesfully")

        TEMPORAL = False

    elif "EMA" in args.pt_model:
        if args.double_ema:
            model = SalEMA.SalEMA2(alpha=args.alpha, ema_loc_1=EMA_LOC, ema_loc_2=EMA_LOC_2)
        else:
            model = SalEMA.SalEMA(alpha=args.alpha, residual=args.residual, dropout = args.dropout, ema_loc=EMA_LOC)

        load_model(args.pt_model, model)
        print("Pre-trained model {} loaded succesfully".format(args.pt_model))
        if args.residual:
            print("Residual connection is included.")

        TEMPORAL = True
        print("Alpha tuned to {}".format(model.alpha))

    elif args.pt_model == 'model_weights/salgan_salicon.pt':

        if EMA_LOC == None:
            model = SalCLSTM.SalGAN()
            TEMPORAL = False
            print("Pre-trained model SalBCE loaded succesfully.")
        else:
            model = SalEMA.SalEMA(alpha=args.alpha, ema_loc=EMA_LOC)
            TEMPORAL = True
            print("Pre-trained model SalBCE loaded succesfully. EMA inference will commence soon.")

        model.salgan.load_state_dict(torch.load(args.pt_model)['state_dict'])


    elif args.pt_model == '/imatge/lpanagiotis/work/SalCLSTM/src/model_weights/gen_model.pt':
        model = SalCLSTM.SalGAN()
        model.salgan.load_state_dict(torch.load(args.pt_model))
        print("Pre-trained model vanilla SalGAN loaded succesfully")

        TEMPORAL = False
    else:
        print("Your model was not recognized, check the name of the model and try again.")
        exit()

    #model = nn.DataParallel(model).cuda()
    dtype = torch.FloatTensor
    if args.use_gpu:
        assert torch.cuda.is_available(), \
            "CUDA is not available in your machine"
        cudnn.benchmark = False #Would cause overhead during inference.
        model = model.cuda()
        dtype = torch.cuda.FloatTensor
    # ==================================================
    # ================== Inference =====================

    if not os.path.exists(dst):
        os.mkdir(dst)
    else:
        print("Be warned, you are about to write on an existing folder {}. If this is not intentional cancel now.".format(dst))

    # switch to evaluate mode
    model.eval()

    for i, video in enumerate(loader):

        count = 0
        state = None # Initially no hidden state

        if args.dataset == "DHF1K":

            video_dst = os.path.join(dst, str(int(args.start)+i).zfill(4))
            if not os.path.exists(video_dst):
                os.mkdir(video_dst)

            for j, (clip, _) in enumerate(video):
                clip = Variable(clip.type(dtype).transpose(0,1), requires_grad=False)
                if args.double_ema:
                    if state == None:
                        state = (None, None)
                    for idx in range(clip.size()[0]):
                        # Compute output
                        state, saliency_map = model.forward(input_ = clip[idx], prev_state_1 = state[0], prev_state_2 = state[1])

                        saliency_map = saliency_map.squeeze(0) # Target is 3 dimensional (grayscale image)

                        post_process_saliency_map = (saliency_map-torch.min(saliency_map))/(torch.max(saliency_map)-torch.min(saliency_map))
                        utils.save_image(post_process_saliency_map, os.path.join(video_dst, "{}.png".format(str(count).zfill(4))))

                else:
                    for idx in range(clip.size()[0]):
                        # Compute output
                        if TEMPORAL:
                            #import time
                            #start = time.time()
                            state, saliency_map = model.forward(input_ = clip[idx], prev_state = state)
                            #print("Inference time of 1 frame is: {}".format(start-time.time()))
                            #exit()
                        else:
                            saliency_map = model.forward(input_ = clip[idx])

                        count+=1
                        saliency_map = saliency_map.squeeze(0)

                        post_process_saliency_map = (saliency_map-torch.min(saliency_map))/(torch.max(saliency_map)-torch.min(saliency_map))
                        utils.save_image(post_process_saliency_map, os.path.join(video_dst, "{}.png".format(str(count).zfill(4))))

                if TEMPORAL:
                    state = repackage_hidden(state)
            print("Video {} done".format(i+int((args.start))))

        elif args.dataset == "Hollywood-2" or args.dataset == "UCF-sports":

            video_dst = os.path.join(dst, video_name_list[i], '{}_predictions'.format(args.pt_model.replace(".pt", "")))
            print("Destination: {}".format(video_dst))
            if not os.path.exists(video_dst):
                os.mkdir(video_dst)

            for j, (clip, _) in enumerate(video):
                clip = Variable(clip.type(dtype).transpose(0,1), requires_grad=False)
                for idx in range(clip.size()[0]):
                    # Compute output
                    if TEMPORAL:
                        state, saliency_map = model.forward(input_ = clip[idx], prev_state = state)
                    else:
                        saliency_map = model.forward(input_ = clip[idx])

                    count+=1
                    saliency_map = saliency_map.squeeze(0)

                    post_process_saliency_map = (saliency_map-torch.min(saliency_map))/(torch.max(saliency_map)-torch.min(saliency_map))
                    utils.save_image(post_process_saliency_map, os.path.join(video_dst, "{}{}.png".format(video_name_list[i][:-1], str(count).zfill(5))))
                    if count == 1:
                        print("The final destination is {}. Cancel now if this is incorrect".format(os.path.join(video_dst, "{}{}.png".format(video_name_list[i][:-1], str(count).zfill(5)))))

                if TEMPORAL:
                    state = repackage_hidden(state)
            print("Video {} done".format(i+int(args.start)))

        elif args.dataset == "DAVIS" or args.dataset == "other":

            video_dst = os.path.join(dst, video_name_list[i])
            # if "shooting" in video_dst:
            #     # CUDA error: out of memory is encountered whenever inference reaches that vid.
            #     continue
            print("Destination: {}".format(video_dst))
            if not os.path.exists(video_dst):
                os.mkdir(video_dst)

            for j, (clip, _) in enumerate(video):
                clip = Variable(clip.type(dtype).transpose(0,1), requires_grad=False)

                for idx in range(clip.size()[0]):
                    # Compute output
                    if TEMPORAL:
                        state, saliency_map = model.forward(input_ = clip[idx], prev_state = state)
                    else:
                        saliency_map = model.forward(input_ = clip[idx])

                    count+=1
                    saliency_map = saliency_map.squeeze(0)

                    post_process_saliency_map = (saliency_map-torch.min(saliency_map))/(torch.max(saliency_map)-torch.min(saliency_map))
                    utils.save_image(post_process_saliency_map, os.path.join(video_dst, "{}.jpg".format(str(count).zfill(5))))
                    if count == 1:
                        print("The final destination is {}. Cancel now if this is incorrect".format(os.path.join(video_dst, "{}.jpg".format(str(count).zfill(5)))))

                if TEMPORAL:
                    state = repackage_hidden(state)
            print("Video {} done".format(i+int(args.start)))

def load_model(pt_model, new_model):

    temp = torch.load(pt_model)['state_dict']
    # Because of dataparallel there is contradiction in the name of the keys so we need to remove part of the string in the keys:.
    from collections import OrderedDict
    checkpoint = OrderedDict()
    for key in temp.keys():
        new_key = key.replace("module.","")
        checkpoint[new_key]=temp[key]

    new_model.load_state_dict(checkpoint, strict=True)

    return new_model

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

if __name__ == '__main__':
    parser = get_inference_parser()
    args = parser.parse_args()
    main(args)
