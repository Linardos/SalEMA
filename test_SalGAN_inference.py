import torch
from SalGANmore import SalGANplus, SalGANmid, SalGAN
from salgan_utils import load_image, postprocess_prediction
import salgan_generator

import os
import cv2

# parameters for demo inference=================================================
PATH_PYTORCH_WEIGHTS = 'model_weights/gen_model.pt'
PATH_SAMPLE_IMAGES = 'sample_images'
PATH_SAMPLE_SALIENCY = 'sample_saliency'
CONV_LSTM_WEIGHTS = './SalConvLSTM.pt'
USE_GPU=False

def main(seed_init):
    """
    Runs pytorch-SalGAN on a sample images

    """
    # create output file
    if not os.path.exists(PATH_SAMPLE_SALIENCY):
        os.makedirs(PATH_SAMPLE_SALIENCY)

    # init model with pre-trained weights
    """
    weights = torch.load(PATH_PYTORCH_WEIGHTS)
    print(type(weights))
    print(weights.keys())
    exit()
    """

    model = SalGANplus(seed_init=seed_init, use_gpu=USE_GPU)
    #model = salgan_generator.create_model()
    model.salgan.load_state_dict(torch.load(PATH_PYTORCH_WEIGHTS), strict=False)
    checkpoint = load_weights(model, CONV_LSTM_WEIGHTS)
    model.Gates.load_state_dict(checkpoint, strict=False)
    model.conv1x1.load_state_dict(checkpoint, strict=False)
    print("Pre-trained model loaded succesfully")
    model.eval()

    # if GPU is enabled
    if USE_GPU:
        model.cuda()

    # load and preprocess images in folder
    for i, name in enumerate(os.listdir(PATH_SAMPLE_IMAGES)):
        filename = os.path.join(PATH_SAMPLE_IMAGES, name)
        image_tensor, image_size = load_image(filename)

        if USE_GPU:
            image_tensor = image_tensor.cuda()

        # run model inference
        prediction = model.forward(image_tensor[None, ...]) # add extra batch dimension

        if type(prediction) is tuple:
            _, prediction = prediction

        # get result to cpu and squeeze dimensions
        if USE_GPU:
            prediction = prediction.squeeze().data.cpu().numpy()
        else:
            prediction = prediction.squeeze().data.numpy()

        """
        # postprocess
        saliency = postprocess_prediction(prediction, image_size)
        """

        # save saliency, name depends on seed
        cv2.imwrite(os.path.join(PATH_SAMPLE_SALIENCY, "{}".format(seed)+name), prediction)
        print("Processed image {}".format(i))

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



if __name__ == '__main__':

    seed = 65
    main(seed)
    print("Done with seed {}".format(seed))
