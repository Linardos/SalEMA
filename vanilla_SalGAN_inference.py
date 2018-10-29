import torch
from SalGANmore import SalGANplus, SalGANmid, SalGAN
from salgan_utils import load_image, postprocess_prediction
import salgan_generator

import os
import cv2

# parameters for demo inference=================================================
PATH_PYTORCH_WEIGHTS = 'model_weights/gen_model.pt'
PATH_SAMPLE_IMAGES = 'sample_images'
PATH_SAMPLE_SALIENCY = 'sample_saliency_2'
USE_GPU=False

def main():
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

    model = SalGANplus(use_gpu=USE_GPU)
    #model = salgan_generator.create_model()
    model.salgan.load_state_dict(torch.load(PATH_PYTORCH_WEIGHTS), strict=False)
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

        # postprocess
        saliency = postprocess_prediction(prediction, image_size)

        # save saliency
        cv2.imwrite(os.path.join(PATH_SAMPLE_SALIENCY, name), saliency)
        print("Processed image {}".format(i))


if __name__ == '__main__':
    main()
