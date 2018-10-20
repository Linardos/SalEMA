import cv2
import torch
import numpy as np
#from IPython import embed

SALGAN_RESIZE = (192, 256) # H, W

def load_image(path_image, size=None, bgr_mean=[103.939, 116.779, 123.68]):
    """
    Loads and pre-process the image for SalGAN model.

    args:
        path_image: abs path to image
        size: size to input to the network (it not specified, uses SalGAN predifined)
        bgr_mean: mean values (BGR) to extract from images
    returns:
        torch tensor with processed image
        original size of the image
    """
    image = cv2.imread(path_image)

    H, W, C = image.shape
    if size is None:
        size = SALGAN_RESIZE

    image = cv2.resize(image, (size[1], size[0]))
    image = image.astype(np.float32)

    bgr_mean=np.array(bgr_mean)
    image -= bgr_mean

    # convert to torch Tensor
    image = torch.FloatTensor(image)

    # swap channel dimensions
    image = image.permute(2,0,1)

    return image, (H, W)

def postprocess_prediction(prediction, size=None):
    """
    Postprocess saliency maps by resizing and applying gaussian blurringself.

    args:
        prediction: numpy array with saliency postprocess_prediction
        size: original (H,W) of the image
    returns:
        numpy array with saliency map normalized 0-255 (int8)
    """
    saliency_map = (prediction * 255).astype(np.uint8)

    if size is None:
        size = SALGAN_RESIZE

    # resize back to original size
    saliency_map = cv2.resize(saliency_map, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    saliency_map = cv2.GaussianBlur(saliency_map, (5, 5), 0)
    # clip again
    saliency_map = np.clip(saliency_map, 0, 255)

    return saliency_map
