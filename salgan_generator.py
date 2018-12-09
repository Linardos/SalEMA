import torch
from torchvision.models import vgg16
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU


def create_model():
    # Create encoder based on VGG16 architecture
    original_vgg16 = vgg16()

    # select only convolutional layers
    encoder = torch.nn.Sequential(*list(original_vgg16.features)[:30])

    # define decoder based on VGG16 (inverse order and Upsampling layers)
    decoder_list=[
        Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Upsample(scale_factor=2, mode='nearest'),

        Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Upsample(scale_factor=2, mode='nearest'),

        Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Upsample(scale_factor=2, mode='nearest'),

        Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Upsample(scale_factor=2, mode='nearest'),

        Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        Sigmoid(),
    ]

    decoder = torch.nn.Sequential(*decoder_list)

    # assamble the full architecture encoder-decoder
    model = torch.nn.Sequential(*(list(encoder.children())+list(decoder.children())))

    return model
