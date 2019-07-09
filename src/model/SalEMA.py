import torch
from torchvision.models import vgg16
from torch import nn, sigmoid
#from torch.nn.functional import interpolate #Upsampling is supposedly deprecated, replace with interpolate, eventually, maybe
from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate, dropout2d
from torch.autograd import Variable
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU

class Upsample(nn.Module):
    # Upsample has been deprecated, this workaround allows us to still use the function within sequential.https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588
    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class SalEMA(nn.Module):
    """
    In this model, we pick a Convolutional layer from the bottleneck and apply EMA as a simple temporal regularizer.
    The smaller the alpha, the less each newly added frame will impact the outcome. This way the temporal information becomes most relevant.
    """
    def  __init__(self, alpha, ema_loc, residual, dropout, use_gpu=True):
        super(SalEMA,self).__init__()

        self.dropout = dropout
        self.residual = residual
        self.use_gpu = use_gpu
        if alpha == None:
            self.alpha = nn.Parameter(torch.Tensor([0.25]))
            print("Initial alpha set to: {}".format(self.alpha))
        else:
            self.alpha = torch.Tensor([alpha])
        assert(self.alpha<=1 and self.alpha>=0)
        self.ema_loc = ema_loc # 30 = bottleneck

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
            Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            Sigmoid(),
        ]

        decoder = torch.nn.Sequential(*decoder_list)

        # assamble the full architecture encoder-decoder
        self.salgan = torch.nn.Sequential(*(list(encoder.children())+list(decoder.children())))

        print("Model initialized, EMA located at {}".format(self.salgan[self.ema_loc]))
        #print(len(self.salgan))

    def forward(self, input_, prev_state=None):
        x = self.salgan[:self.ema_loc](input_)
        residual = x
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]

        if self.dropout == True:
            x = dropout2d(x)
        # salgan[self.ema_loc] will act as the temporal state
        if prev_state is None:
            current_state = self.salgan[self.ema_loc](x) #Initially don't apply alpha as there is no prev state we will consistently have bad saliency maps at the start if we were to do so.
        else:
            current_state = sigmoid(self.alpha)*self.salgan[self.ema_loc](x)+(1-sigmoid(self.alpha))*prev_state
            #current_state = (self.alpha)*self.salgan[self.ema_loc](x)+(1-(self.alpha))*prev_state

        if self.residual == True:
            x = current_state+residual
        else:
            x = current_state

        if self.ema_loc < len(self.salgan)-1:
            x = self.salgan[self.ema_loc+1:](x)

        return current_state, x #x is a saliency map at this point


class SalGAN_EMA2(nn.Module):
    """
    In this model, we pick two Convolutional layers from decoder and encoder  and apply EMA
    The smaller the alpha, the less each newly added frame will impact the outcome. This way the temporal information becomes most relevant.
    """
    def  __init__(self, alpha, ema_loc_1, ema_loc_2, use_gpu=True):
        super(SalGAN_EMA2,self).__init__()

        self.use_gpu = use_gpu
        self.alpha = alpha
        self.ema_loc_1 = ema_loc_1 # 30 = bottleneck
        self.ema_loc_2 = ema_loc_2 # 30 = bottleneck
        assert(self.alpha<=1 and self.alpha>=0)

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
            Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            Sigmoid(),
        ]

        decoder = torch.nn.Sequential(*decoder_list)

        # assamble the full architecture encoder-decoder
        self.salgan = torch.nn.Sequential(*(list(encoder.children())+list(decoder.children())))

        print("Model initialized, EMAs located at {} and {}".format(self.salgan[self.ema_loc_1], self.salgan[self.ema_loc_2]))
        #print(len(self.salgan))

    def forward(self, input_, prev_state_1=None, prev_state_2=None):
        x = self.salgan[:self.ema_loc_1](input_)

        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]

        if prev_state_1 is None:
            x = self.salgan[self.ema_loc_1](x)
        else:
            x = self.alpha*self.salgan[self.ema_loc_1](x)+(1-self.alpha)*prev_state_1

        prev_state_1 = x
        x = self.salgan[self.ema_loc_1+1:self.ema_loc_2](x)

        if prev_state_2 is None:
            x = self.salgan[self.ema_loc_2](x)
        else:
            x = self.alpha*self.salgan[self.ema_loc_2](x)+(1-self.alpha)*prev_state_2

        prev_state_2 = x
        x = self.salgan[self.ema_loc_2+1:](x)

        return (prev_state_1, prev_state_2), x #x is a saliency map at this point

if __name__ == '__main__':
    model = SalEMA(alpha=0.1, ema_loc=7)
    print(model)
