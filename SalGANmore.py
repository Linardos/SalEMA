import torch
from torchvision.models import vgg16
from torch import nn
from torch.nn.functional import interpolate #Upsampling is supposedly deprecated, replace with interpolate, eventually, maybe
from torch.autograd import Variable
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU
from clstm import ConvLSTM


class SalGAN(nn.Module):
    def  __init__(self):
        super(SalGAN,self).__init__()
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

    def forward(self, input_):
        return self.salgan(input_)


#Includes two variations: SalGANplus and SalGANmid, which refer to adding the convLSTM layer in the end or in the middle.

class SalGANplus(nn.Module):

    def __init__(self, use_gpu=True):
        super(SalGANplus,self).__init__()

        self.use_gpu = use_gpu

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

            #During Upsampling operation we may end up losing 1 dimension if it was an odd number before

        ]

        decoder = torch.nn.Sequential(*decoder_list)
        # assamble the full architecture encoder-decoder
        self.salgan = torch.nn.Sequential(*(list(encoder.children())+list(decoder.children())))
        #print(self.salgan)
        # ConvLSTM
        self.input_size = 128
        self.hidden_size = 128
        self.Gates = nn.Conv2d(in_channels = self.input_size + self.hidden_size, out_channels = 4 * self.hidden_size, kernel_size = (3, 3), padding = 1) #padding 1 to preserve HxW dimensions
        self.conv1x1 = nn.Conv2d(in_channels = self.hidden_size, out_channels = 1, kernel_size = 1)
        self.sigmoid = Sigmoid()

        # Initialize weights of ConvLSTM

        for param in self.Gates.parameters():
            nn.init.normal_(param)
        for param in self.conv1x1.parameters():
            nn.init.normal_(param)


    def forward(self, input_, prev_state=None):

        #print(input_.size())
        x = self.salgan(input_)
        #print(x.size())
        # get batch and spatial sizes
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                prev_state = (
                    Variable(torch.zeros(state_size)).cuda(),
                    Variable(torch.zeros(state_size)).cuda()
                )
            else:
                prev_state = (
                    Variable(torch.zeros(state_size)),
                    Variable(torch.zeros(state_size))
                )


        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]

        #print(prev_hidden.size())
        stacked_inputs = torch.cat((x, prev_hidden), 1)
        #print("stacked input size {}".format(stacked_inputs.size()))
        gates = self.Gates(stacked_inputs)
        #print("stacked gates size {}".format(gates.size()))

        # chunk across channel dimension
        in_gate, forget_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)
        # compute current cell and hidden state

        #print("forget gate size {}".format(forget_gate.size()))
        #print("in gate size {}".format(in_gate.size()))
        #print("cell gate size {}".format(cell_gate.size()))
        #print("previous cell size {}".format(prev_cell.size()))
        forget = (forget_gate * prev_cell)
        update = (in_gate * cell_gate)
        cell = forget + update
        hidden = out_gate * torch.tanh(cell)

        state = [hidden,cell]
        saliency_map = self.sigmoid(self.conv1x1(cell))

        return (hidden, cell), saliency_map




class SalGANmid(nn.Module):

    def __init__(self, use_gpu=True):
        super(SalGANmid,self).__init__()

        self.use_gpu = use_gpu
        # Create encoder based on VGG16 architecture
        original_vgg16 = vgg16()

        # select only convolutional layers
        self.salganEncoder = torch.nn.Sequential(*list(original_vgg16.features)[:30]) #reduce from 30?
        #print(self.salganEncoder)
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
        self.salganDecoder = torch.nn.Sequential(*(list(decoder.children())))
        #print(self.salgan)
        # ConvLSTM
        self.input_size = 512
        self.hidden_size = 512
        self.Gates = nn.Conv2d(in_channels = self.input_size + self.hidden_size, out_channels = 4 * self.hidden_size, kernel_size = (3, 3), padding = 1) #padding 1 to preserve HxW dimensions


        for param in self.Gates.parameters():
            nn.init.normal_(param)
        for param in self.conv1x1.parameters():
            nn.init.normal_(param)


    def forward(self, input_, prev_state=None):

        x = self.salganEncoder(input_)
        #y = self.salganDecoder(x)
        #print("Without ConvLSTM output: {}".format(y.size())) #Verdict: ConvLSTM irrelevant to size discrepancy
        # get batch and spatial sizes
        #print("Before ConvLSTM {}".format(x.size()))
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                prev_state = (
                    Variable(torch.zeros(state_size)).cuda(),
                    Variable(torch.zeros(state_size)).cuda()
                )
            else:
                prev_state = (
                    Variable(torch.zeros(state_size)),
                    Variable(torch.zeros(state_size))
                )


        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]

        #print(prev_hidden.size())
        stacked_inputs = torch.cat((x, prev_hidden), 1)
        #print("stacked input size {}".format(stacked_inputs.size()))
        gates = self.Gates(stacked_inputs)
        #print("stacked gates size {}".format(gates.size()))

        # chunk across channel dimension
        in_gate, forget_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)
        # compute current cell and hidden state

        #print("forget gate size {}".format(forget_gate.size()))
        #print("in gate size {}".format(in_gate.size()))
        #print("cell gate size {}".format(cell_gate.size()))
        #print("previous cell size {}".format(prev_cell.size()))
        forget = (forget_gate * prev_cell)
        update = (in_gate * cell_gate)
        cell = forget + update
        hidden = out_gate * torch.tanh(cell)

        state = [hidden,cell]

        #print("After ConvLSTM {}".format(cell.size()))
        saliency_map = self.salganDecoder(cell)
        #print(saliency_map.size())
        return (hidden, cell), saliency_map
