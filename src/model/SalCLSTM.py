import torch
from torchvision.models import vgg16
from torch import nn
#from torch.nn.functional import interpolate #Upsampling is supposedly deprecated, replace with interpolate, eventually, maybe
from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate
from torch.autograd import Variable
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU

class Upsample(nn.Module):
    # Upsampling has been deprecated for some reason, this workaround allows us to still use the function within sequential.https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588
    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

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


#Includes two variations: SalCLSTM56 and SalCLSTM30, which refer to adding the convLSTM layer in the end or in the middle.

class SalCLSTM56(nn.Module):

    def __init__(self, seed_init, freeze = True, use_gpu=True):
        super(SalCLSTM56,self).__init__()

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


        final_convolutions = [
            Conv2d(self.hidden_size, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            Sigmoid(),
        ]
        self.final_convs = torch.nn.Sequential(*final_convolutions)


        # Initialize weights of ConvLSTM

        torch.manual_seed(seed_init)
        for name, param in self.Gates.named_parameters():
                if "weight" in name:
                    nn.init.xavier_normal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)
                else:
                    print("There is some uninitiallized parameter. Check your parameters and try again.")
                    exit()
        for name, param in self.final_convs.named_parameters():
                if "weight" in name:
                    nn.init.xavier_normal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)
                else:
                    print("There is some uninitiallized parameter. Check your parameters and try again.")
                    exit()

        # Freeze SalGAN
        if freeze:
            for child in self.salgan.children():
                for param in child.parameters():
                    param.requires_grad = False



    def thaw(self, epoch, optimizer):

        """
        A function to gradually unfreeze layers.

        The requires_grad of the corresponding layers is switched to True and then the new parameters are added to the optimizer.

        (https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/3)

        """

        if epoch < 2:
            return optimizer

        elif epoch < 3:
            for child in list(self.salgan.children())[-5:-1]:
                for name, param in child.named_parameters():
                    param.requires_grad = True
                    optimizer.add_param_group({"params": param})

            return optimizer

        elif epoch < 4:
            for child in list(self.salgan.children())[-10:-5]:
                for name, param in child.named_parameters():
                    param.requires_grad = True
                    optimizer.add_param_group({"params": param})

            return optimizer

        elif epoch < 5:
            for child in list(self.salgan.children())[-15:-10]:
                for name, param in child.named_parameters():
                    param.requires_grad = True
                    optimizer.add_param_group({"params": param})

            return optimizer

        elif epoch == 5:
            for child in list(self.salgan.children())[0:-15]:
                for name, param in child.named_parameters():
                    param.requires_grad = True
                    optimizer.add_param_group({"params": param})

            return optimizer

        else:
            return optimizer

    def print_layers(self):

        for child in self.salgan.children():
            print("{}".format(child))
            for name, param in child.named_parameters():
                print("For {} the requires_grad is {}".format(name, param.requires_grad))

        print("ConvLSTM")
        for name, param in self.Gates.named_parameters():
            print("For {} the requires_grad is {}".format(name, param.requires_grad))

        print("Saliency 1x1 Convolution")
        for name, param in self.final_convs.named_parameters():
            print("For {} the requires_grad is {}".format(name, param.requires_grad))



    def forward(self, input_, prev_state=None):

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
        saliency_map = self.final_convs(cell)

        return (hidden, cell), saliency_map




class SalCLSTM30(nn.Module):

    def __init__(self, seed_init, residual, freeze=True, use_gpu=True):
        super(SalCLSTM30,self).__init__()

        self.residual = residual
        self.use_gpu = use_gpu
        # Create encoder based on VGG16 architecture
        original_vgg16 = vgg16()

        # select only convolutional layers
        encoder = torch.nn.Sequential(*list(original_vgg16.features)[:30]) #reduce from 30?
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
        self.salgan = torch.nn.Sequential(*(list(encoder.children())+list(decoder.children())))
        assert(str(encoder)==str(self.salgan[:30]))

        # ConvLSTM
        self.input_size = 512
        self.hidden_size = 512
        self.Gates = nn.Conv2d(in_channels = self.input_size + self.hidden_size, out_channels = 4 * self.hidden_size, kernel_size = (3, 3), padding = 1) #padding 1 to preserve HxW dimensions

        # Initialize weights of ConvLSTM
        torch.manual_seed(seed_init)
        for name, param in self.Gates.named_parameters():
                if "weight" in name:
                    nn.init.xavier_normal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)
                else:
                    print("There is some uninitiallized parameter. Check your parameters and try again.")
                    exit()

        # Freeze SalGAN
        if freeze:
            for child in self.salgan.children():
                for param in child.parameters():
                    param.requires_grad = False


    def thaw(self, epoch, optimizer):

        """
        A function to gradually unfreeze layers.

        The requires_grad of the corresponding layers is switched to True and then the new parameters are added to the optimizer.

        (https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/3)

        """

        if epoch < 2:
            return optimizer

        elif epoch < 3:
            for child in list(self.salgan.children())[-5:-1]:
                for name, param in child.named_parameters():
                    param.requires_grad = True
                    optimizer.add_param_group({"params": param})

            return optimizer

        elif epoch < 4:
            for child in list(self.salgan.children())[-10:-5]:
                for name, param in child.named_parameters():
                    param.requires_grad = True
                    optimizer.add_param_group({"params": param})

            return optimizer

        elif epoch < 5:
            for child in list(self.salgan.children())[-15:-10]:
                for name, param in child.named_parameters():
                    param.requires_grad = True
                    optimizer.add_param_group({"params": param})

            return optimizer

        elif epoch == 5:
            for child in list(self.salgan.children())[0:-15]:
                for name, param in child.named_parameters():
                    param.requires_grad = True
                    optimizer.add_param_group({"params": param})

            return optimizer

        else:
            return optimizer

    def print_layers(self):

        for child in self.salgan.children():
            print("{}".format(child))
            for name, param in child.named_parameters():
                print("For {} the requires_grad is {}".format(name, param.requires_grad))

        print("ConvLSTM")
        for name, param in self.Gates.named_parameters():
            print("For {} the requires_grad is {}".format(name, param.requires_grad))


    def forward(self, input_, prev_state=None):

        x = self.salgan[:30](input_) # Encoder
        residual = x
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
        if self.residual == True:
            x = cell+residual
        else:
            x = cell

        saliency_map = self.salgan[30:](x) # Decoder
        #print(saliency_map.size())
        return (hidden, cell), saliency_map

if __name__ == '__main__':
    model = SalCLSTM30(seed_init=65)
