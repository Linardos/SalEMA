import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable

#Not using peephole connections yet
class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, use_gpu, input_size, hidden_size, kernel_size):
        super(ConvLSTMCell,self).__init__()
        self.use_gpu = use_gpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(in_channels = input_size + hidden_size, out_channels = 4 * hidden_size, kernel_size = kernel_size, padding = 1) #padding 1 to preserve HxW dimensions
        self.conv1x1 = nn.Conv2d(in_channels = hidden_size, out_channels = 1, kernel_size = 1)

    def forward(self, input_, prev_state=None):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

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
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        #print("stacked input size {}".format(stacked_inputs.size()))
        gates = self.Gates(stacked_inputs)
        #print("stacked gates size {}".format(gates.size()))

        # chunk across channel dimension
        in_gate, forget_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        forget_gate = f.sigmoid(forget_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)
        # compute current cell and hidden state

        #print("forget gate size {}".format(forget_gate.size()))
        #print("in gate size {}".format(in_gate.size()))
        #print("cell gate size {}".format(cell_gate.size()))
        #print("previous cell size {}".format(prev_cell.size()))
        forget = (forget_gate * prev_cell)
        update = (in_gate * cell_gate)
        cell = forget + update
        hidden = out_gate * f.tanh(cell)

        state = [hidden,cell]
        saliency_map = self.conv1x1(cell)

        return (hidden, cell), saliency_map



class ConvLSTM(nn.Module):
    """docstring for ConvLSTM"""
    def __init__(self, use_gpu, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()
        self.use_gpu = use_gpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        self.cell=ConvLSTMCell(self.use_gpu, self.input_size, self.hidden_size, self.kernel_size)

    def forward(self, input_, hidden_state):

        seq_len=current_input.size(0)

        all_output = []
        output_inner = []
        saliency_maps = []
        for t in range(seq_len):#loop for every step
            hidden_cell, smap = self.cell(current_input[t],hidden_state)
            hidden_state, cell_state = hidden_cell
            output_inner.append(hidden_state)
            saliency_maps.append(smap)

        return output_inner, saliency_maps

#For ablation study
class Conv(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, use_gpu, input_size, num_filters, kernel_size):
        super(Conv,self).__init__()
        self.use_gpu = use_gpu
        self.input_size = input_size
        self.conv = nn.Conv2d(in_channels = input_size, out_channels = num_filters, kernel_size = kernel_size, padding = 1) #padding 1 to preserve HxW dimensions
        self.conv1x1 = nn.Conv2d(in_channels = num_filters, out_channels = 1, kernel_size = 1)

    def forward(self, input_, prev_state=None):

        x = self.conv(input_)
        saliency_map = self.conv1x1(x)

        return saliency_map


