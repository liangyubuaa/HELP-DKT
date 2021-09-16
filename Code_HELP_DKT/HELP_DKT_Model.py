import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings('ignore')


class HELP_DKT_Model(nn.Module):
    """HELP-DKT model"""

    def __init__(self, rnn_type, args, num_skills, timeSteps, dropout=0.6, tie_weights=False):
        super(HELP_DKT_Model, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = getattr(nn, rnn_type)(
            args.input_size, args.hidden_size, args.hidden_layer_num, batch_first=True, dropout=dropout)
        if args.taskModel == 'taskA':
            self.decoder = nn.Linear(args.hidden_size*timeSteps, 1)
        elif args.taskModel == 'taskB':
            self.decoder = nn.Linear(args.hidden_size, 1)
        elif args.taskModel == 'taskC':
            self.decoder1 = nn.Sequential(
                nn.Linear(args.hidden_size, args.Qmatrix_size),
                nn.Sigmoid()
            )
            self.decoder2 = nn.Sequential(
                nn.Sigmoid()
            )

        self.rnn_type = rnn_type
        self.nhid = args.hidden_size
        self.nlayers = args.hidden_layer_num
        self.taskModel = args.taskModel
        self.timeSteps = timeSteps
        self.multiLinearLayers = args.multiLinearLayers
        self.masked = args.masked
        self.QmatrixSize = args.Qmatrix_size
        self.subQmatrix = args.subQmatrix
        self.linearWithQmatrix = args.linearWithQmatrix

        self.init_weights()

    def init_weights(self):
        if self.multiLinearLayers == 'False':
            initrange = 0.05
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)
        else:
            if self.taskModel == 'taskA' or self.taskModel == 'taskB':
                for name, param in self.decoder.named_parameters():
                    if 'weight' in name:
                        initrange = 0.05
                        nn.init.uniform_(param, -initrange, initrange)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            else:
                for name, param in self.decoder1.named_parameters():
                    if 'weight' in name:
                        initrange = 0.05
                        nn.init.uniform_(param, -initrange, initrange)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                for name, param in self.decoder2.named_parameters():
                    if 'weight' in name:
                        initrange = 0.05
                        nn.init.uniform_(param, -initrange, initrange)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, input, hidden, Qmatrix, problemQmatrixMask,
                problemQmatrixSub, problemQmatrixAbilityMask, problemQmatrixProd, test=False):
        '''
        @description: run model
        @demands:
        @params:
        @return:
        '''

        output, hidden = self.rnn(input, hidden)
        if self.multiLinearLayers == 'False':
            if self.taskModel == 'taskA':
                decoded = self.decoder(
                    output.reshape(-1, self.timeSteps*self.nhid))
            elif self.taskModel == 'taskB':
                decoded = self.decoder(output)
            else:
                raise ValueError('model forward ERROR!')
        else:  # two layers
            decoded1 = self.decoder1(output)
            ability = decoded1
            if self.masked == 'True':  # &
                decoded1 = torch.mul(decoded1, problemQmatrixAbilityMask)

            if self.subQmatrix == 'True':
                decoded1 = torch.sub(decoded1, problemQmatrixSub)

            decoded = torch.prod(
                self.decoder2(torch.mul(decoded1, problemQmatrixProd)), dim=2, keepdim=True)

        if test == False:
            return decoded, hidden
        else:
            tmp = torch.ones_like(problemQmatrixAbilityMask)
            for i in range(problemQmatrixAbilityMask.size()[0]):
                for j in range(problemQmatrixAbilityMask.size()[1]):
                    if 1 not in problemQmatrixAbilityMask[i, j, :]:
                        tmp[i, j, :] = 0
            if self.multiLinearLayers == 'True':
                return decoded, hidden, ability.mul(tmp).tolist()
            else:
                return decoded, hidden, tmp.tolist()

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
