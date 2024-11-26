import math

import torch
import torch.nn as nn

import torch.nn.functional as F

import random


class classifier(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        seq_len=31
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.lin0 = nn.Linear(in_features, 256)
        self.lin1 = nn.Linear(in_features, out_features)
        self.lin2 = nn.Linear(128, out_features)
        
        self.batch_norm = nn.BatchNorm1d(in_features)

        self.lstm = nn.LSTM(256, 64, 1, batch_first=True, bidirectional=True)

        self.hsl = seq_len//2
        self.sw = 5


    def forward_linear(self, x):

        x = self.lin1(x)

        return x[:, self.hsl-self.sw:self.hsl+self.sw+1, :].mean(dim=1)
    
    def forward_lstm(self, x):

        logits = self.lstm(x)[0][:, self.hsl-self.sw:self.hsl+self.sw+1,:].mean(dim=1)
        
        x = self.lin2(logits)

        return x, logits
    
    def forward(self, x):
        
        x = self.batch_norm(x.permute(0,2,1)).permute(0,2,1)

        amount = random.randint(64, 256)

        rand_inds = torch.randperm(x.size(2))[:amount]
        
        x[:, :, rand_inds] = torch.randn_like(x[:, :, rand_inds]).to(x.device)
        
        linear_logits = self.forward_linear(x)

        x = self.lin0(x)

        x = x - x.mean(dim=1, keepdim=True)

        lstm_logits, rawm = self.forward_lstm(x)

        return lstm_logits, linear_logits, rawm
    
    def forward_nodrop(self, x):
        
        x = self.batch_norm(x.permute(0,2,1)).permute(0,2,1)
        
        linear_logits = self.forward_linear(x)

        x = self.lin0(x)

        x = x - x.mean(dim=1, keepdim=True)

        lstm_logits, raw = self.forward_lstm(x)

        return lstm_logits + linear_logits
    
    
    
