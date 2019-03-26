import torch
from torch import nn
import numpy as np

def accuracy(predictions, targets):
    return torch.sum(predictions.argmax(1) == targets).item() / (targets.shape[0]*targets.shape[1])

def accuracy_w(predictions, targets):
    mask = targets>0
    return torch.sum(predictions.argmax(1)[mask] == targets[mask]).item() / mask.sum().item()

class SpecNet(nn.Module):

    def __init__(self, embed_size=256):
        super(SpecNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(128,1), stride=1,
                                   bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=(1,3), stride=1,
                                   bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, embed_size, kernel_size=(1,3), stride=1,
                                   bias=False),
            nn.BatchNorm2d(embed_size),
            nn.LeakyReLU()
            )
        self.gru = nn.GRU(input_size=embed_size, hidden_size=embed_size)
        self.ce_loss = nn.CrossEntropyLoss()

    def trunk(self, data):
        conv_out = self.convnet(data)
        gru_in = conv_out[:,:,0,:].transpose(1,2).transpose(0,1)
        gru_out = self.gru(gru_in)[0]
        prediction = self.lin(gru_out).transpose(0,1).transpose(1,2)
        return prediction, gru_out

    def loss_p(self, predictions, targets):
        return [
            self.ce_loss(predictions[0][:,:self.n_p,:], targets[1][:,2:-2])
        ]

    def accuracy_p(self, predictions, targets):
        batch_size = targets[1].shape[0]
        return [
            accuracy(predictions[0][:,:self.n_p,:], targets[1][:,2:-2])
        ]

    def loss_s(self, predictions, targets):
        targets_pid = targets[0][:,None].repeat(1,predictions[0].shape[2])
        return [
            self.ce_loss(predictions[0][:,:self.n_s,:], targets_pid)
        ]

    def accuracy_s(self, predictions, targets):
        return [
            accuracy(predictions[0][:,:self.n_s,:], targets[0][:,None].repeat(1,predictions[0].shape[2]))
        ]

    def loss_ps(self, predictions, targets):
        targets_pid = targets[0][:,None].repeat(1,predictions[0].shape[2])
        return [
            self.ce_loss(predictions[0][:,:self.n_p,:], targets[1][:,2:-2]),
            self.ce_loss(predictions[0][:,-self.n_s:,:], targets_pid)*self.loss_factor_s
        ]

    def accuracy_ps(self, predictions, targets):
        return [
            accuracy(predictions[0][:,:self.n_p,:], targets[1][:,2:-2]),
            accuracy(predictions[0][:,-self.n_s:,:], targets[0][:,None].repeat(1,predictions[0].shape[2]))
        ]

    def loss_pw(self, predictions, targets):
        return [
            self.ce_loss(predictions[0][:,:self.n_p,:], targets[1][:,2:-2]),
            self.ce_loss_w(predictions[1][:,:self.n_w,:], targets[4][:,2:-2])*self.loss_factor_w
        ]

    def accuracy_pw(self, predictions, targets):
        return [
            accuracy(predictions[0][:,:self.n_p,:], targets[1][:,2:-2]),
            accuracy_w(predictions[1][:,:self.n_w,:], targets[4][:,2:-2])
        ]

    def loss_pws(self, predictions, targets):
        targets_pid = targets[0][:,None].repeat(1,predictions[0].shape[2])
        return [
            self.ce_loss(predictions[0][:,:self.n_p,:], targets[1][:,2:-2]),
            self.ce_loss(predictions[0][:,self.n_p:self.n_p*2,:], targets[2][:,2:-2])/2,
            self.ce_loss(predictions[0][:,self.n_p*2:self.n_p*3,:], targets[3][:,2:-2])/3,
            self.ce_loss_w(predictions[1][:,:self.n_w,:], targets[4][:,2:-2])*self.loss_factor_w,
            self.ce_loss_w(predictions[1][:,self.n_w:,:], targets[5][:,2:-2])*self.loss_factor_w/2,
            self.ce_loss(predictions[0][:,-self.n_s:,:], targets_pid)*self.loss_factor_s
        ]

    def accuracy_pws(self, predictions, targets):
        batch_size = targets[1].shape[0]
        targets_pid = targets[0][:,None].repeat(1,predictions[0].shape[2])
        return [
            torch.sum(predictions[0][:,:self.n_p,:].argmax(1) == targets[1][:,2:-2]).item() / (batch_size*(targets[1].shape[1]-4)),
            torch.sum(predictions[0][:,self.n_p:self.n_p*2,:].argmax(1) == targets[2][:,2:-2]).item() / (batch_size*(targets[2].shape[1]-4)),
            torch.sum(predictions[0][:,self.n_p*2:self.n_p*3,:].argmax(1) == targets[3][:,2:-2]).item() / (batch_size*(targets[3].shape[1]-4)),
            torch.sum(predictions[1][:,:self.n_w,:].argmax(1) == targets[4][:,2:-2]).item() / (batch_size*(targets[4].shape[1]-4)),
            torch.sum(predictions[1][:,self.n_w:,:].argmax(1) == targets[5][:,2:-2]).item() / (batch_size*(targets[5].shape[1]-4)),
            torch.sum(predictions[0][:,-self.n_s:,:].argmax(1) == targets_pid).item() / (batch_size*(targets_pid.shape[1]))
        ]

class SpecNetP(SpecNet):

    def __init__(self, embed_size=256, n_p=41, n_future_p=1, n_s=0):
        super(SpecNetP, self).__init__(embed_size)
        self.lin = nn.Linear(embed_size, n_p*n_future_p+n_s)
        if n_s>0:
            if n_future_p==0:
                self.loss = self.loss_s
                self.accuracy = self.accuracy_s
            else:
                self.loss = self.loss_ps
                self.accuracy = self.accuracy_ps
            self.n_losses = n_future_p+1
            self.loss_factor_s = -np.log(1/n_p)/-np.log(1/n_s)
        else:
            self.loss = self.loss_p
            self.accuracy = self.accuracy_p
            self.n_losses = n_future_p
        self.n_p = n_p
        self.n_s = n_s

    def forward(self, data):
        return self.trunk(data)[0], None

class SpecNetPW(SpecNetP):

    def __init__(self, embed_size=256, n_p=41, n_future_p=1, n_w=7727, n_future_w=1, n_s=0):
        super(SpecNetPW, self).__init__(embed_size, n_p, n_future_p, n_s)
        self.gru_w = nn.GRU(input_size=embed_size, hidden_size=embed_size)
        self.lin_w = nn.Linear(embed_size, n_w*n_future_w)
        self.loss_factor_w = -np.log(1/n_p)/-np.log(1/n_w)
        if n_s>0:
            self.loss = self.loss_pws
            self.accuracy = self.accuracy_pws
            self.n_losses = n_future_p+n_future_w+1
        else:
            self.loss = self.loss_pw
            self.accuracy = self.accuracy_pw
            self.n_losses = n_future_p+n_future_w+1
        self.n_p = n_p
        self.n_w = n_w
        self.n_s = n_s
        self.ce_loss_w = nn.CrossEntropyLoss()

    def forward(self, data):
        prediction, gru_out = self.trunk(data)
        gru_out_w = self.gru_w(gru_out)[0]
        prediction_w = self.lin_w(gru_out_w).transpose(0,1).transpose(1,2)
        return prediction, prediction_w
