from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.utils.parametrize as P
import torch.nn.functional as F

class KappaLoss(nn.Module):
    def __init__(self):
        super(KappaLoss, self).__init__()
        self.loss = nn.CosineSimilarity(dim=-1)

    def forward(self, inputs, targets, w, beta):
        w_hat = torch.sum(torch.abs(torch.fft.fft(w,dim=1))**2,dim=0)
        B = torch.max(w_hat,dim=0).values
        A = torch.min(w_hat,dim=0).values
        loss = 0.5*torch.mean(1-self.loss(inputs, targets)) + beta*B/A
        return loss

class Student(pl.LightningModule):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.kappa = []
        self.train_outputs = []
        self.cos = nn.CosineSimilarity(dim=-1)
        self.loss = KappaLoss()

    def step(self, batch, baseline, spec, beta):
        feat = batch['feature'].squeeze()
        x = batch['x']
        outputs = self(x)
        w = baseline.psi.weight[:,0,:]
        w = F.pad(w,(0,spec["seg_length"]-spec["win_length"]), value=0)
        loss = 0.5 * (1-self.loss(outputs[:,1:,:], feat[:,1:,:], w, beta).mean())
        loss_cos = 0.5 * (1-self.cos(outputs[:,1:,:], feat[:,1:,:]).mean())
        self.train_outputs.append(loss)
        return {'loss': loss}, 
    
    def on_train_epoch_start(self):
        self.train_outputs = []

    def on_train_epoch_end(self):
        avg_loss = torch.tensor(self.train_outputs).mean()
        self.log('train_loss', avg_loss, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class TDFilterbank_real(torch.nn.Module):
    def __init__(self, spec, w):
        super().__init__()
        
        self.psi = torch.nn.Conv1d(
            in_channels=1,
            out_channels=spec["num_filters"],
            kernel_size=spec["window_size"],
            stride=1,
            padding=0,
            bias=False)

        self.psi.weight.data = w[:, :spec["window_size"]].unsqueeze(1).float()        
    
    def forward(self, x, spec):
        x = x.reshape(x.shape[0], 1, x.shape[-1])
        x = F.pad(x, (0, spec["window_size"]-1), mode='circular',)
        Wx = torch.abs(self.psi(x))
        phi = torch.ones(spec["num_filters"], spec["num_filters"], spec["window_size"])
        Ux = F.conv1d(Wx, phi, bias=None, stride=256, padding=0)
        return Ux