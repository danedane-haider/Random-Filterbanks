from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.utils.parametrize as P
import torch.nn.functional as F
import teacher

# where do the initialization values go?

def weight_init()
    w_init, w_tight = fb.random_filterbank(spec["seg_length"], spec["num_filters"], spec["win_length"], tight=True, to_torch=True, support_only=True)


class KappaLoss(nn.Module):
    def __init__(self):
        super(KappaLoss, self).__init__()
        self.loss = nn.CosineSimilarity(dim=-1)

    def forward(self, inputs, targets, w, beta):
        w_hat = torch.sum(torch.abs(torch.fft.fft(w,dim=1))**2,dim=0)
        B = torch.max(w_hat,dim=0).values
        A = torch.min(w_hat,dim=0).values
        kappa = B/A
        loss = 0.5*torch.mean(1-self.loss(inputs, targets)) + beta*kappa
        return loss, kappa

class Student(pl.LightningModule):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.cond = []
        self.train_outputs = []
        self.cos = nn.CosineSimilarity(dim=-1)
        self.loss = KappaLoss()

    def step(self, batch, baseline, spec, beta):
        feat = batch['feature'].squeeze()
        x = batch['x']
        outputs = self(x)
        w = baseline.psi.weight[:,0,:]
        w = F.pad(w,(0,spec["seg_length"]-spec["win_length"]), value=0)
        loss, kappa = 0.5 * (1-self.loss(outputs[:,1:,:], feat[:,1:,:], w, beta).mean())
        loss_cos = 0.5 * (1-self.cos(outputs[:,1:,:], feat[:,1:,:]).mean())
        self.train_outputs.append(loss)
        self.cond.append(kappa)
        return {'loss': loss, 'loss_cos': loss_cos, 'kappa': kappa}
    
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
        hann = torch.hann_window(spec["window_length"]).unsqueeze(0).unsqueeze(0)
        phi = torch.ones(spec["num_filters"], spec["num_filters"], spec["window_length"])*hann
        Ux = F.conv1d(Wx, phi, bias=None, stride=256, padding=0)
        return Ux