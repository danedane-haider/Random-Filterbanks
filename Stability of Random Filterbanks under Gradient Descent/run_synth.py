import numpy as np
import scipy.signal
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torchaudio
import itertools
import torch.nn.functional as F
import scipy as sp
import torch.nn as nn
import fb
import pickle

HYPERPARAMS = {
    "synth": {
        "N": 4092,
        "J": 58,
        "T": 1024,
        "sr": 16000,
        "fmin": 45,
        "fmax": 8000,
        "batch_size": 64
    }
}

# some parameters 

spec = HYPERPARAMS["synth"]
n_epochs = 100
epoch_size = 8000
lr = 1e-4
beta = 0.00005



###############################################################################
# set directory where to save the output, losses and condition numbers

save_dir = '/scratch/vl1019/icassp24_data/'

###############################################################################



# enable GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the teacher fb

with open('Freqz/CQT.pkl', 'rb') as fp:
    CQT = pickle.load(fp)

CQT_freq = CQT["freqz"]
CQT_torch = torch.from_numpy(CQT_freq.T) # is now shape as (J, N)

# compute the initializations

w_init = fb.random_filterbank(spec["N"], spec["J"], spec["T"], tight=False, to_torch=True, support_only=True)

# load the data such that it comes in shape (batch_size, 1, spec["N"])
# a dataloading routine for the speech dataset

data = fb.generate_random_sine(
    spec["sr"],
    spec["fmin"],
    spec["fmax"],
    spec["N"],
    spec["batch_size"])


def filterbank_response_fft(x, w):
    x = x.reshape(x.shape[0], 1, x.shape[-1])
    w = w.unsqueeze(0).float()
    Wx = torch.fft.ifft(torch.fft.fft(x, dim=-1) * w, dim=-1)
    Wx = torch.abs(Wx)
    hann = torch.hann_window(spec["T"]).unsqueeze(0).unsqueeze(0)
    phi = torch.ones(spec["J"], spec["J"], spec["T"])*hann
    Ux = F.conv1d(Wx, phi, bias=None, stride=256, padding=0)
    return Ux

# the student

class TDFilterbank_real(torch.nn.Module):
    def __init__(self, spec, w):
        super().__init__()
        
        self.psi = torch.nn.Conv1d(
            in_channels=1,
            out_channels=spec["J"],
            kernel_size=spec["T"],
            stride=1,
            padding=0,
            bias=False)

        self.psi.weight.data = w[:, :spec["T"]].unsqueeze(1).float()        
    
    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[-1])
        x = F.pad(x, (0, spec["T"]-1), mode='circular',)
        Wx = torch.abs(self.psi(x))
        hann = torch.hann_window(spec["T"]).unsqueeze(0).unsqueeze(0)
        phi = torch.ones(spec["J"], spec["J"], spec["T"])*hann
        Ux = F.conv1d(Wx, phi, bias=None, stride=256, padding=0)
        return Ux

# the loss

class KappaLoss(nn.Module):
    def __init__(self):
        super(KappaLoss, self).__init__()
        self.loss = nn.CosineSimilarity(dim=0)

    def forward(self, inputs, targets, w, beta):
        w_hat = torch.sum(torch.abs(torch.fft.fft(w,dim=1))**2,dim=0)
        B = torch.max(w_hat,dim=0).values
        A = torch.min(w_hat,dim=0).values
        loss = 0.5*torch.mean(1-self.loss(inputs, targets)) + beta*B/A
        return loss
    
# the training routine

def train(baseline, penalization, lr, beta, n_epochs, epoch_size):
    optimizer = torch.optim.Adam(baseline.parameters(), lr=lr)
    cos = torch.nn.CosineSimilarity(dim=0)
    criterion = KappaLoss()
    if penalization == 'cos':
        beta=0
    if penalization == 'kappa':
        beta=beta
    losses = []
    conditions = []

    running_loss = 0.0
    for _ in range(epoch_size):
        inputs = next(data)
        outputs = baseline(inputs)
        targets = filterbank_response_fft(inputs, CQT_torch)
        loss = 0.5*torch.mean(1-cos(outputs, targets))
        running_loss += loss.item()
    print(1000 * running_loss)

    w = baseline.psi.weight.detach().numpy()[:,0,:]
    w = np.pad(w, ((0,0),(0, spec["N"]-spec["T"])), constant_values=0)
    A,B = fb.frame_bounds_lp(w)
    conditions.append(B/A)
    
    losses.append(running_loss)

    for _ in range(n_epochs):
        running_loss = 0.0
        for i in range(epoch_size):
            inputs = next(data)
            optimizer.zero_grad()
            outputs = baseline(inputs)
            targets = filterbank_response_fft(inputs, CQT_torch)

            w = baseline.psi.weight[:,0,:]
            w = F.pad(w,(0,spec["N"]-spec["T"]), value=0)
            loss = criterion(outputs, targets, w, beta)

            loss.backward()
            optimizer.step()

            loss2 = 0.5*torch.mean(1-cos(outputs, targets))
            running_loss += loss2.item()
        losses.append(running_loss)

        w = baseline.psi.weight.detach().numpy()[:,0,:]
        w = np.pad(w, ((0,0),(0, spec["N"]-spec["T"])), constant_values=0)
        A,B = fb.frame_bounds_lp(w)
        conditions.append(B/A)

        print(1000 * running_loss)

    return losses, conditions


########################
# train
########################

# only cosine loss

baseline_no = TDFilterbank_real(spec, w_init).to(device)

losses_no, conditions_no = train(
    baseline=baseline_no,
    penalization='cos',
    lr=lr,
    beta=beta,
    n_epochs=n_epochs,
    epoch_size=epoch_size)

w_no = baseline_no.psi.weight.detach().numpy()[:,0,:]
np.save(save_dir+'losses_no_synth.npy', losses_no)
np.save(save_dir+'conditions_no_synth.npy', conditions_no)
np.save(save_dir+'w_no_synth.npy', w_no)

# kappa loss

baseline_kappa = TDFilterbank_real(spec, w_init).to(device)

losses_kappa, conditions_kappa = train(
    baseline=baseline_kappa,
    penalization='kappa',
    lr=lr,
    beta=beta,
    n_epochs=n_epochs,
    epoch_size=epoch_size)

w_kappa = baseline_kappa.psi.weight.detach().numpy()[:,0,:]
np.save(save_dir+'losses_kappa_synth.npy', losses_no)
np.save(save_dir+'conditions_kappa_synth.npy', conditions_no)
np.save(save_dir+'w_kappa_synth.npy', w_kappa)