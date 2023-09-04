import numpy as np
import torch
import scipy as sp
import torch.nn.functional as F
import torch.nn as nn


# computes the frame bounds of a filterbank given as a numpy array of row vectors via Littlewood-Payley

def frame_bounds_lp(w, freq=False):
    # if the filters are given already as frequency responses
    if freq:
        w_hat = np.sum(np.abs(w)**2,axis=0)
    else:
        w_hat = np.sum(np.abs(np.fft.fft(w,axis=1))**2,axis=0)
    Lam_max = np.max(w_hat)
    Lam_min = np.min(w_hat)

    return Lam_min, Lam_max


# creates a random filterbank of J filters of support T, padded with zeros to have length N
# and optionally its tightened version

def random_filterbank(N, J, T, tight=True, to_torch=True):
    if T == None:
        T = N
    # normalization
    w = np.random.randn(J, T)/np.sqrt(T)/np.sqrt(J)
    z = np.zeros([J, N-T])
    w_cat = np.concatenate((w,z),axis=1)
    if tight==False:
        return torch.from_numpy(w_cat)
    if tight==True:
        # analysis operator
        W = np.concatenate([sp.linalg.circulant(w_cat[k, :]) for k in range(w_cat.shape[0])])
        # frame operator
        S = np.matmul(W.T,W)
        S_sq = np.linalg.inv(sp.linalg.sqrtm(S))
        w_tight = np.matmul(S_sq,w_cat.T).T
    if to_torch:
        return torch.from_numpy(w_cat), torch.from_numpy(w_tight)
    else:
        return w_cat, w_tight
    


# generate a dataset of random sine waves

def generate_random_sine(sample_rate, f_min, f_max, length, batch_size):
    time = torch.arange(length).reshape(1, -1) / sample_rate
    log2_min = np.log2(f_min)
    log2_range = np.log2(f_max) - np.log2(f_min)
    while True:
        log2_f0 = log2_min + log2_range * torch.rand(batch_size)
        f0 = (2**log2_f0).reshape(-1, 1)
        yield torch.sin(2 * torch.pi * f0 * time)


# compute filterbank responses using conv1D (circulant)

def filterbank_response(x, w):
    # some shaping
    x = x.reshape(x.shape[0], 1, x.shape[-1])
    x = F.pad(x, (0, x.shape[-1]-1), mode='circular',)
    w = w.unsqueeze(1).float()

    # filtering
    out = F.conv1d(x, w, bias=None, stride=1, padding=0)
    # magnitude
    mag = torch.abs(out)
    return mag


# computes the circulant matrix version of w (tensor)

def circulant(w):
    dim = 1
    N = w.shape[dim]
    J = w.shape[0]
    tmp = torch.cat([w.flip((dim,)), torch.narrow(w.flip((dim,)), dim=dim, start=0, length=N-1)], dim=dim)
    tmp = tmp.unfold(dim, N, 1).flip((-1,))
    return tmp.reshape(J*N, N)