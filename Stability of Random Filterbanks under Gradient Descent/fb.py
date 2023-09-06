import numpy as np
import torch
import scipy as sp
import torch.nn.functional as F
import torch.nn as nn


# computes the frame bounds of a filterbank given as a numpy array of row vectors via Littlewood-Payley

def frame_bounds_lp(w, freq=False):
    # if the filters are given already as frequency responses
    if freq:
        w_hat = np.sum(np.abs(w)**2,axis=1)
    else:
        w_hat = np.sum(np.abs(np.fft.fft(w,axis=1))**2,axis=0)
    Lam_max = np.max(w_hat)
    Lam_min = np.min(w_hat)

    return Lam_min, Lam_max


# creates a random filterbank of J filters of support T, padded with zeros to have length N
# and optionally its tightened version

def random_filterbank(N, J, T, norm=True, tight=True, to_torch=True, support_only=False):
    if T == None:
        T = N
    if norm:
        w = np.random.randn(J, T)/np.sqrt(T)/np.sqrt(J)
    else:
        w = np.random.randn(J, T)
    if support_only:
        w_cat = w
    else:
        z = np.zeros([J, N-T])
        w_cat = np.concatenate((w,z),axis=1)
    if tight:
        W = np.concatenate([sp.linalg.circulant(w_cat[k, :]) for k in range(w_cat.shape[0])])
        S = np.matmul(W.T,W)
        S_sq = np.linalg.inv(sp.linalg.sqrtm(S))
        w_tight = np.matmul(S_sq,w_cat.T).T
        if to_torch:
            return torch.from_numpy(w_cat), torch.from_numpy(w_tight)
        else:
            return w_cat, w_tight
    if to_torch:
        return torch.from_numpy(w_cat)
    else:
        return w_cat

    

# diagonal of the frame operator of a filterbank with filters w

def S_diag(w):
    N = w.shape[1]
    diag = torch.zeros(N)
    for l in range(N):
        rolled_w = torch.roll(w, shifts=l, dims=0)
        squared_norms = torch.sum(rolled_w**2, dim=1)
        diag[l] = torch.sum(squared_norms)

    return diag


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

def filterbank_response(x, w, mag=True):
    # some shaping
    x = x.reshape(x.shape[0], 1, x.shape[-1])
    x = F.pad(x, (0, x.shape[-1]-1), mode='circular',)
    w = w.unsqueeze(1).float()

    # filtering
    out = F.conv1d(x, w, bias=None, stride=1, padding=0)
    # magnitude
    if mag:
        out = torch.abs(out)
    return out




# computes the circulant matrix version of w (tensor)

def circulant(w):
    dim = 1
    N = w.shape[dim]
    J = w.shape[0]
    tmp = torch.cat([w.flip((dim,)), torch.narrow(w.flip((dim,)), dim=dim, start=0, length=N-1)], dim=dim)
    tmp = tmp.unfold(dim, N, 1).flip((-1,))
    return tmp.reshape(J*N, N)




def imp2freq(imp, mag=True, complex=False):
    fr = np.fft.fft(imp,axis=1)#/np.sqrt(2)
    if complex:
        return fr.T
    if mag:
        return np.abs(fr.T)
    return np.real(fr.T), np.imag(fr.T)

def freq2imp(freq, mag=True, complex=False):
    imp = np.fft.ifft(freq,axis=0)
    if complex:
        return imp.T
    if mag:
        return np.abs(imp.T)
    return np.real(imp.T), np.imag(imp.T)