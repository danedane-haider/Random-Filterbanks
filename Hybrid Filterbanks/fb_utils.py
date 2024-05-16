import numpy as np
import scipy as sp

def hybrid_filterbank(Psi, w, N, T, freq=False):
    if T == None:
        T = N
    J = Psi.shape[0]
    T_psi = Psi.shape[1]
    w_pad = np.pad(w, ((0,0),(0, N-T)), constant_values=0)
    Psi_pad = np.pad(Psi, ((0,0),(0, N-T_psi)), constant_values=0)
    if freq:
        hybrid_filters = np.fft.fft(Psi_pad, axis=1) * np.fft.fft(w_pad, axis=1)
    else:
        hybrid_filters = np.fft.ifft(np.fft.fft(Psi_pad, axis=1) * np.fft.fft(w_pad, axis=1))
    return hybrid_filters

def random_hybrid_filterbank_energy(x, Psi, N, T):
    y = 0
    J = Psi.shape[0]
    w = np.random.randn(J, T)/np.sqrt(T)
    hybrid_filters_f = hybrid_filterbank(Psi, w, N, T, freq=True)
    for ii in range(J):
        y += np.linalg.norm(np.fft.ifft(np.fft.fft(x) * hybrid_filters_f[ii,:]))**2
    return y

def random_filterbank_experiment(x, Psi, N, T_vals, num = 1000):
    T_len = len(T_vals)
    Y = np.zeros([T_len, num])
    for jj in range(T_len):
        for ii in range(num):
            Y[jj,ii] = random_hybrid_filterbank_energy(x, Psi, N, T_vals[jj])
    return Y

def circ_autocorr(x,l):
    return np.dot(x,np.roll(x,l))

def variance(w,T):
    v = [(T-np.abs(t))*circ_autocorr(w, t)**2 for t in range(-T,T+1)]
    return np.sum(v)

def frame_bounds(w):
    w_hat = np.fft.fft(w,axis=1)
    w_hat = np.abs(w_hat)**2
    lp = np.real(np.sum(w_hat,axis=0))
    B = np.max(lp)
    A = np.min(lp)
    return A, B

def random_filterbank_energy(x, J, N, T):
    if T == None:
        T = N
    w = np.random.randn(J, T)/np.sqrt(T*J)
    w_pad = np.pad(w, ((0,0),(0, N-T)), constant_values=0)
    y = 0
    for ii in range(J):
        y += np.linalg.norm(np.fft.ifft(np.fft.fft(x)*np.fft.fft(w_pad[ii,:])))**2
    return y

def eig_QT(x, J, T):
    C = sp.linalg.circulant(x)
    CT = C[:,:T]
    QT = np.matmul(CT.T,CT)
    return np.linalg.eigvals(QT)

def brownian_noise(N,p):
    prob = [p, 1-p] 
    start = 0
    walk = [start]
    rr = np.random.random(N-1)
    downp = rr < prob[0]
    upp = rr > prob[1]
    for idownp, iupp in zip(downp, upp):
        down = idownp and walk[-1] > -1
        up = iupp and walk[-1] < 1
        walk.append(walk[-1] - down + up)
    return walk

# def random_filterbank_frame_bounds(N, J, T):
#     if T == None:
#         T = N
#     w = np.random.randn(J, T)/np.sqrt(T*J)
#     w = np.pad(w, ((0,0),(0, N-T)), constant_values=0)
#     A,B = frame_bounds(w)
#     return A,B

def exp_fb(N, J, T_vals):
    num = 1000
    lma_d = np.zeros(num)
    lmi_d = np.zeros(num)
    lma_all_mean = []
    lmi_all_mean = []
    lma_all_std = []
    lmi_all_std = []
    cond = []
    cond_mean = []
    cond_std = []

    for T in T_vals:
        for i in range(num):
            lmi, lma = frame_bounds(hybrid_filterbank(Psi, np.random.randn(J,N)/np.sqrt(T*J), N, T))
            lma_d[i] = lma
            lmi_d[i] = lmi
        lma_all_mean.append(np.mean(lma_d))
        lmi_all_mean.append(np.mean(lmi_d))
        lma_all_std.append(np.std(lma_d))
        lmi_all_std.append(np.std(lmi_d))
        cond.append(np.mean(lma_d)/np.mean(lmi_d))
        cond_mean.append(np.mean(cond))
        cond_std.append(np.std(cond))

    return np.array(lmi_all_mean), np.array(lmi_all_std), np.array(lma_all_mean), np.array(lma_all_std), np.array(cond_mean),np.array(cond_std)