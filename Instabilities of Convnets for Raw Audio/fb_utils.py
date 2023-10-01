import numpy as np
import scipy as sp
from scipy.optimize import fsolve

def random_filterbank_energy(x, J, N, T):
    if T == None:
        T = N
    w = np.random.randn(J, T)/np.sqrt(T*J)
    w_pad = np.pad(w, ((0,0),(0, N-T)), constant_values=0)
    y = 0
    for ii in range(J):
        y += np.linalg.norm(np.fft.ifft(np.fft.fft(x)*np.fft.fft(w_pad[ii,:])))**2
    return y

def random_filterbank_experiment(x, J, N, T_vals, num = 1000):
    T_len = len(T_vals)
    Y = np.zeros([T_len, num])
    for jj in range(T_len):
        for ii in range(num):
            Y[jj,ii] = random_filterbank_energy(x, J, N, T_vals[jj])
    return Y

def eig_QT(x, J, T):
    C = sp.linalg.circulant(x)
    CT = C[:,:T]
    QT = np.matmul(CT.T,CT)
    return np.linalg.eigvals(QT)

def cher_bounds(x, J, T_vals, alpha):
    cc = np.zeros(len(T_vals))
    for i,T in enumerate(T_vals):
        lams = eig_QT(x, J, T)
        cc[i] = np.exp(-(alpha**2*J*T**2*np.linalg.norm(x)**4)/
                (2*alpha*T*np.max(np.abs(lams))*np.linalg.norm(x)**2 + 2*np.linalg.norm(lams)**2))
    return cc

def alpha_cher(x,J,T):
    p = 0.05
    if T[0] < 256:
        alpha_init = 0.1
    else:
        alpha_init = 0.01
    def find_alpha(a, T):
        return cher_bounds(x, J, T, a) - p
    a_T = fsolve(lambda a: find_alpha(a, T), alpha_init)
    return a_T[0]

def circ_autocorr(x,l):
    return np.dot(x,np.roll(x,l))

def variance(x,T):
    v = [(T-np.abs(t))*circ_autocorr(x, t)**2 for t in range(-T,T+1)]
    return np.sum(v)

def alpha_cheb(x,J,T_vals,p):
    alpha = np.zeros(len(T_vals))
    for jj in range(len(T_vals)):
        alpha[jj] = np.sqrt((p*J*(T_vals[jj]*circ_autocorr(x,0))**2)**(-1)*variance(x,T_vals[jj])*2)
    return alpha

def alpha_can(x,J,T_vals,p):
    alpha = np.zeros(len(T_vals))
    for jj in range(len(T_vals)):
        alpha[jj] = circ_autocorr(x,0)**(-1)*np.sqrt(2*(p**(-1)-1)*variance(x,T_vals[jj])/(J*T_vals[jj]**2))
    return alpha

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

def frame_bounds(w):
    w_hat = np.fft.fft(w,axis=1)
    w_hat = np.abs(w_hat)**2
    lp = np.sum(w_hat,axis=0)
    B = np.max(lp)
    A = np.min(lp)
    return A, B

def random_filterbank_frame_bounds(N, J, T):
    if T == None:
        T = N
    w = np.random.randn(J, T)/np.sqrt(T*J)
    w = np.pad(w, ((0,0),(0, N-T)), constant_values=0)
    A,B = frame_bounds(w)
    return A,B

def chi_extreme(J, T_vals, num=1000):   
    Y_min = np.zeros(num)
    Y_max = np.zeros(num)
    Y_min_mean = []
    Y_min_std = []
    Y_max_mean = []
    Y_max_std = []
    for j,T in enumerate(T_vals):
        for i in range(num):
            Y_min[i] = np.min(np.random.chisquare(J, T))
            Y_max[i] = np.max(np.random.chisquare(J, T))
        Y_min_mean.append(np.mean(Y_min))
        Y_min_std.append(np.std(Y_min))
        Y_max_mean.append(np.mean(Y_max))
        Y_max_std.append(np.std(Y_max))
    return np.array(Y_min_mean), np.array(Y_max_mean)

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
            lmi, lma = random_filterbank_frame_bounds(N, J, T)
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