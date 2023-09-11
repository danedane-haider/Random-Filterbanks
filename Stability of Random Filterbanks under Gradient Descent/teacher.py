from nnAudio.features.cqt import CQT
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import soundfile as sf
import torch.nn.functional as F
import pickle
import librosa

device = "cuda" if torch.cuda.is_available() else "cpu"

# load everything about the teachers as dictionaries, containing:
#   "freqz" frequency responses as 2D array (time,channels)
#   "centerfreq" 
#   "bandwidths"
#   "framebounds" computed without subsampling
with open('Freqz/MEL.pkl', 'rb') as fp:
    GAM = pickle.load(fp)
with open('Freqz/VQT.pkl', 'rb') as fp:
    VQT = pickle.load(fp)
with open('Freqz/CQT.pkl', 'rb') as fp:
    THIRD = pickle.load(fp)


HYPERPARAMS = {
    "speech": { #gam
        "n_filters": 42,
        "seg_length": 4095,
        "win_length": 1024,
        "stride": 1,
        "a_opt": 8,
        "sr": 16000, 
        "fmin": 25, 
        "fmax": 8000,
    },
    "music": { #vqt
        "n_filters": 96,
        "seg_length": 4096,
        "win_length": 1024, 
        "stride": 1,
        "a_opt": 16,
        "sr": 44100,
        "fmin": 100,
        "fmax": 22050,
    }, 
    "synth": {
        "bins_per_octave": 8,
        "n_filters": 58,
        "win_length": 2**10,
        "stride": 2**8,
        "sr": 44100,
        "fmin": 22050 // 2**8,
        "fmax": 22050,
        "seg_length": 4092,
        "n_samples": 1000,
    },
}


class CQTSineData(torch.utils.data.Dataset):
    def __init__(self, spec, n_samples):
        super().__init__()
        self.spec = spec
        self.cqt_params = {
            "bins_per_octave": spec["bins_per_octave"],
            "fmin": spec["fmin"],
            "hop_length": spec["stride"],
            "n_bins": spec["n_filters"],
            "sr": spec["sr"],
        }
        self.seg_length = spec["seg_length"]
        self.freqs = np.logspace(
            np.log10(spec["fmin"]/2),
            np.log10(spec["fmax"]*2),
            n_samples,
            endpoint=False
        )
        self.closure = CQT(**self.cqt_params)

    def __getitem__(self, freq_id):
        freq = self.freqs[freq_id]
        t = torch.arange(
            0, self.seg_length/self.spec["sr"], 1/self.spec["sr"])
        x = torch.sin(2*np.pi*freq*t)
        Y = self.closure(x).squeeze()
        return {'freq': freq, 'x': x, 'feature': Y}
    
    def __len__(self):
        return len(self.freqs)


class CQTSineDataModule(pl.LightningDataModule):
    def __init__(self, domain, batch_size):
        super().__init__()
        self.domain = domain
        self.num_workers = 0
        self.batch_size = batch_size

    def setup(self, stage=None):
        spec = HYPERPARAMS[self.domain]
        n_samples = spec["n_samples"]
        self.train_dataset = CQTSineData(spec, n_samples)

    def collate_batch(self, batch):
        freq = torch.cat([item['freq'] for item in batch])
        x = torch.cat([item['x'] for item in batch])
        Y = torch.cat([item['feature'] for item in batch])
        return {'freq': freq, 'x': x, 'feature': Y}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )



class SpectrogramDataModule(pl.LightningDataModule):
    def __init__(self, sav_dir, domain, batch_size):
        super().__init__()
        self.audio_dir = sav_dir
        self.batch_size = batch_size
        if domain == "speech":
            self.feature = "mel"
            self.coef_dir = 'Freqz/MEL.pkl'
            self.data_dir = 'audio/NTVOW'
        elif domain == "music":
            self.feature = "vqt"
            self.coef_dir = 'Freqz/VQT.pkl'
            self.data_dir = "audio/SOL"
        self.ids, self.file_names = self.get_ids()
        self.seg_length = HYPERPARAMS[domain]["seg_length"]
        self.stride = HYPERPARAMS[domain]["stride"]
        self.num_workers = 0

    def setup(self, stage=None):
        N = len(self.ids)
        train_ids = self.ids[:-N//5]
        self.train_dataset = SpectrogramData(train_ids, self.audio_dir, self.coef_dir, self.file_names, self.feature, self.seg_length, self.stride)
        
    def get_ids(self):
        files = []
        for root, dirs, fs in os.walk(os.path.join(self.audio_dir, self.data_dir)):
            for f in fs:
                if f[-3:] == "wav":
                    files.append(os.path.join(root, f))
        length = len(files)
        return list(i for i in np.arange(1, length + 1)), files

    def collate_batch(self, batch):
        feat = torch.stack([s['feature'] for s in batch], axis=0)
        x = torch.stack([s['x']for s in batch]) #batch, time, channel
        return {'feature': feat, 'x': x}


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=self.collate_batch,
                          num_workers=self.num_workers)

class SpectrogramData(Dataset):
    def __init__(self,
                 ids,
                 audio_dir,
                 coef_dir,
                 file_names, 
                 feature,
                 seg_length,
                 stride,
        ):
        super().__init__()
        self.feature = feature
        self.audio_dir = audio_dir #path to hdf5 file
        self.file_names = file_names
        self.seg_length = seg_length
        self.stride = stride
        self.ids = ids
        #load filter coefficients:
        with open(coef_dir, 'rb') as fp:
            self.freqz = pickle.load(fp)['freqz']

    def __getitem__(self, idx): 
        id = self.ids[idx]
        x, feat = self.feat_from_id(id)
        return {'feature': feat, 'x': x}

    def __len__(self):
        return len(self.ids)

    def feat_from_id(self, id):
        x, sr = sf.read(self.file_names[int(id)])
        #sample from middle a segment 
        start = max(x.shape[0]//2 - self.seg_length//2, 0)
        stop = min(x.shape[0]//2 + self.seg_length//2, x.shape[0]-1)
        x = torch.tensor(x[start:stop], dtype=torch.float32)
        x = librosa.util.fix_length(x, size=self.seg_length)
        feat = filterbank_response_fft(x, self.w, self.spec)
        return x, feat
    

# Teacher filtering:

def filterbank_response_fft(x, w, spec):
    # some shaping
    x = x.reshape(x.shape[0], 1, x.shape[-1])
    w = w.unsqueeze(0).float()
    Wx = torch.fft.ifft(torch.fft.fft(x, dim=-1) * w, dim=-1)
    Wx = torch.abs(Wx)
    phi = torch.ones(spec["num_filters"], spec["num_filters"], spec["window_length"])
    Ux = F.conv1d(Wx, phi, bias=None, stride=256, padding=0)
    return Ux







##############################################
# Old Teacher filtering routines:

def filtering_frequency(x, freqz, stride):
    """
    x: (sig_length, )
    freqz: (sig_length, n_filters)

    output: filterbank magnitude responses for x (n_filters, sig_length_subsampled)
    """
    
    freqz = torch.from_numpy(freqz)
    N = x.shape[0]

    # x to Fourier domain
    x_fft = torch.fft.fft(x) #(nfft)
    assert freqz.shape[0] == x_fft.shape[0]

    # complex pointwise multiplication with freqz and ifft with renormalization
    y = torch.fft.ifft(x_fft * freqz, dim=0)*freqz.shape[0] #(nfft, n_filters)

    # strided convolution = subsampling the full convolution
    y = y[torch.arange(0, N - np.mod(N,stride), stride), :].T #(n_filters, nfft_subsampled)

    # magnitude
    mag = torch.tensor(torch.imag(y) ** 2 + torch.real(y) ** 2, dtype=torch.float) #(frequency, n_filters)
    return mag


def get_imp(freqz, centering = True, to_torch = True):
    """
    Get the impulse reponses as tensors

    freqz: (sig_length, n_filters)

    output: centered impulse reponses (n_filters, sig_length)
    """

    # to time-domain --> impulse reponse
    imp = np.fft.ifft(freqz,axis=0)*np.sqrt(freqz.shape[0])
    imp_r = np.real(imp)
    imp_i = np.imag(imp)

    # center the impulse responses within the vector instead of at the origin/end
    if centering:
        imp_r = np.roll(imp_r,len(imp_r)//2,axis = 0).T
        imp_i = np.roll(imp_i,len(imp_i)//2,axis = 0).T
    if to_torch:
        imp_r = torch.from_numpy(imp_r)
        imp_i = torch.from_numpy(imp_i)
    return imp_r, imp_i


def filtering_time(x, freqz, stride, kernel_size):
    """
    Do the filtering with conv1D.

    x: (sig_length, )
    freqz: (sig_length, n_filters)

    output: filterbank magnitude responses for x (n_filters, sig_length_subsampled)
    """

    # some shaping to fit for conv1D
    x = x.T
    x = x.unsqueeze(0)
    x = F.pad(x, (0, kernel_size-1), mode='circular')
    imp_r, imp_i = get_imp(freqz) # remove lowpass filter 
    imp_rt = imp_r.unsqueeze(1).float()
    imp_it = imp_i.unsqueeze(1).float()

    out_r = F.conv1d(x, imp_rt, bias=None, stride=stride, padding=0)
    out_i = F.conv1d(x, imp_it, bias=None, stride=stride, padding=0)
    # magnitude
    mag = out_r ** 2 + out_i ** 2
    return mag


