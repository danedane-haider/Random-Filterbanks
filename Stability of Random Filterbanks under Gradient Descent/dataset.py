import torch
from torch.utils.data import Dataset
import numpy as np
from fb_utils import filterbank_response_fft
import soundfile
import os
import pandas as pd
import librosa

class TinySol(Dataset):
    def __init__(self,
                 info_csv_path,
                 data_dir,
                 filterbank_specs,
                 target_filterbank,
                 dataset_type='train'):

        info_df = pd.read_csv(info_csv_path)
        self.data_dir = data_dir
        self.filterbank_specs = filterbank_specs
        self.target_filterbank = target_filterbank
        self.seg_length = 4096/44100

        # split dataset into train, val, test
        dataset_length = len(info_df)
        train_dataset_length = int(dataset_length*0.9)
        val_dataset_length = int(dataset_length*0.1)

        # scramble dataset
        info_df = info_df.sample(frac=1).reset_index(drop=True)

        # get dataset type
        if dataset_type == 'train':
            self.info_df = info_df.iloc[:train_dataset_length]
            self.length = train_dataset_length
        elif dataset_type == 'val':
            self.info_df = info_df.iloc[train_dataset_length:]
            self.length = val_dataset_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        info = self.info_df.iloc[idx]

        # Get info
        fold = info["Fold"]
        instrument_family = info["Family"]
        instrument = info["Instrument (abbr.)"]
        pitch_id = info["Pitch ID"]
        dynamics_id = info["Dynamics ID"]

        # Get audio
        x, fs = soundfile.read(os.path.join(self.data_dir, info["Path"]))

        seg_length = self.seg_length * fs

        # sample from middle a segment
        start = int(max(x.shape[0]//2 - seg_length//2, 0))
        stop = int(min(x.shape[0]//2 + seg_length//2, x.shape[0]-1))
        x = torch.tensor(x[start:stop], dtype=torch.float32)
        # x = librosa.util.fix_length(x, size=seg_length)

        x_out = filterbank_response_fft(x.unsqueeze(0), self.target_filterbank, self.filterbank_specs).squeeze(0)

        return {
            "x": x,
            "x_out": x_out,
            "instrument_family": instrument_family,
            "instrument": instrument,
            "pitch_id": pitch_id,
            "dynamics_id": dynamics_id,
            "fold": fold,
        }
    
class NTVOW(Dataset):
    def __init__(self,
                 dataset_path:str,
                 dataset_type:str,
                 filterbank_specs:dict,
                 target_filterbank:torch.tensor) -> None:
        super().__init__()

        self.dataset_path = dataset_path
        files = os.listdir(self.dataset_path)

        if '.DS_Store' in files:
            files.remove('.DS_Store')
        
        self.seg_length = 4092/48000
        self.filterbank_specs = filterbank_specs
        self.target_filterbank = target_filterbank
        length = len(files)
        # scramle files
        np.random.shuffle(files)

        train_length = int(length*0.9)
        val_length = int(length*0.1)

        train_files = files[:train_length]
        val_files = files[train_length:]

        if dataset_type == 'train':
            self.files = train_files
            self.length = train_length
        elif dataset_type == 'val':
            self.files = val_files
            self.length = val_length
        else:
            raise ValueError("dataset_type must be either 'train' or 'val'")


    def __len__(self):
        return self.length
    
    def __getitem__(self, idx:int):

        file = self.files[idx]
        x, fs = soundfile.read(os.path.join(self.dataset_path, file))
        seg_length = self.seg_length * fs

        # sample from middle a segment
        start = int(max(x.shape[0]//2 - seg_length//2, 0))
        stop = int(min(x.shape[0]//2 + seg_length//2, x.shape[0]-1))
        x = torch.tensor(librosa.util.fix_length(x[start:stop], size=int(seg_length)), dtype=torch.float32)

        x_out = filterbank_response_fft(x.unsqueeze(0), self.target_filterbank, self.filterbank_specs).squeeze(0)

        return {
            "x": x,
            "x_out": x_out,
        }

if __name__ == "__main__":
    import pickle
    from fb_utils import HYPERPARAMS
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    target = 'MEL'
    # get current working directory of file
    cwd = os.path.dirname(os.path.abspath(__file__))

    with open(cwd+'/targets/'+target+'.pkl', 'rb') as fp:
        FB = pickle.load(fp)

    FB_freq = FB["freqz"]
    FB_torch = torch.from_numpy(FB_freq.T)

    spec = {
        "N": 2**12,
        "J": 96,
        "T": 1024,
        "sr": 16000,
        "fmin": 64,
        "fmax": 8000,
        "stride": 512,
        "batch_size": 1
    }

    # example = TinySol(
    #     info_csv_path="/Users/felixperfler/Documents/ISF/Random-Filterbanks/TinySOL_metadata.csv",
    #     data_dir="/Users/felixperfler/Documents/ISF/Random-Filterbanks/TinySOL2020",
    #     target_filterbank=FB_torch,
    #     filterbank_specs=spec,
    #     dataset_type='test'
    # )


    example = NTVOW(
        "/Users/felixperfler/Documents/ISF/Random-Filterbanks/NTVOW",
        'val',
        target_filterbank=FB_torch,
        filterbank_specs=spec,
    )

    dataloader = DataLoader(example, batch_size=1, shuffle=True, num_workers=0)
    print(len(dataloader))

    for batch in dataloader:
        x_out = batch["x_out"]
        plt.imshow(x_out[0].numpy())
        plt.show()
