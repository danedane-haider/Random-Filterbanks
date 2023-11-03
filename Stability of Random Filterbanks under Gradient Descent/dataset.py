import os
import torch
from torch.utils.data import Dataset
from fb_utils import filterbank_response_fft
import soundfile
import pandas as pd

class TinySol(Dataset):
    def __init__(self,
                 info_csv_path: str,
                 data_dir: str,
                 filterbank_specs: dict,
                 target_filterbank: torch.Tensor,
                 dataset_type:str='train'):

        info_df = pd.read_csv(info_csv_path)
        self.data_dir = data_dir
        self.filterbank_specs = filterbank_specs
        self.target_filterbank = target_filterbank
        self.seg_length = 4096/44100

        # split dataset into train, val, test
        dataset_length = len(info_df)
        train_dataset_length = int(dataset_length*0.8)
        val_dataset_length = int(dataset_length*0.1)
        test_dataset_length = int(dataset_length*0.1)

        # scramble dataset
        info_df = info_df.sample(frac=1).reset_index(drop=True)

        # get dataset type
        if dataset_type == 'train':
            self.info_df = info_df.iloc[:train_dataset_length]
            self.length = train_dataset_length
        elif dataset_type == 'val':
            self.info_df = info_df.iloc[train_dataset_length:train_dataset_length+val_dataset_length]
            self.length = val_dataset_length
        elif dataset_type == 'test':
            self.info_df = info_df.iloc[train_dataset_length+val_dataset_length:]
            self.length = test_dataset_length

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
