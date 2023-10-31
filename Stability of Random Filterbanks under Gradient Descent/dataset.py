import torch
from torch.utils.data import Dataset
from fb_utils import filterbank_response_fft, random_filterbank
import librosa
import soundfile
import os
import pandas as pd

class TinySol(Dataset):
    def __init__(self,
                 info_csv_path,
                 data_dir,
                 filterbank_specs,
                 target_filterbank):
        
        self.info_df = pd.read_csv(info_csv_path)
        self.data_dir = data_dir
        self.length = len(self.info_df)
        self.filterbank_specs = filterbank_specs
        self.target_filterbank = target_filterbank
        self.seg_length = 4096/44100

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



if __name__ == "__main__":
    import pickle
    from fb_utils import HYPERPARAMS
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    target = 'VQT'
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

    example = TinySol(
        info_csv_path="/Users/felixperfler/Documents/ISF/Random-Filterbanks/TinySOL_metadata.csv",
        data_dir="/Users/felixperfler/Documents/ISF/Random-Filterbanks/TinySOL2020",
        target_filterbank=FB_torch,
        filterbank_specs=spec,
    )

    dataloader = DataLoader(example, batch_size=1, shuffle=True, num_workers=0)

    for batch in dataloader:
        x_out = batch["x_out"]
        plt.imshow(x_out[0].numpy())
        plt.show()
