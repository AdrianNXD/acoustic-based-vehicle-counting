import pandas as pd
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

class ChunkDataset(Dataset):
    def __init__(self, chunk_folder:str, chunk_labels:pd.DataFrame, samplerate:int, transf:list=None, n_mels:int=80, n_frames:int=80, n_fft:int=1024):
        
        self.chunk_folder = Path(chunk_folder)
        self.chunk_labels = chunk_labels
        self.transf = transf
            
        self.samplerate = samplerate

        # parameters for the mel spectogram representation
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.n_fft = n_fft

        # normally a mel spectogram is defined by the size of the n_mels and the hop_length.
        # however we want to have spectograms with determined resolutions. Thats why we dont
        # use the hop_length but rather calculate it by the number of required n_frames. Thus
        # the resulting spectograms have the size n_mels x n_frames which is 80x80 pixels by 
        # default.
        self.clip_samples = self.samplerate * 2
        self.hop_length = (self.clip_samples - self.n_fft) // (self.n_frames - 1)

        # create the melspec and db converters which we will use to generate our mel specs.
        self.db_converter = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.melspec_converter = torchaudio.transforms.MelSpectrogram(
            sample_rate = self.samplerate,
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            n_mels = self.n_mels,
        )
        
    def __len__(self):
        return len(self.chunk_labels)

    def __getitem__(self, idx):

        chunk_idx = self.chunk_labels["chunk_idx"].iloc[idx]
        path = self.chunk_folder / f"{chunk_idx}.wav"

        # load audio
        waveform, sr = torchaudio.load(path)

        # convert it to the mel spectogram
        mel = self.db_converter(self.melspec_converter(waveform)).squeeze(0)
        mel = (mel + 80) / 80 # values are between -80 and 0. Convert them to a range of [0, 1]
        mel = mel.unsqueeze(0)

        if self.transf:
            mel = self.transf(mel)

        # convert the label into the tensor format
        label = torch.tensor(
            self.chunk_labels["pass_event"].iloc[idx],
            dtype=torch.long
        )

        return mel, label