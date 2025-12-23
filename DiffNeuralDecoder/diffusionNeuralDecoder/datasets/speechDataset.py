import pathlib
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]

PHONE_DEF_SIL = [
    '<pad>','AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH', ' ', '<eos>'
]

PHONE_TO_ID = {phone: idx for idx, phone in enumerate(PHONE_DEF_SIL)}

CHANG_PHONE_DEF = [
    'AA', 'AE', 'AH', 'AW',
    'AY', 'B',  'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'P', 'R', 'S',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z'
]

CONSONANT_DEF = ['CH', 'SH', 'JH', 'R', 'B',
                 'M',  'W',  'V',  'F', 'P',
                 'D',  'N',  'L',  'S', 'T',
                 'Z',  'TH', 'G',  'Y', 'HH',
                 'K', 'NG', 'ZH', 'DH']
VOWEL_DEF = ['EY', 'AE', 'AY', 'EH', 'AA',
             'AW', 'IY', 'IH', 'OY', 'OW',
             'AO', 'UH', 'AH', 'UW', 'ER']

SIL_DEF = ['SIL']

class BrainToTextDataset(Dataset):
    """
    PyTorch Dataset for Brain-to-Text competition data.
    Args:
        data_path (str): Path to the preprocessed data directory (recommended to be an OS path env variable).
        partition (str): Data partition to use ('train', 'test', 'competitionHoldOut').

    The preprocessing script creates train_data.pt, test_data.pt, and competitionHoldOut_data.pt files in the specified data_path.
    """

    def __init__(self, data_path, partition = 'train'):
        self.data_path = data_path
        self.partition = partition
        self.data = torch.load(os.path.join(data_path, f'{partition}, _data.pt'))

class PhonemeDataset(Dataset):
    """Dataset for pre-computed phoneme sequences"""
    
    def __init__(self, data_path):
        # Load compressed numpy file
        data = np.load(data_path)
        
        self.phoneme_data = data['phoneme_data']    # (N, 128) int16
        self.phoneme_mask = data['phoneme_mask']    # (N, 128) bool
        self.phoneme_to_id = data['phoneme_to_id'].item()
        self.max_phoneme_len = data['max_phoneme_len'].item()
        
        print(f"Loaded {len(self.phoneme_data)} phoneme sequences")
    
    def __len__(self):
        return len(self.phoneme_data)
    
    def __getitem__(self, idx):
        # Direct array indexing - very fast!
        phonemes = torch.from_numpy(self.phoneme_data[idx]).long()  # (128,)
        mask = torch.from_numpy(self.phoneme_mask[idx])  # (128,)
        
        return {
            'phonemes': phonemes,
            'mask': mask
        }

# Usage
# dataset = PhonemeDataset('../../../preprocessed_data/phoneme_data.npz')