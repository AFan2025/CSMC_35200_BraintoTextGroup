import scipy.io
import numpy as np
import torch
from g2p_en import G2p
from torch.utils.data import Dataset
import os
from neuralDecoder.datasets.speechDataset import PHONE_DEF, VOWEL_DEF, CONSONANT_DEF, SIL_DEF, PHONE_DEF_SIL

def load_mat_file(file_path):
    """Load a .mat file and return its content as a numpy array."""
    mat_contents = scipy.io.loadmat(file_path)
    # Assuming the relevant data is stored under the key 'data'
    data = mat_contents.get('data')
    if data is None:
        raise KeyError(f"'data' key not found in the .mat file: {file_path}")
    return data