import scipy.io
import numpy as np
import torch
from g2p_en import G2p
from torch.utils.data import Dataset, DataLoader
import os
from diffusionNeuralDecoder.datasets.speechDataset import PHONE_DEF_SIL, PHONE_TO_ID
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
import logging
import re
import nltk
from datasets import load_dataset
from tqdm import tqdm

from diffusionNeuralDecoder.preprocess_brain import ensure_nltk_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    load_dotenv()
    ensure_nltk_data()

    data_dir = os.getenv("GENERAL_PHONEME_CORPUS_DIR", None)
    output_dir = os.getenv("PREPROCESSED_DATA_DIR", None)
    if data_dir is None or output_dir is None:
        raise ValueError("Please set the GENERAL_PHONEME_CORPUS_DIR and PREPROCESSED_DATA_DIR environment variables.")
    max_phoneme_len = int(os.getenv("MAX_PHONEME_LEN", 128))
    logger.info(f"Max phoneme length set to {max_phoneme_len}")
    os.makedirs(output_dir, exist_ok=True)
    
    #loading data from data dir
    logger.info(f"Loading data from {data_dir}")
    with open(os.path.join(data_dir, "libriSpeechASR_text.txt"), "r") as f:
        lines = f.readlines()

    #initialize the G2p converter
    logger.info(f"initializing the G2p converter")
    g2p = G2p()
    phoneme_data = []
    phoneme_mask_data = []

    for i, line in tqdm(enumerate(lines)):
        if i % 1000 == 0:
            logger.info(f"Processing line: {i}")
        phoneme_sequence = g2p(line)
        # Convert phonemes to IDs, using <pad> for unknown phonemes
        phoneme_ids = np.array([PHONE_TO_ID.get(p, PHONE_TO_ID['<pad>']) for p in phoneme_sequence])
        
        #pad and mask phonemes
        phoneme_mask = np.zeros(max_phoneme_len, dtype=np.bool_)
        if len(phoneme_sequence) < max_phoneme_len:
            phoneme_mask[:len(phoneme_sequence)] = True
            
            #pad phoneme sequence
            phoneme_sequence = phoneme_sequence + [PHONE_TO_ID['<pad>']] * (max_phoneme_len - len(phoneme_sequence))
            padder = np.zeros((max_phoneme_len - len(phoneme_sequence),), dtype=np.int32)* PHONE_TO_ID['<pad>']
            phoneme_sequence = np.concatenate([phoneme_sequence, padder], axis=0)

        phoneme_data.append(phoneme_ids)
        phoneme_mask_data.append(phoneme_mask)

    #converting to numpy arrays
    phoneme_data = np.array(phoneme_data, dtype=np.int16)  # (N, 128)
    phoneme_mask_data = np.array(phoneme_mask_data, dtype=np.bool_)  # (N, 128)
    logger.info(f"Processed {len(phoneme_data)} phoneme sequences.")

    #saving preprocessed data
    output_dir_path = os.path.join(output_dir, "phoneme_data.npz")
    np.savez_compressed(
        output_dir_path,
        phoneme_data=phoneme_data,        # (N, max_phoneme_len) int16
        phoneme_mask=phoneme_mask_data,   # (N, max_phoneme_len) bool
        phoneme_to_id=PHONE_TO_ID,        # Save vocabulary for reference
        max_phoneme_len=max_phoneme_len   # Save config
    )
    logger.info(f"Phoneme preprocessing completed. Data saved to {output_dir_path}")