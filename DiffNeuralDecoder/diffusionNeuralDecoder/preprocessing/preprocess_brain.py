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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

COMPETITION_DATA_DIR = os.getenv('COMPETITION_DATA_DIR', '/data/CSMC_35200_BraintoTextGroup/competitionData')
PREPROCESSED_DATA_DIR = os.getenv('PREPROCESSED_DATA_DIR', '/data/CSMC_35200_BraintoTextGroup/preprocessed_data')

ROWS = np.array([
    [62, 60, 63, 58, 59, 61, 56, 57, 125, 123, 121, 119, 117, 115, 113, 127],
    [51, 53, 54, 55, 45, 49, 52, 50, 126, 124, 122, 120, 118, 116, 114, 111],
    [43, 41, 47, 48, 46, 42, 39, 37, 112, 110, 109, 108, 107, 106, 105, 104],
    [35, 33, 44, 40, 38, 36, 34, 32, 103, 102, 101, 100, 99, 97, 98, 96],
    [94, 95, 93, 92, 91, 90, 89, 88, 31, 29, 27, 25, 23, 21, 17, 30],
    [87, 86, 84, 85, 82, 83, 81, 80, 28, 26, 19, 15, 13, 20, 24, 22],
    [79, 77, 75, 73, 71, 69, 67, 65, 11, 9, 18, 12, 10, 7, 14, 16, ],
    [78, 76, 74, 72, 70, 68, 66, 64, 8, 5, 4, 6, 3, 2, 0, 1]
]).T #originally columns but rows seem easier for indexing, Shape (16, 8)

TOLERABLE_SEQ_LEN = os.getenv('TOLERABLE_SEQ_LEN', None)  #extra tolerance length when finding max sequence length for brain data
TOLERABLE_SEQ_PERCENTILE = os.getenv('TOLERABLE_SEQ_PERCENTILE', 95)  #percentile for tolerable sequence length if MAX_SEQ_LEN not set
MAX_PHONEME_LEN = int(os.getenv("MAX_PHONEME_LEN", 128))

def ensure_nltk_data():
    """Download NLTK data if not already present."""
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('punkt', quiet=True)

#G2p requires an additional nltk data download
ensure_nltk_data()

def find_max_seq_len(competition_data_dir=COMPETITION_DATA_DIR, tolerable_len=None, tolerable_percentile=95):
    """
    Find the maximum sequence length across all .mat files in the Brain-to-Text competition dataset.
    Assumes that data is organized in train, test, and competitionHoldOut folders.
    Args:
        competition_data_dir (str): Path to the competition data directory.
        tolerable_len (int, optional): Extra tolerance length when finding max sequence length.
        tolerable_percentile (int, optional): Percentile for tolerable sequence length if tolerable_len is not set.
    Returns:
        int: Maximum sequence length found.
        list: List of all sequence lengths found.
    """
    logger.info("Finding maximum sequence length in dataset...")
    seq_lengths = []

    for division in ["train", "test", "competitionHoldOut"]:
        names_list = os.listdir(os.path.join(competition_data_dir, division))
        for name in names_list:
            data_path = os.path.join(competition_data_dir, division, name)
            dat = scipy.io.loadmat(data_path)
            # features = np.concatenate([dat['tx1'][0,i][:,0:128], dat['spikePow'][0,i][:,0:128]], axis=1)
            for i in range(dat['sentenceText'].shape[0]):    
                input_features = dat['tx1'][0,i] # Shape: (num_time_steps, 256)
                seq_length = input_features.shape[0]
                seq_lengths.append(seq_length)
    if tolerable_len is not None:
        max_seq_len = tolerable_len
    else:
        max_seq_len = int(np.percentile(seq_lengths, tolerable_percentile))
    return max_seq_len, seq_lengths

#TO DO for outputting files
def preprocess_1D(sessionName, dataPath, max_seq_len, outputFolder):
    """
    Preprocess raw .mat data files into .pt files format for Brain-to-Text competition.
    This is the step for the model architecture that diregards spatial features and only uses 1D features like tx1 or spike power along the 256 flattented feature space.
    The output for each datapoint will be (T, 256)
    """
    partNames = ['train','test','competitionHoldOut']
    
    for partIdx in range(len(partNames)):
        sessionPath = dataPath + '/' + partNames[partIdx] + '/' + sessionName + '.mat'
        if not os.path.isfile(sessionPath):
            continue
            
        dat = scipy.io.loadmat(sessionPath)

        input_features = []
        transcriptions = []
        frame_lens = []
        n_trials = dat['sentenceText'].shape[0]

        #collect area 6v tx1 and spikePow features
        for i in range(n_trials):    
            #get time series of TX and spike power for this trial
            #first 128 columns = area 6v only
            features = np.concatenate([dat['tx1'][0,i][:,0:128], dat['spikePow'][0,i][:,0:128]], axis=1)

            sentence_len = features.shape[0]
            sentence = dat['sentenceText'][i].strip()

            input_features.append(features)
            transcriptions.append(sentence)
            frame_lens.append(sentence_len)

        #block-wise feature normalization
        blockNums = np.squeeze(dat['blockIdx'])
        blockList = np.unique(blockNums)
        blocks = []
        for b in range(len(blockList)):
            sentIdx = np.argwhere(blockNums==blockList[b])
            sentIdx = sentIdx[:,0].astype(np.int32)
            blocks.append(sentIdx)

        for b in range(len(blocks)):
            feats = np.concatenate(input_features[blocks[b][0]:(blocks[b][-1]+1)], axis=0)
            feats_mean = np.mean(feats, axis=0, keepdims=True)
            feats_std = np.std(feats, axis=0, keepdims=True)
            for i in blocks[b]:
                input_features[i] = (input_features[i] - feats_mean) / (feats_std + 1e-8)

        #convert to tfRecord file
        session_data = {
            'inputFeatures': input_features,
            'transcriptions': transcriptions,
            'frameLens': frame_lens
        }

def preprocess_2D(sessionName, dataPath, outputFolder, max_seq_len = 512, max_phoneme_len=128):
    """
    Preprocess raw .mat data files into .pt files format for Brain-to-Text competition.
    This is the step for the model architecture that uses the convolutional layers to process spatial features
    The output for each datapoint will be (T,H,W,2) where 2 channels are tx1 and spike power respectively.
    This is different from the 1D feature preprocessing where the output is (T, feature_dim)
    """
    
    partNames = ['train','test','competitionHoldOut']
    
    for partIdx in range(len(partNames)):
        sessionPath = dataPath + '/' + partNames[partIdx] + '/' + sessionName + '.mat'
        if not os.path.isfile(sessionPath):
            continue
            
        dat = scipy.io.loadmat(sessionPath)

        tx1_features = [] #List of arrays Datapoints x (Seq_len, 128)
        spikePow_features = [] #List of arrays Datapoints x (Seq_len, 128)
        transcriptions = [] #List of strings Datapoints x (sentence)
        frame_lens = [] # List of ints Datapoints x (Seq_len)
        n_trials = dat['sentenceText'].shape[0]
        input_features = []
        phoneme_tokens = []
        phoneme_masks = []

        #collect area 6v tx1 and spikePow features
        for i in range(n_trials):    
            #get time series of TX and spike power for this trial
            #first 128 columns = area 6v only
            #tx1: (time, 128), spikePow: (time, 128)
            tx1 = dat['tx1'][0,i][:,0:128]
            spikePow = dat['spikePow'][0,i][:,0:128]

            assert tx1.shape[0] == spikePow.shape[0]
            seq_len = tx1.shape[0]
            sentence = dat['sentenceText'][i].strip()

            #convert to phonemes
            phoneme_sequence = g2p_transcription(sentence)
            phoneme_sequence = np.array(phoneme_sequence, dtype=np.int32)

            #pad and mask phonemes
            phoneme_mask = np.zeros(max_phoneme_len, dtype=np.bool_)
            if len(phoneme_sequence) < max_phoneme_len:
                phoneme_mask[:len(phoneme_sequence)] = True
                #pad phoneme sequence
                phoneme_sequence = phoneme_sequence + [PHONE_TO_ID['<pad>']] * (max_phoneme_len - len(phoneme_sequence))
                padder = np.zeros((max_phoneme_len - len(phoneme_sequence),), dtype=np.int32)* PHONE_TO_ID['<pad>']
                phoneme_sequence = np.concatenate([phoneme_sequence, padder], axis=0)

            if seq_len >= max_seq_len:
                continue #skip this data point if too long because truncating would lose information and mismatch with transcriptions
            tx1_features.append(tx1)
            spikePow_features.append(spikePow)
            transcriptions.append(sentence)
            phoneme_tokens.append(phoneme_sequence)
            phoneme_masks.append(phoneme_mask)
            frame_lens.append(seq_len)

        #block-wise feature normalization
        #this is needed to be done because different blocks have different signal characteristics/conditions
        #normalize across the entire block x sequence for every feature in the block
        logger.info(f'Normalizing features block-wise for session {sessionName} in partition {partNames[partIdx]}')
        blockNums = np.squeeze(dat['blockIdx'])
        blockList = np.unique(blockNums)

        for block_id in blockList:
            #list of indexes of all trials in a single block
            block_trials = np.where(blockNums == block_id)[0]

            #normalize tx1 first 
            block_tx_features = np.concatenate([tx1_features[i] for i in block_trials], axis=0)
            tx_mean = np.mean(block_tx_features, axis=0, keepdims=True)
            tx_std = np.std(block_tx_features, axis=0, keepdims=True)

            #normalize spike power next
            block_spike_features = np.concatenate([spikePow_features[i] for i in block_trials], axis=0)
            spike_mean = np.mean(block_spike_features, axis=0, keepdims=True)
            spike_std = np.std(block_spike_features, axis=0, keepdims=True)
            for i in block_trials:
                tx1_features[i] = (tx1_features[i] - tx_mean) / (tx_std + 1e-8)
                spikePow_features[i] = (spikePow_features[i] - spike_mean) / (spike_std + 1e-8)

        #reshape features into H x W spatial maps
        ## THIS MIGHT NOT BE CORRECT - NEED TO DOUBLE CHECK THE INDEXING, if something looks off later come back here
        logger.info(f'Reshaping features into spatial maps for session {sessionName} in partition {partNames[partIdx]}')
        for i in range(len(tx1_features)):
            tx1_features[i] = tx1_features[i][:,ROWS]
            spikePow_features[i] = spikePow_features[i][:,ROWS]
            input_features.append(np.stack([tx1_features[i], spikePow_features[i]], axis=-1) ) #shape (T, 16, 8, 2)

        #pad until max_seq_len
        logger.info(f'Padding sequences to max sequence length: {max_seq_len}')
        inputMasks = []
        for i in range(len(input_features)):
            seq_len = input_features[i].shape[0]
            if seq_len < max_sequence_len:
                pad_width = ((0, max_sequence_len - seq_len), (0, 0), (0, 0), (0, 0))
                input_features[i] = np.pad(input_features[i], pad_width, mode='constant', constant_values=0)
                frame_lens[i] = seq_len
                mask = np.zeros(max_sequence_len, dtype=np.bool_)
                mask[:seq_len] = True
                inputMasks.append(mask)

        #convert to tfRecord file
        session_data = {
            'inputFeatures': input_features,
            'brainfeatureMasks': inputMasks,
            'phonemeTokens': phoneme_tokens,
            'phonemeMasks': phoneme_masks,
            'transcriptions': transcriptions,
            'frameLens': frame_lens
        }
        logger.info(f'Preprocessed {len(input_features)} trials for session {sessionName} in partition {partNames[partIdx]}')
        os.makedirs(outputFolder, exist_ok=True)
        output_path = os.path.join(outputFolder, f'{sessionName}_data.pt')
        torch.save(session_data, output_path)
        logger.info(f'Saved preprocessed data to {output_path}')

def g2p_transcription(sentence):
    """
    Convert a sentence into its phoneme representation using g2p_en.
    Args:
        sentence (str): Input sentence.
    Returns:
        list: List of phonemes.
    """
    tokenized_sentence = []
    g2p = G2p()
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)  # Remove punctuation except hyphens and apostrophes
    sentence = sentence.replace('--', '').lower()
    phonemes = g2p(sentence)
    for phoneme in phonemes:
        if phoneme not in PHONE_TO_ID:
            logger.warning(f'Phoneme {phoneme} not in PHONE_TO_ID mapping.')
        else:
            tokenized_sentence.append(PHONE_TO_ID[phoneme])
    tokenized_sentence.append(PHONE_TO_ID['<eos>'])
    return phonemes

if __name__ == "__main__":

    if TOLERABLE_SEQ_LEN is None:
        logger.info(f"Default tolerable sequence length not found, using percentile: {TOLERABLE_SEQ_PERCENTILE}")
        max_sequence_len, seq_lengths = find_max_seq_len(COMPETITION_DATA_DIR,
                                                        tolerable_len=TOLERABLE_SEQ_LEN,
                                                        tolerable_percentile=TOLERABLE_SEQ_PERCENTILE)
        logger.info(f'Determined max_seq_len: {max_sequence_len}')
    else:
        max_sequence_len = int(TOLERABLE_SEQ_LEN)
        logger.info(f'Using provided tolerable_seq_len: {max_sequence_len}')


    