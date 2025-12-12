Using the LibreSpeech Dataset


## Dataset info and Choices
### Competition Dataset:
#### Brain Sequence Bins:
- Max = 919
- Median = 295
- Mean = ~311
- 99th Percentile = 715
Means the Median time per trial is 295*20/1000 = 5.9 seconds

#### Phoneme Bins
- Max = 77
- Median = 24
- Mean = 24.8
- 99th Percentile = 62

#### Character/Words
- Max = 87 Characters
- Median = 29 Characters
- Mean = 29 Characters
- 99th Percentile = 72

### Pretraining Dataset Options:
The choice was to use the LibriSpeech ASR Dataset, the choice was so that the phoneme lenght was on average longer than the BCI Phoneme Pairs. This is because the Pretraining dataset will be used to generate the unconditional model and the BCI Phonemes will be primarily used as finetuning. In a fixed window diffusion model, this would mean that the maximum BCI Phoneme length of the data used should be at most the maximum Phoneme length of the pretraining datapoints.
The LibriSpeech ASR dataset was chosen as it was semantically similar to the BCI dataset and designed for spoken word. General text corpa could've been used such as OpenWebText but there might've been a confounder on the semantic properties of the text data as spoken versus written vernacular and structure are very different. 

We are using the train.clean.360 split from the LibriSpeech set which is 104k rows long.
#### LibriSpeech ASR Character length
- 34.5% 5-63
- 32.4% 63-121
- 17.0% 121-179

Good Range + greater by a large margin of the character lengths. Will likely use max 179 character lenght from this dataset. If training too long, could consider using up to 121 characters. Limiting to below 179 achieves ~84k rows which is 8x our finetuning BCI set. limiting to 121 rows achieves around ~65k which is 6x our finetuning BCI set. 121 is also around 4x our mean character length in the BCI dataset, which is close to ideal due to allowing the model to see a wealth of phoneme lengths without being too long.

For the limits of this project, 121 characters was chosen for compute time using around 65k rows, however, later tests can be made using higher numbers. 
