from .speechDataset import BrainToTextDataset

def getDataset(datasetName):
    if datasetName == 'BrainToText':
        return BrainToTextDataset
    else:
        raise ValueError(f"Unknown dataset: {datasetName}")