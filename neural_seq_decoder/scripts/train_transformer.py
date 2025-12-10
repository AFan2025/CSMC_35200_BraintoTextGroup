modelName = 'speechTransformer1'

args = {}
args['outputDir'] = 'C:\\Users\\Sammy\\repos\\CSMC_35200_BraintoTextGroup\\speech_logs\\' + modelName
args['datasetPath'] = 'C:\\Users\\Sammy\\repos\\CSMC_35200_BraintoTextGroup\\NeuralDecoder\\data\\ptDecoder_ctc'
args['modelType'] = 'transformer'  # Use transformer instead of GRU

# Sequence parameters
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 16

# Learning rate schedule
args['lrStart'] = 0.1  # More conservative LR with gradient clipping
args['lrEnd'] = 0.1
args['nBatch'] = 10000

# Model architecture
args['nUnits'] = 512  # Reduce to 512 for stability (8 heads × 64 dims)
args['nLayers'] = 4  # Start with fewer layers for faster training
args['nhead'] = 8  # Number of attention heads (must divide nUnits evenly)
args['dim_feedforward'] = 2048  # FFN inner dimension

# Data parameters
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256

# Regularization
args['dropout'] = 0.2  # Increase dropout for better regularization
args['l2_decay'] = 1e-5

# Data augmentation
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0

# Striding parameters (not used in transformer, but kept for compatibility)
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False  # Not applicable for transformer (self-attention is bidirectional by default)

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
