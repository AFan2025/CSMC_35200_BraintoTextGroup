import torch
from torch import nn
import math

from .augmentations import GaussianSmoothing


class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # GRU layers
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder with multi-head self-attention.
    Replaces GRU with transformer encoder architecture.
    """
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0.1,
        device="cuda",
        strideLen=4,
        kernelLen=32,
        gaussianSmoothWidth=0,
        bidirectional=False,  # Kept for compatibility, not used in transformer
        nhead=8,
        dim_feedforward=2048,
    ):
        super(TransformerDecoder, self).__init__()
        
        # Model parameters
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        
        # Preprocessing layers (same as GRU model)
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        
        # Day-specific transformation (domain adaptation)
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        
        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)
        
        # Input projection: reduce dimensionality from neural_dim to hidden_dim
        # This is crucial to make transformer computationally efficient
        self.input_projection = nn.Linear(neural_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            hidden_dim, max_len=5000, dropout=dropout
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm architecture (more stable)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layer_dim,
        )
        
        # Output projection to phoneme classes
        self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate schemes."""
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.fc_decoder_out.weight)
        nn.init.zeros_(self.fc_decoder_out.bias)
    
    def forward(self, neuralInput, dayIdx):
        """
        Args:
            neuralInput: [batch_size, time_steps, neural_dim]
            dayIdx: [batch_size] - day indices for domain adaptation
        
        Returns:
            seq_out: [batch_size, output_time_steps, n_classes + 1]
        """
        # Apply Gaussian smoothing (optional preprocessing)
        if self.gaussianSmoothWidth > 0:
            neuralInput = torch.permute(neuralInput, (0, 2, 1))
            neuralInput = self.gaussianSmoother(neuralInput)
            neuralInput = torch.permute(neuralInput, (0, 2, 1))
        
        # Apply day-specific transformation (domain adaptation)
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)
        
        # Apply downsampling via striding to match GRU sequence length
        # This is CRITICAL for comparable CTC performance
        # Extract every strideLen-th frame (default: every 4th frame)
        transformedNeural = transformedNeural[:, ::self.strideLen, :]
        
        # Project to hidden dimension
        x = self.input_projection(transformedNeural)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer encoder
        # Note: Transformers are inherently bidirectional (full self-attention)
        transformer_out = self.transformer_encoder(x)
        
        # Project to output classes
        seq_out = self.fc_decoder_out(transformer_out)
        
        return seq_out
