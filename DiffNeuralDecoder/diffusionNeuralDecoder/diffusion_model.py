import torch
import torch.nn as nn
import numpy as np
import math

class BrainConvolutionalEncoder(nn.Module):
    """
    Docstring for BrainConvolutionalEncoder
    Convolutional Encoder for processing brain embeddings, takes the BxSx2x16x8 input and encodes it into a B x sequence_encoded_dim x z_brain_dim representation.
    The dimensions for each datapoint are 2 channels (tx1, spikePow) with 16x8 spatial dimensions representing the electrode grid for 6v region.
    For training speed for cross-attention conditioning in the Phoneme DiT model, default behaviour sets sequence_encoded_dim == phoneme_encoded_dim.
    However, these can be set differently for ablation studies, such as setting sequence_encoded_dim == S to see if higher capacity brain embeddings help.
    """
    def __init__(self, 
                 input_channels=2, 
                 sequence_encoded_dim=128, 
                 z_brain_dim=256, use_layer_norm = True,
                 mlp_num_hidden_layers = 2):
        super(BrainConvolutionalEncoder, self).__init__()
        self.input_channels = input_channels
        self.sequence_encoded_dim = sequence_encoded_dim
        self.z_brain_dim = z_brain_dim
        self.use_layer_norm = use_layer_norm

        # Convolutional layers to process spatial dimensions
        self.conv_layers = nn.Sequential( #(b, s, 2, 16, 8)
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=2, padding=1), # (b, s, 16, 8, 4)
            # nn.BatchNorm2d(16),
            nn.GroupNorm(num_groups=8, num_channels=16), # using group norm instead of batch norm for better performance on smaller batch sizes
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2), # (b, s, 16, 8, 4)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # (b, s, 32, 4, 2)
            nn.GroupNorm(num_groups=8, num_channels=32), # using group norm instead of batch norm for better performance on smaller batch sizes
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2), # (b, s, 32, 4, 2)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # (b, s, 64, 4, 2)
            nn.GroupNorm(num_groups=8, num_channels=64), # using group norm instead of batch norm for better performance on smaller batch sizes
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)) # (b, s, 64, 1, 1)
        )

        # Convolutional Layer converstions
        # (b, s, 2, 16, 8)
        # (b, s, 16, 8, 4)
        # (b, s, 32, 4, 2)
        # (b, s, 64, 4, 2)
        # (b, s, 64, 1, 1)

        if mlp_num_hidden_layers > 1:
        # Linear layer to project to desired z_brain_dim
            self.ffn = nn.Sequential(
                nn.Flatten(), # (b, s, 64)
                nn.Linear(64, 128), # (b, s, 128)
                nn.ReLU(inplace=True),
                nn.Dropout(p = 0.2),
                nn.Linear(128, z_brain_dim) # (b, s, z_brain_dim)
                )
        else:
            self.ffn = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, z_brain_dim) # (b, s, z_brain_dim)
            )
        
        #maybe a layer here for better convergence?
        self.ln_self = nn.LayerNorm(z_brain_dim)
        
    def forward(self, x):
        # x.shape = (b, s, 2, 16, 8)
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w) # (b*s, 2, 16, 8)
        x = self.conv_layers(x)
        x = x.view(b, s, -1) # (b, s, 64)
        x = self.ffn(x)

        # decide here if using layer norm or not
        if self.use_layer_norm:
            x = self.ln_self(x)

        return x

class TimeEmbedding(nn.Module):
    pass

class SinPosEmbedding(nn.Module): # use until ROPE is implemented, for simplicity for now
    def __init__(self, max_len, d_model):
        super(SinPosEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x.shape = (b, s, d_model)
        b, s, d_model = x.shape
        positions = torch.arange(0, s, device=x.device).unsqueeze(0).expand(b, s) # (b, s)
        pos_emb = self.pos_embedding(positions) # (b, s, d_model)
        return pos_emb

class PhonemeDiTBlock(nn.Module):
    """
    Docstring for PhonemeDiTBlock
    Single block of the Phoneme Diffusion-Transformer model
    This block works as a forward pass for a single transformer block to operate Phoneme to Phoneme diffusion.
    The block flows through a self-attention layer, optional cross-attention layer (for conditioning), and a feed-forward network (MLP).
    The cross-attention layer will be used to condition in the brain embeddings during fine-tuning for transfer learning the brain-to-text task.
    Args:
        hidden_dim (int): Dimension of the hidden representations.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of the MLP hidden dimension to the hidden_dim (default set to 4 according to online as best practice)
        cross_attention (bool): Whether to include cross-attention for conditioning.
    """
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, use_cross_attention=False, cond_dim=None):
        super(PhonemeDiTBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        if cond_dim is not None:
            kdim = cond_dim
            vdim = cond_dim
            self.cond_dim = cond_dim
        else:
            kdim = hidden_dim
            vdim = hidden_dim
            self.cond_dim = hidden_dim

        # whether or not to use cross-attention for conditioning
        self.use_cross_attention = use_cross_attention

         # LayerNorm layers
        self.ln_self = nn.LayerNorm(hidden_dim)
        self.ln_cross = nn.LayerNorm(hidden_dim)
        self.ln_ff = nn.LayerNorm(hidden_dim)

        # Multi-head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Cross-Attention (also can ablate later using Adaptive Layer Norm for conditioning instead of cross-attention)
        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, kdim=kdim, vdim=vdim, num_heads=num_heads, batch_first=True)

        # MLP
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            #for speed and efficiency, can ablate later for higher accuracy
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden_dim, hidden_dim)
        )

    def forward(self, x, x_mask=None, cond_sequence=None, cond_mask=None):
        #inverting masks for padding
        if x_mask is not None:
            x_mask = ~x_mask
        if cond_mask is not None:
            cond_mask = ~cond_mask

        # Self-Attention block
        x_norm = self.ln_self(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=x_mask)
        x = x + attn_output

        # Cross-Attention block (if conditioning is provided)
        if self.use_cross_attention and cond_sequence is not None:
            x_norm = self.ln_cross(x)
            cross_attn_output, _ = self.cross_attn(x_norm, cond_sequence, cond_sequence, key_padding_mask=cond_mask)
            x = x + cross_attn_output

        # Feed-Forward block
        x_norm = self.ln_ff(x)
        ff_output = self.mlp(x_norm)
        x = x + ff_output
        return x
        

class PhonemeDiT(nn.Module):
    pass
