
# Libraries
import logging
import torch
import argparse
import logging
import os

# Modules
from diffusion_model import PhonemeDiT



logging.basicConfig(
    filename='app.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """
    Primary Training Script
    """

    assert torch.cuda._is_available(), "Using a GPU"