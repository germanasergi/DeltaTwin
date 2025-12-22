import os
import torch
import random
import numpy as np
from loguru import logger
import segmentation_models_pytorch as smp
#import torch.nn as nn



def define_model(
    name,
    encoder_name,
    out_channels=3,
    in_channel=3,
    encoder_weights=None,
    activation=None,
):
    # Get the model class dynamically based on name
    try:
        # Get the model class from segmentation_models_pytorch
        ModelClass = getattr(smp, name)

        # Create the model
        model = ModelClass(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channel,
            classes=out_channels,
            decoder_attention_type="scse",
            activation=None,
        )

        # Add ReLU activation after the model
        if activation == "relu":
            model = nn.Sequential(
                model,
                nn.ReLU()
            )
        if activation == "sigmoid":
            model = nn.Sequential(
                model,
                nn.Sigmoid()
            )



        return model
    
    except AttributeError:
        # If the model name is not found in the library
        raise ValueError(f"Model '{name}' not found in segmentation_models_pytorch. Available models: {dir(smp)}")
    

def load_model_weights(config, ckpt, device):

    model = define_model(
        name=config['MODEL']['model_name'],
        encoder_name=config['MODEL']['encoder_name'],
        encoder_weights = config['MODEL']['encoder_weights'],
        in_channel=len(config['DATASET']['bands']),
        out_channels=config['MODEL']['num_classes'],
        activation=config['MODEL']['activation'])

    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model