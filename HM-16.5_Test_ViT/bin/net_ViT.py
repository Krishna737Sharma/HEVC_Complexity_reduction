# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================================================================
#                      CONSTANTS AND CONFIG
# ===================================================================

IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_EXT_FEATURES = 1
NUM_LABEL_BYTES = 16
DEVICE = torch.device('cpu')  # Use CPU for inference in the encoder


# --- START OF ADDED SECTION ---
# This section is added from your net_CNN.py file

def get_thresholds(thr_file):
    # Check if the file exists before trying to open it
    f = open(thr_file, 'r+')
    line = f.readline()
    str_arr = line.split(' ')
    thr_l1_lower = float(str_arr[1])
    thr_l2_lower = float(str_arr[3])
    f.close()
    return thr_l1_lower, thr_l2_lower


# Load the thresholds to be used in the prediction function
THR_L1_LOWER, THR_L2_LOWER = get_thresholds('Thr_info.txt')

# ===================================================================
#                  VISION TRANSFORMER MODEL DEFINITION
#     (Model classes are now included directly in this file)
# ===================================================================

class PatchEmbed(nn.Module):
    """Converts a 2D image into a sequence of flattened patch embeddings."""

    def __init__(self, img_size=64, patch_size=8, in_chans=1, embed_dim=196):
        super(PatchEmbed, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class CustomTransformerEncoderLayer(nn.Module):
    """A custom Transformer Encoder layer that integrates the QP value."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="gelu"):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model + 1, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward + 1, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, src, qp):
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        B, seq_len = src.shape[1], src.shape[0]
        qp_exp = qp.view(1, B, 1).expand(seq_len, B, 1)
        src_cat = torch.cat([src, qp_exp], dim=-1)
        src2 = self.linear1(src_cat)
        src2 = self.activation(src2)
        src2 = self.dropout1(src2)
        src2_cat = torch.cat([src2, qp_exp], dim=-1)
        src2 = self.linear2(src2_cat)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class VisionTransformer(nn.Module):
    """The main Vision Transformer model."""

    def __init__(self, img_size=64, patch_size=8, in_chans=1, num_classes=21,
                 embed_dim=196, depth=5, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        dim_feedforward = int(embed_dim * mlp_ratio)
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                          dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, qp):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = x.transpose(0, 1)
        for layer in self.encoder_layers:
            x = layer(x, qp)
        x = x[0]
        x = self.norm(x)
        logits = self.head(x)
        out = torch.sigmoid(logits)
        return out


# ===================================================================
#               MODIFIED VIT PREDICTOR IMPLEMENTATION
# ===================================================================

class ViTPredictor:
    """
    Manages the ViT model, loading it once and running predictions efficiently.
    This replaces the slow subprocess-based bridge.
    """

    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        print("ViT model loaded directly into memory for efficient inference.")

    def _load_model(self, model_path):
        """Loads the trained ViT model and its weights."""
        model = VisionTransformer(
            img_size=64, patch_size=8, in_chans=1, num_classes=21,
            embed_dim=196, depth=5, num_heads=4
        )
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        return model

    def predict_partitions(self, cu_patches, qp_value):
        """Processes a batch of CU patches and returns raw model predictions."""
        with torch.no_grad():
            batch_size = cu_patches.shape[0]
            # Convert to tensor and normalize pixel values
            input_tensor = torch.from_numpy(cu_patches).to(DEVICE, dtype=torch.float32)
            input_tensor = input_tensor.permute(0, 3, 1, 2) / 255.0
            # Create and normalize QP tensor
            qp_tensor = torch.full((batch_size,), float(qp_value) / 51.0, device=DEVICE)
            # Get predictions from the model
            predictions_tensor = self.model(input_tensor, qp_tensor)
            return predictions_tensor.cpu().numpy()


# Global ViT predictor instance
vit_predictor = None


def initialize_vit_predictor(model_path):
    """
    Initializes the global ViT predictor.
    NOTE: This function now only takes ONE argument.
    """
    global vit_predictor
    vit_predictor = ViTPredictor(model_path)
    print("ViT predictor initialized with model: {}".format(model_path))


# --- START OF MODIFIED SECTION ---

def net_vit_predictions(input_batch, qp_value):
    """Get ViT predictions for a batch of CU patches with CORRECT per-sample hierarchical pruning."""
    global vit_predictor
    if vit_predictor is None:
        raise Exception("ViT predictor not initialized. Call initialize_vit_predictor() first.")

    # Get the model's raw predictions (outputs are NumPy arrays)
    predictions = vit_predictor.predict_partitions(input_batch, qp_value)

    # Split the raw predictions
    y_conv_flat_64 = predictions[:, 0:1]
    y_conv_flat_32 = predictions[:, 1:5]
    y_conv_flat_16 = predictions[:, 5:21]


    return y_conv_flat_64, y_conv_flat_32, y_conv_flat_16
