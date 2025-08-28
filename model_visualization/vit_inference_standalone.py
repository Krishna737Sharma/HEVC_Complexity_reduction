# vit_inference_standalone.py
# Run this in your ViT repository (PyTorch environment)

import torch
import numpy as np
import sys
import os

# Import your ViT model (since you're in the same directory)
from Vit_model import VisionTransformer


def run_vit_inference():
    print("Loading ground truth data...")
    # Load ground truth data (copy this file from HEVC repo)
    gt_data = np.load('ground_truth.npy', allow_pickle=True).item()

    print("Loading ViT model...")
    # Load ViT model
    model = VisionTransformer(
        img_size=64, patch_size=8, in_chans=1, num_classes=21,
        embed_dim=196, depth=5, num_heads=4, mlp_ratio=4.0, dropout=0.1
    )

    # Update this path to your actual model file
    model_path = '/home/ai-iitkgp/PycharmProjects/HEVC_Intra_Models-ViT/ViT_2.3M/best_vit_model.pth'  # or wherever your .pth file is
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Running ViT inference...")
    # Prepare input - normalize the image like in your training
    image_tensor = torch.FloatTensor(gt_data['image']).permute(2, 0, 1).unsqueeze(0)  # [1, 1, 64, 64]
    qp_value = gt_data['qp'][0] if isinstance(gt_data['qp'], np.ndarray) else gt_data['qp']
    qp_tensor = torch.FloatTensor([qp_value / 51.0])  # Normalize QP like in training

    # Run inference
    with torch.no_grad():
        output = model(image_tensor, qp_tensor)
        pred_64 = output[0, 0:1].cpu().numpy()
        pred_32 = output[0, 1:5].cpu().numpy()
        pred_16 = output[0, 5:21].cpu().numpy()

    # Save results
    np.save('vit_predictions.npy', {
        'pred_64': pred_64,
        'pred_32': pred_32,
        'pred_16': pred_16
    })

    print("ViT predictions saved to vit_predictions.npy")
    print(f"Prediction shapes - 64: {pred_64.shape}, 32: {pred_32.shape}, 16: {pred_16.shape}")


if __name__ == "__main__":
    run_vit_inference()
