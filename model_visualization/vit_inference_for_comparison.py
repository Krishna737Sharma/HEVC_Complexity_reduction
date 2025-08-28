# vit_inference_for_comparison.py
# Run this in your ViT repository (PyTorch environment)

import torch
import numpy as np
from Vit_model import VisionTransformer


def run_vit_inference():
    print("Loading ground truth data...")
    # Copy the ground_truth.npy from HEVC repo to ViT repo first
    gt_data = np.load('ground_truth.npy', allow_pickle=True).item()

    print("Loading ViT model...")
    model = VisionTransformer(
        img_size=64, patch_size=8, in_chans=1, num_classes=21,
        embed_dim=196, depth=5, num_heads=4, mlp_ratio=4.0, dropout=0.1
    )

    # Update path to your model file
    checkpoint = torch.load('/home/ai-iitkgp/PycharmProjects/HEVC_Intra_Models-ViT/ViT_2.3M/best_vit_model.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Running ViT inference...")
    # Prepare input exactly like in your training
    if len(gt_data['image'].shape) == 3:
        image_tensor = torch.FloatTensor(gt_data['image']).permute(2, 0, 1).unsqueeze(0)  # [1, 1, 64, 64]
    else:
        image_tensor = torch.FloatTensor(gt_data['image']).unsqueeze(0).unsqueeze(0)  # [1, 1, 64, 64]

    qp_value = gt_data['qp'][0] if isinstance(gt_data['qp'], np.ndarray) else gt_data['qp']
    qp_tensor = torch.FloatTensor([qp_value / 51.0])  # Normalize like in training

    with torch.no_grad():
        output = model(image_tensor, qp_tensor)
        pred_64 = output[0, 0:1].cpu().numpy()
        pred_32 = output[0, 1:5].cpu().numpy()
        pred_16 = output[0, 5:21].cpu().numpy()

    # Save as simple text files to avoid numpy compatibility issues
    np.savetxt('vit_pred_64.txt', pred_64, fmt='%.6f')
    np.savetxt('vit_pred_32.txt', pred_32, fmt='%.6f')
    np.savetxt('vit_pred_16.txt', pred_16, fmt='%.6f')

    print("ViT predictions saved as text files")
    print("Statistics - 64x64: {:.1f}, 32x32: {:.1f}, 16x16: {:.1f}".format(
        pred_64.sum(), pred_32.sum(), pred_16.sum()
    ))
    print("Copy vit_pred_*.txt files to your HEVC repository")


if __name__ == "__main__":
    run_vit_inference()
