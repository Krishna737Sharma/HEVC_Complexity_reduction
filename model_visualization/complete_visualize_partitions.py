# complete_visualize_partitions.py (Python 3.5 compatible)
# Complete visualization including ViT

import cv2
import numpy as np
import os


def draw_partitions(image, predictions, x_offset, y_offset, size, color=(0, 255, 0), thickness=1):
    """Draw partition lines recursively with customizable color and thickness."""
    if size < 16:
        return

    should_split = False
    if size == 64:
        if len(predictions[0]) > 0 and predictions[0][0] > 0.5:
            should_split = True
    elif size == 32:
        quad_idx = ((y_offset // 32) % 2) * 2 + ((x_offset // 32) % 2)
        if quad_idx < len(predictions[1]) and predictions[1][quad_idx] > 0.5:
            should_split = True
    elif size == 16:
        cu_idx = (y_offset // 16) * 4 + (x_offset // 16)
        if cu_idx < len(predictions[2]) and predictions[2][cu_idx] > 0.5:
            should_split = True

    if should_split:
        half_size = size // 2
        cv2.line(image, (x_offset, y_offset + half_size),
                 (x_offset + size, y_offset + half_size), color, thickness)
        cv2.line(image, (x_offset + half_size, y_offset),
                 (x_offset + half_size, y_offset + size), color, thickness)

        # Recurse
        draw_partitions(image, predictions, x_offset, y_offset, half_size, color, thickness)
        draw_partitions(image, predictions, x_offset + half_size, y_offset, half_size, color, thickness)
        draw_partitions(image, predictions, x_offset, y_offset + half_size, half_size, color, thickness)
        draw_partitions(image, predictions, x_offset + half_size, y_offset + half_size, half_size, color, thickness)


def create_complete_visualization():
    print("Loading prediction files...")

    # Load data
    gt_data = np.load('ground_truth.npy', allow_pickle=True).item()
    cnn_data = np.load('cnn_predictions.npy', allow_pickle=True).item()

    # Check for ViT predictions
    vit_available = (os.path.exists('vit_pred_64.txt') and
                     os.path.exists('vit_pred_32.txt') and
                     os.path.exists('vit_pred_16.txt'))

    if vit_available:
        vit_pred_64 = np.loadtxt('vit_pred_64.txt').reshape(-1)
        vit_pred_32 = np.loadtxt('vit_pred_32.txt').reshape(-1)
        vit_pred_16 = np.loadtxt('vit_pred_16.txt').reshape(-1)
        print("ViT predictions loaded from text files")
    else:
        print("ViT prediction files not found. Run ViT inference first.")

    # Prepare base image
    if len(gt_data['image'].shape) == 3:
        base_image = (gt_data['image'][:, :, 0] * 255).astype(np.uint8)
    else:
        base_image = (gt_data['image'] * 255).astype(np.uint8)

    base_image_color = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    scale = 8

    # Ground Truth - Green
    gt_vis = base_image_color.copy()
    draw_partitions(gt_vis, (gt_data['pred_64'], gt_data['pred_32'], gt_data['pred_16']),
                    0, 0, 64, color=(0, 255, 0), thickness=1)
    gt_vis_scaled = cv2.resize(gt_vis, (64 * scale, 64 * scale), interpolation=cv2.INTER_NEAREST)

    # CNN - Red
    cnn_vis = base_image_color.copy()
    draw_partitions(cnn_vis, (cnn_data['pred_64'], cnn_data['pred_32'], cnn_data['pred_16']),
                    0, 0, 64, color=(0, 0, 255), thickness=1)
    cnn_vis_scaled = cv2.resize(cnn_vis, (64 * scale, 64 * scale), interpolation=cv2.INTER_NEAREST)

    if vit_available:
        # ViT - Blue
        vit_vis = base_image_color.copy()
        draw_partitions(vit_vis, (vit_pred_64, vit_pred_32, vit_pred_16),
                        0, 0, 64, color=(255, 0, 0), thickness=1)  # Blue
        vit_vis_scaled = cv2.resize(vit_vis, (64 * scale, 64 * scale), interpolation=cv2.INTER_NEAREST)

        # Three-way comparison
        comparison = cv2.hconcat([gt_vis_scaled, cnn_vis_scaled, vit_vis_scaled])
        cv2.imwrite('complete_comparison.png', comparison)

        # Triple overlay
        overlay_vis = base_image_color.copy()
        draw_partitions(overlay_vis, (gt_data['pred_64'], gt_data['pred_32'], gt_data['pred_16']),
                        0, 0, 64, color=(0, 255, 0), thickness=1)  # Green = GT
        draw_partitions(overlay_vis, (cnn_data['pred_64'], cnn_data['pred_32'], cnn_data['pred_16']),
                        0, 0, 64, color=(0, 0, 255), thickness=1)  # Red = CNN
        draw_partitions(overlay_vis, (vit_pred_64, vit_pred_32, vit_pred_16),
                        0, 0, 64, color=(255, 0, 0), thickness=1)  # Blue = ViT
        overlay_vis_scaled = cv2.resize(overlay_vis, (64 * scale, 64 * scale), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite('triple_overlay.png', overlay_vis_scaled)

        # Save individual ViT image
        cv2.imwrite('vit_prediction_clean.png', vit_vis_scaled)

        print("Complete visualizations saved:")
        print("- complete_comparison.png (GT | CNN | ViT)")
        print("- triple_overlay.png (All three overlaid)")
        print("- vit_prediction_clean.png (ViT only)")

        # Statistics
        print("\nPartition Statistics Comparison:")
        print("Ground Truth - 64x64: {}, 32x32: {}, 16x16: {}".format(
            int(gt_data['pred_64'].sum()),
            int(gt_data['pred_32'].sum()),
            int(gt_data['pred_16'].sum())
        ))
        print("CNN predictions - 64x64: {:.1f}, 32x32: {:.1f}, 16x16: {:.1f}".format(
            cnn_data['pred_64'].sum(),
            cnn_data['pred_32'].sum(),
            cnn_data['pred_16'].sum()
        ))
        print("ViT predictions - 64x64: {:.1f}, 32x32: {:.1f}, 16x16: {:.1f}".format(
            vit_pred_64.sum(),
            vit_pred_32.sum(),
            vit_pred_16.sum()
        ))
    else:
        print("Run ViT inference first to get complete comparison")


if __name__ == "__main__":
    create_complete_visualization()
