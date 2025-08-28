# clean_visualize_partitions.py (Python 3.5 compatible)
# Clean visualization without text overlay

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
        # Draw split lines with specified color and thickness
        cv2.line(image, (x_offset, y_offset + half_size),
                 (x_offset + size, y_offset + half_size), color, thickness)
        cv2.line(image, (x_offset + half_size, y_offset),
                 (x_offset + half_size, y_offset + size), color, thickness)

        # Recurse for four quadrants
        draw_partitions(image, predictions, x_offset, y_offset, half_size, color, thickness)
        draw_partitions(image, predictions, x_offset + half_size, y_offset, half_size, color, thickness)
        draw_partitions(image, predictions, x_offset, y_offset + half_size, half_size, color, thickness)
        draw_partitions(image, predictions, x_offset + half_size, y_offset + half_size, half_size, color, thickness)


def create_clean_visualization():
    print("Loading prediction files...")

    # Load available data
    gt_data = np.load('ground_truth.npy', allow_pickle=True).item()
    cnn_data = np.load('cnn_predictions.npy', allow_pickle=True).item()

    print("Creating clean visualizations...")

    # Prepare base image
    if len(gt_data['image'].shape) == 3:
        base_image = (gt_data['image'][:, :, 0] * 255).astype(np.uint8)
    else:
        base_image = (gt_data['image'] * 255).astype(np.uint8)

    # Convert to color
    base_image_color = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)

    # Scale factor for better visibility
    scale = 8  # Larger scale for better detail

    # Ground Truth - Green lines (correct partitions)
    gt_vis = base_image_color.copy()
    draw_partitions(gt_vis, (gt_data['pred_64'], gt_data['pred_32'], gt_data['pred_16']),
                   0, 0, 64, color=(0, 255, 0), thickness=1)  # Green, thin lines
    gt_vis_scaled = cv2.resize(gt_vis, (64 * scale, 64 * scale), interpolation=cv2.INTER_NEAREST)

    # CNN - Red lines (model predictions)
    cnn_vis = base_image_color.copy()
    draw_partitions(cnn_vis, (cnn_data['pred_64'], cnn_data['pred_32'], cnn_data['pred_16']),
                   0, 0, 64, color=(0, 0, 255), thickness=1)  # Red, thin lines
    cnn_vis_scaled = cv2.resize(cnn_vis, (64 * scale, 64 * scale), interpolation=cv2.INTER_NEAREST)

    # Save individual images without titles
    cv2.imwrite('ground_truth_clean.png', gt_vis_scaled)
    cv2.imwrite('cnn_prediction_clean.png', cnn_vis_scaled)

    # Create side-by-side comparison
    comparison = cv2.hconcat([gt_vis_scaled, cnn_vis_scaled])
    cv2.imwrite('clean_comparison.png', comparison)

    # Create overlay version (both partitions on same image)
    overlay_vis = base_image_color.copy()
    # Draw ground truth in green (thinner)
    draw_partitions(overlay_vis, (gt_data['pred_64'], gt_data['pred_32'], gt_data['pred_16']),
                   0, 0, 64, color=(0, 255, 0), thickness=1)
    # Draw CNN predictions in red (slightly thicker to see difference)
    draw_partitions(overlay_vis, (cnn_data['pred_64'], cnn_data['pred_32'], cnn_data['pred_16']),
                   0, 0, 64, color=(0, 0, 255), thickness=1)
    overlay_vis_scaled = cv2.resize(overlay_vis, (64 * scale, 64 * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('overlay_comparison.png', overlay_vis_scaled)

    print("Clean visualizations saved:")
    print("- ground_truth_clean.png (Green lines = correct partitions)")
    print("- cnn_prediction_clean.png (Red lines = CNN predictions)")
    print("- clean_comparison.png (Side by side)")
    print("- overlay_comparison.png (Both overlaid - Green=GT, Red=CNN)")

    # Print statistics
    print("\nPartition Statistics:")
    print("Ground Truth splits - 64x64: {}, 32x32: {}, 16x16: {}".format(
        int(gt_data['pred_64'].sum()),
        int(gt_data['pred_32'].sum()),
        int(gt_data['pred_16'].sum())
    ))
    print("CNN predictions - 64x64: {:.1f}, 32x32: {:.1f}, 16x16: {:.1f}".format(
        cnn_data['pred_64'].sum(),
        cnn_data['pred_32'].sum(),
        cnn_data['pred_16'].sum()
    ))


if __name__ == "__main__":
    create_clean_visualization()
