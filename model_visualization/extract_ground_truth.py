import sys
import os

sys.path.append('/home/ai-iitkgp/PycharmProjects/HEVC-CNN/HEVC-Complexity-Reduction')  # Update this path
import input_data as input_data
import numpy as np

# Configuration
DATA_SWITCH = 0  # Use full dataset
SAMPLE_INDEX = 100  # Choose which sample to visualize


def extract_single_sample():
    # Initialize data readers
    input_data.DATA_SWITCH = DATA_SWITCH
    data_sets = input_data.read_data_sets()

    # Get a single sample
    images, labels, qps = data_sets.validation.next_batch(1)

    # Process the labels to get hierarchical predictions (same as in your ViT code)
    y_image = labels.reshape(1, 4, 4, 1)

    # Convert to the same format your models output
    y_image_16 = np.maximum(y_image - 2, 0)
    y_image_32 = np.maximum(np.mean(y_image.reshape(1, 2, 2, 4), axis=3, keepdims=True) - 1, 0) - \
                 np.maximum(np.mean(y_image.reshape(1, 2, 2, 4), axis=3, keepdims=True) - 2, 0)
    y_image_64 = np.maximum(np.mean(y_image) - 0, 0) - np.maximum(np.mean(y_image) - 1, 0)

    # Flatten
    gt_64 = y_image_64.flatten()
    gt_32 = y_image_32.flatten()
    gt_16 = y_image_16.flatten()

    # Save
    np.save('ground_truth.npy', {
        'image': images[0],
        'qp': qps[0],
        'pred_64': gt_64,
        'pred_32': gt_32,
        'pred_16': gt_16
    })
    print("Ground truth saved to ground_truth.npy")


if __name__ == "__main__":
    extract_single_sample()
