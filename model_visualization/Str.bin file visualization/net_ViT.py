# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import subprocess
import os
import tempfile

IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_EXT_FEATURES = 1
NUM_LABEL_BYTES = 16


class ViTPredictor:
    def __init__(self, vit_repo_path, model_path):
        self.vit_repo_path = vit_repo_path
        self.model_path = model_path

    def predict_partitions(self, cu_patches, qp_value):
        """Send CU patches to ViT model and get predictions"""
        input_file = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        output_file = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        input_path = input_file.name
        output_path = output_file.name
        input_file.close()
        output_file.close()

        try:
            # Save input data
            np.save(input_path, cu_patches)
            print("ViT input shape: {}, QP: {}".format(cu_patches.shape, qp_value))

            # Prepare command
            bridge_script = os.path.join(self.vit_repo_path, 'vit_inference_bridge.py')
            python_path = os.path.join(self.vit_repo_path, '.venv', 'bin', 'python')

            # Add environment variables for consistency
            env = os.environ.copy()
            env['PYTHONPATH'] = self.vit_repo_path

            cmd = [
                python_path,
                bridge_script,
                '--input_file', input_path,
                '--output_file', output_path,
                '--model_path', self.model_path,
                '--qp', str(int(qp_value))  # Ensure integer QP
            ]

            print("Running command: {}".format(' '.join(cmd)))
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, cwd=self.vit_repo_path, env=env)

            if result.returncode != 0:
                print("ViT prediction failed!")
                print("STDERR:", result.stderr)
                print("STDOUT:", result.stdout)
                raise Exception("ViT prediction failed: {}".format(result.stderr))

            # Load predictions
            if not os.path.exists(output_path):
                raise Exception("Output file not created: {}".format(output_path))

            predictions = np.load(output_path)
            print("ViT output shape: {}, QP used: {}".format(predictions.shape, qp_value))

            # Validate prediction range
            if np.any(predictions > 1.0) or np.any(predictions < 0.0):
                print("WARNING: Predictions outside [0,1] range!")

            return predictions

        finally:
            # Clean up
            for temp_file in [input_path, output_path]:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass


def hierarchical_mask_and_bias(predictions, threshold_64=0.4, threshold_32=0.4, bias_factor=0.85):
    """
    Apply hierarchical constraints with new logic:
    32x32 splits only if MORE THAN ONE of its 16x16 children want to split
    """
    # Copy to avoid overwriting original
    masked = predictions.copy()
    batch_size = masked.shape[0]

    # Extract level outputs
    out_64 = masked[:, 0:1]  # [B, 1] - 64x64 decisions
    out_32 = masked[:, 1:5]  # [B, 4] - 32x32 decisions
    out_16 = masked[:, 5:21]  # [B, 16] - 16x16 decisions

    # Mask 32-level decisions if 64-level doesn't split
    mask_64 = (out_64 > threshold_64).astype(np.float32)
    masked[:, 1:5] = out_32 * mask_64

    # NEW LOGIC: Mask 16-level based on 32-level with "more than one child" rule
    index_mapping = [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]

    for batch_idx in range(batch_size):
        for i in range(4):  # For each 32x32 quadrant
            # Skip if parent 64x64 doesn't want to split
            if mask_64[batch_idx, 0] == 0:
                # Zero out this 32x32 and all its children
                masked[batch_idx, i + 1] = 0
                for j in range(4):
                    idx_16 = 5 + index_mapping[i][j]
                    masked[batch_idx, idx_16] = 0
                continue

            # Count how many 16x16 children of this 32x32 want to split
            children_wanting_split = 0
            for j in range(4):
                idx_16 = 5 + index_mapping[i][j]
                if predictions[batch_idx, idx_16] > threshold_32:
                    children_wanting_split += 1

            # NEW RULE: Allow 32x32 split only if MORE THAN ONE child wants to split
            if children_wanting_split > 1:
                # Keep the 32x32 split decision and its children as predicted
                pass  # No change needed
            else:
                # Block this 32x32 from splitting and zero out all its children
                masked[batch_idx, i + 1] = 0
                for j in range(4):
                    idx_16 = 5 + index_mapping[i][j]
                    masked[batch_idx, idx_16] = 0

    # Apply conservative bias to reduce excessive splitting
    biased = masked * bias_factor
    return biased


# Global ViT predictor instance
vit_predictor = None


def initialize_vit_predictor(vit_repo_path, model_path):
    """Initialize the global ViT predictor"""
    global vit_predictor
    vit_predictor = ViTPredictor(vit_repo_path, model_path)
    print("ViT predictor initialized with model: {}".format(model_path))


def net_vit_predictions(input_batch, qp_value):
    """Get ViT predictions for batch of CU patches"""
    global vit_predictor
    if vit_predictor is None:
        raise Exception("ViT predictor not initialized. Call initialize_vit_predictor() first.")

    # Get ViT predictions - shape should be [batch_size, 21]
    predictions = vit_predictor.predict_partitions(input_batch, qp_value)

    # *** KEY CHANGE: Apply hierarchical masking and conservative bias ***
    predictions = hierarchical_mask_and_bias(predictions, threshold_64=0.4, threshold_32=0.4, bias_factor=0.85)

    # Split predictions into 64x64, 32x32, and 16x16 components
    y_conv_flat_64 = predictions[:, 0:1]  # First 1 value for 64x64
    y_conv_flat_32 = predictions[:, 1:5]  # Next 4 values for 32x32
    y_conv_flat_16 = predictions[:, 5:21]  # Last 16 values for 16x16

    return y_conv_flat_64, y_conv_flat_32, y_conv_flat_16