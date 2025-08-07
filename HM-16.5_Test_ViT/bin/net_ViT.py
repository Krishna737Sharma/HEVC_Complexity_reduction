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

        # Create temporary files
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

            # Prepare command for Python virtual environment (.venv)
            bridge_script = os.path.join(self.vit_repo_path, 'vit_inference_bridge.py')
            python_path = os.path.join(self.vit_repo_path, '.venv', 'bin', 'python')

            cmd = [
                python_path,  # Use the .venv Python interpreter
                bridge_script,
                '--input_file', input_path,
                '--output_file', output_path,
                '--model_path', self.model_path,
                '--qp', str(qp_value)
            ]

            print("Running ViT prediction command...")

            # Execute ViT prediction - FIXED for Python 3.5
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                                    cwd=self.vit_repo_path)

            if result.returncode != 0:
                print("ViT stderr: {}".format(result.stderr))
                print("ViT stdout: {}".format(result.stdout))
                raise Exception("ViT prediction failed: {}".format(result.stderr))

            print("ViT stdout: {}".format(result.stdout))

            # Load predictions
            predictions = np.load(output_path)
            print("ViT output shape: {}".format(predictions.shape))
            return predictions

        finally:
            # Clean up temporary files
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except:
                pass


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

    # Split predictions into 64x64, 32x32, and 16x16 components
    y_conv_flat_64 = predictions[:, 0:1]  # First 1 value for 64x64
    y_conv_flat_32 = predictions[:, 1:5]  # Next 4 values for 32x32
    y_conv_flat_16 = predictions[:, 5:21]  # Last 16 values for 16x16

    return y_conv_flat_64, y_conv_flat_32, y_conv_flat_16
