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
            np.save(input_path, cu_patches)

            bridge_script = os.path.join(self.vit_repo_path, 'vit_inference_bridge.py')
            python_path = os.path.join(self.vit_repo_path, '.venv', 'bin', 'python')
            env = os.environ.copy()
            env['PYTHONPATH'] = self.vit_repo_path

            cmd = [
                python_path, bridge_script,
                '--input_file', input_path,
                '--output_file', output_path,
                '--model_path', self.model_path,
                '--qp', str(int(qp_value))
            ]

            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, cwd=self.vit_repo_path, env=env)

            if result.returncode != 0:
                print("ViT prediction failed!")
                print("STDERR:", result.stderr)
                print("STDOUT:", result.stdout)
                raise Exception("ViT prediction failed: {}".format(result.stderr))

            predictions = np.load(output_path)
            return predictions

        finally:
            for temp_file in [input_path, output_path]:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)


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

    # --- THIS IS THE FIX ---
    # Get the model's raw predictions without applying extra heuristics or biases.
    # This allows for a fair evaluation of the model's learned performance.
    predictions = vit_predictor.predict_partitions(input_batch, qp_value)

    # Split predictions into 64x64, 32x32, and 16x16 components
    y_conv_flat_64 = predictions[:, 0:1]
    y_conv_flat_32 = predictions[:, 1:5]
    y_conv_flat_16 = predictions[:, 5:21]

    return y_conv_flat_64, y_conv_flat_32, y_conv_flat_16

