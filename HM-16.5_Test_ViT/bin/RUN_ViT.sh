#!/bin/bash

# Activate your ViT (PyTorch) Python environment if needed
# source /home/ai-iitkgp/PycharmProjects/HEVC_Intra_Models-ViT/.venv/bin/activate

# Move to this script's directory so relative paths work
cd "$(dirname "$0")"

# Clean old output files if re-running
rm -f str.bin rec.yuv

# MAIN COMMAND: Run the encoder with ViT integration
./TAppEncoderStatic \
    -c encoder_intra_main.cfg \
    -c encoder_yuv_source.cfg

# If your encoder requires the ViT bridge script to be running separately
# you may need to launch it in background before the above command, e.g.:
# python run_vit_bridge.py &

# You can add echo statements for status messages if desired
echo "HEVC Encoding with ViT-based partitioning completed."
