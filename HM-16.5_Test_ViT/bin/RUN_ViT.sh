#!/bin/bash
# Move to this script's directory so relative paths work
cd "$(dirname "$0")"

# Clean old output files if re-running
rm -f str_ViT.bin rec_ViT.yuv

# MAIN COMMAND: Run the encoder with ViT integration
./TAppEncoderStatic \
    -c encoder_intra_main.cfg \
    -c encoder_yuv_source.cfg

# You can add echo statements for status messages if desired
echo "HEVC Encoding with ViT-based partitioning completed."
