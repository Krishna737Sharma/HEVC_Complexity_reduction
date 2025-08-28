# vit_frame_inference.py
# Run in ViT repository (PyTorch environment)

import torch
import numpy as np
import os
import math
from Vit_model import VisionTransformer

# Configuration
VIT_MODEL_PATH = '/home/ai-iitkgp/PycharmProjects/HEVC_Intra_Models-ViT/ViT_2.3M/best_vit_model.pth'
FRAME_INDICES = [4, 5]  # Same as in extraction script
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_vit_model():
    """Load ViT model with proper error handling."""
    print("Loading ViT model...")

    if not os.path.exists(VIT_MODEL_PATH):
        raise FileNotFoundError("ViT model not found: {}".format(VIT_MODEL_PATH))

    try:
        model = VisionTransformer(
            img_size=64, patch_size=8, in_chans=1, num_classes=21,
            embed_dim=196, depth=5, num_heads=4, mlp_ratio=4.0, dropout=0.1
        )

        checkpoint = torch.load(VIT_MODEL_PATH, map_location=DEVICE, weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(DEVICE)
        model.eval()
        print("ViT model loaded on device: {}".format(DEVICE))
        return model

    except Exception as e:
        print("Error loading ViT model: {}".format(e))
        raise


def load_ctus_from_directory(ctu_dir):
    """Load CTUs from directory with proper sorting and error handling."""
    if not os.path.exists(ctu_dir):
        print("CTU directory not found: {}".format(ctu_dir))
        return None

    # Get all CTU text files
    ctu_files = [f for f in os.listdir(ctu_dir) if f.startswith('ctu_') and f.endswith('.txt')]

    if len(ctu_files) == 0:
        print("No CTU files found in: {}".format(ctu_dir))
        return None

    # Sort by CTU index (extract number from filename)
    def extract_ctu_index(filename):
        try:
            # Extract index from filename like 'ctu_0001_pos_0_0.txt'
            parts = filename.split('_')
            return int(parts[1])
        except:
            return 0

    ctu_files.sort(key=extract_ctu_index)
    print("Found {} CTU files in {}".format(len(ctu_files), ctu_dir))

    # Load CTUs
    ctus = []
    ctu_positions = []

    for ctu_file in ctu_files:
        try:
            ctu_path = os.path.join(ctu_dir, ctu_file)
            ctu = np.loadtxt(ctu_path, dtype=np.uint8)

            # Ensure CTU is 64x64
            if ctu.shape != (64, 64):
                print("Warning: CTU {} has shape {}, expected (64, 64)".format(ctu_file, ctu.shape))
                continue

            ctus.append(ctu)

            # Extract position from filename if available
            try:
                parts = ctu_file.split('_')
                if len(parts) >= 5:
                    x_pos = int(parts[3])
                    y_pos = int(parts[4].split('.')[0])
                    ctu_positions.append((x_pos, y_pos))
                else:
                    ctu_positions.append((0, 0))
            except:
                ctu_positions.append((0, 0))

        except Exception as e:
            print("Error loading CTU file {}: {}".format(ctu_file, e))
            continue

    print("Successfully loaded {} CTUs".format(len(ctus)))
    return ctus, ctu_positions


def get_vit_predictions_for_frame(frame_idx, model, qp_value=32):
    """Get ViT predictions for all CTUs in a frame."""
    ctu_dir = 'frame_{}_ctus'.format(frame_idx)

    # Load CTUs
    ctu_data = load_ctus_from_directory(ctu_dir)
    if ctu_data is None:
        return None

    ctus, ctu_positions = ctu_data
    num_ctus = len(ctus)

    print("Processing {} CTUs with ViT (QP={})...".format(num_ctus, qp_value))

    predictions = []
    batch_size = 16
    num_batches = int(math.ceil(num_ctus / float(batch_size)))

    with torch.no_grad():
        for i in range(0, num_ctus, batch_size):
            end_idx = min(i + batch_size, num_ctus)
            batch_ctus = ctus[i:end_idx]
            batch_size_actual = len(batch_ctus)

            # Prepare batch - normalize pixel values to [0, 1]
            batch_images = np.array(batch_ctus, dtype=np.float32) / 255.0
            batch_images = batch_images.reshape(batch_size_actual, 1, 64, 64)  # [B, C, H, W]
            batch_tensor = torch.FloatTensor(batch_images).to(DEVICE)

            # Normalize QP value to [0, 1] range
            qp_tensor = torch.full((batch_size_actual,), qp_value / 51.0, dtype=torch.float32).to(DEVICE)

            try:
                # Get predictions
                output = model(batch_tensor, qp_tensor)
                output = output.cpu().numpy()

                # Split predictions according to CU sizes
                for j in range(batch_size_actual):
                    pred_64 = output[j, 0:1]  # 1 prediction for 64x64
                    pred_32 = output[j, 1:5]  # 4 predictions for 32x32
                    pred_16 = output[j, 5:21]  # 16 predictions for 16x16

                    predictions.append({
                        'pred_64': pred_64,
                        'pred_32': pred_32,
                        'pred_16': pred_16,
                        'position': ctu_positions[i + j]
                    })

            except Exception as e:
                print("Error in ViT inference batch {}: {}".format(i // batch_size + 1, e))
                # Add dummy predictions for failed batch
                for j in range(batch_size_actual):
                    predictions.append({
                        'pred_64': np.zeros(1),
                        'pred_32': np.zeros(4),
                        'pred_16': np.zeros(16),
                        'position': ctu_positions[i + j] if i + j < len(ctu_positions) else (0, 0)
                    })

            # Progress update
            if (i // batch_size + 1) % 5 == 0 or (i // batch_size + 1) == num_batches:
                print("Processed {}/{} ViT batches".format(i // batch_size + 1, num_batches))

    return predictions


def save_vit_predictions(frame_idx, predictions):
    """Save ViT predictions in multiple formats."""
    if predictions is None or len(predictions) == 0:
        return

    # Create output directory
    output_dir = 'frame_{}_vit_analysis'.format(frame_idx)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save as text file (human readable)
    txt_file = '{}/vit_predictions.txt'.format(output_dir)
    with open(txt_file, 'w') as f:
        f.write("Frame {} ViT Predictions\n".format(frame_idx))
        f.write("Total CTUs: {}\n\n".format(len(predictions)))

        for i, pred in enumerate(predictions):
            pos = pred['position']
            f.write("CTU {:04d} at position ({}, {}):\n".format(i, pos[0], pos[1]))
            f.write("  64x64: {} (mean: {:.4f})\n".format(pred['pred_64'], pred['pred_64'].mean()))
            f.write("  32x32: {} (mean: {:.4f})\n".format(pred['pred_32'][:3], pred['pred_32'].mean()))
            f.write("  16x16: {} (mean: {:.4f})\n\n".format(pred['pred_16'][:5], pred['pred_16'].mean()))

    # Save as numpy file for further processing
    np.save('{}/vit_predictions.npy'.format(output_dir), predictions)

    # Save summary statistics
    stats_file = '{}/prediction_stats.txt'.format(output_dir)
    with open(stats_file, 'w') as f:
        pred_64_values = [p['pred_64'] for p in predictions]
        pred_32_values = [p['pred_32'] for p in predictions]
        pred_16_values = [p['pred_16'] for p in predictions]

        f.write("ViT Prediction Statistics for Frame {}\n".format(frame_idx))
        f.write("64x64 predictions: min={:.4f}, max={:.4f}, mean={:.4f}\n".format(
            np.min(pred_64_values), np.max(pred_64_values), np.mean(pred_64_values)))
        f.write("32x32 predictions: min={:.4f}, max={:.4f}, mean={:.4f}\n".format(
            np.min(pred_32_values), np.max(pred_32_values), np.mean(pred_32_values)))
        f.write("16x16 predictions: min={:.4f}, max={:.4f}, mean={:.4f}\n".format(
            np.min(pred_16_values), np.max(pred_16_values), np.mean(pred_16_values)))

    print("ViT predictions saved to {}".format(output_dir))


def process_frames():
    """Process all frames with ViT."""
    print("=== ViT Frame Processing ===")

    # Load model
    try:
        model = load_vit_model()
    except Exception as e:
        print("Failed to load ViT model: {}".format(e))
        return

    # Process each frame
    for frame_idx in FRAME_INDICES:
        print("\nProcessing frame {} with ViT...".format(frame_idx))

        vit_preds = get_vit_predictions_for_frame(frame_idx, model, qp_value=32)
        if vit_preds is None:
            print("Skipping frame {} due to prediction error".format(frame_idx))
            continue

        # Save predictions
        save_vit_predictions(frame_idx, vit_preds)
        print("Frame {} ViT analysis complete!".format(frame_idx))

    print("\n=== ViT Processing Complete ===")
    print("Copy the generated 'frame_X_vit_analysis' directories back to HEVC repository for comparison")


def load_from_complete_data(frame_idx):
    """Alternative: Load data from complete_data.npy file."""
    npy_file = 'frame_{}_analysis/complete_data.npy'.format(frame_idx)

    if not os.path.exists(npy_file):
        print("Complete data file not found: {}".format(npy_file))
        return None

    try:
        data = np.load(npy_file, allow_pickle=True).item()
        ctus = data['ctus']
        ctu_positions = data['ctu_positions']
        print("Loaded {} CTUs from complete data file".format(len(ctus)))
        return ctus, ctu_positions
    except Exception as e:
        print("Error loading complete data: {}".format(e))
        return None


if __name__ == "__main__":
    # Check if CTU directories exist
    missing_dirs = []
    for frame_idx in FRAME_INDICES:
        ctu_dir = 'frame_{}_ctus'.format(frame_idx)
        if not os.path.exists(ctu_dir):
            missing_dirs.append(ctu_dir)

    if missing_dirs:
        print("Missing CTU directories:")
        for d in missing_dirs:
            print("  - {}".format(d))
        print("\nCopy CTU directories from HEVC repository first:")
        for frame_idx in FRAME_INDICES:
            print("  cp -r /path/to/hevc/frame_{}_analysis/ctus ./frame_{}_ctus".format(frame_idx, frame_idx))
        print("")

    process_frames()
