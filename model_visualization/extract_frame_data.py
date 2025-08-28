import cv2
import numpy as np
import os
import math
import sys

# Add HEVC repo path
sys.path.append('/home/ai-iitkgp/PycharmProjects/HEVC-CNN/HEVC-Complexity-Reduction')

import input_data as input_data
import net_CTU64 as nt
import tensorflow as tf

# Configuration - UPDATE THESE PATHS
YUV_FILE_PATH = '/home/ai-iitkgp/PycharmProjects/HEVC-CNN/test_video/IntraTest_4928x3264.yuv'
FRAME_WIDTH = 4928
FRAME_HEIGHT = 3264
FRAME_INDICES = [4, 5]  # Which frames to analyze
# CORRECTED MODEL PATH - Remove the .dat extension and use the base name
CNN_MODEL_PATH = '/home/ai-iitkgp/PycharmProjects/HEVC-CNN/HEVC-Complexity-Reduction/ETH-CNN_Training_AI/Models/model_20181225_154616_1000000_qp27.dat'
YUV_FORMAT = '420'  # Confirmed as YUV420 from your diagnostic


class FrameExtractor:
    def __init__(self, yuv_file, width, height, yuv_format='420'):
        self.yuv_file = yuv_file
        self.width = width
        self.height = height
        self.yuv_format = yuv_format

        # Calculate frame size based on format
        if yuv_format == '420':
            self.frame_size = width * height * 3 // 2
            self.u_size = width * height // 4
            self.v_size = width * height // 4
        elif yuv_format == '422':
            self.frame_size = width * height * 2
            self.u_size = width * height // 2
            self.v_size = width * height // 2
        elif yuv_format == '444':
            self.frame_size = width * height * 3
            self.u_size = width * height
            self.v_size = width * height
        else:
            raise ValueError("Unsupported YUV format: {}".format(yuv_format))

        self.y_size = width * height

        # Verify file exists and get basic info
        self._verify_file()

    def _verify_file(self):
        """Verify YUV file and print diagnostic information."""
        if not os.path.exists(self.yuv_file):
            raise FileNotFoundError("YUV file not found: {}".format(self.yuv_file))

        file_size = os.path.getsize(self.yuv_file)
        expected_frames = file_size // self.frame_size

        print("YUV File: {}".format(self.yuv_file))
        print("Dimensions: {}x{}".format(self.width, self.height))
        print("Format: YUV{}".format(self.yuv_format))
        print("File size: {:,} bytes".format(file_size))
        print("Frame size: {:,} bytes".format(self.frame_size))
        print("Expected frames: {}".format(expected_frames))
        print("Y channel size: {}".format(self.y_size))
        print("U channel size: {}".format(self.u_size))
        print("V channel size: {}".format(self.v_size))

    def get_frame(self, frame_idx):
        """Extract a single frame from YUV file with proper error handling."""
        try:
            with open(self.yuv_file, 'rb') as f:
                # Check if frame exists
                f.seek(0, 2)  # Go to end
                file_size = f.tell()
                max_frames = file_size // self.frame_size

                if frame_idx >= max_frames:
                    print("Error: Frame {} exceeds available frames (0-{})".format(frame_idx, max_frames - 1))
                    return None

                # Seek to frame position
                frame_offset = frame_idx * self.frame_size
                f.seek(frame_offset)

                print("Extracting frame {} from offset {}".format(frame_idx, frame_offset))

                # Read Y channel (luminance)
                y_data = f.read(self.y_size)
                if len(y_data) != self.y_size:
                    print("Error: Incomplete Y data read: {} vs {}".format(len(y_data), self.y_size))
                    return None

                # Convert to numpy array and reshape
                y_frame = np.frombuffer(y_data, dtype=np.uint8)
                y_frame = y_frame.reshape(self.height, self.width)

                # Verify frame statistics
                print("Frame {} stats: min={}, max={}, mean={:.1f}".format(
                    frame_idx, y_frame.min(), y_frame.max(), y_frame.mean()))

                return y_frame

        except Exception as e:
            print("Error reading frame {}: {}".format(frame_idx, e))
            return None

    def save_frame_as_image(self, frame, frame_idx, output_dir="debug_frames"):
        """Save frame as PNG for visual inspection."""
        if frame is None:
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = "{}/frame_{}_original.png".format(output_dir, frame_idx)
        cv2.imwrite(output_path, frame)
        print("Saved frame {} as {}".format(frame_idx, output_path))


def check_model_files(base_path):
    """Check if all required model files exist and return the correct path."""
    required_extensions = ['.data-00000-of-00001', '.index', '.meta']

    # Check if files exist with all required extensions
    for ext in required_extensions:
        full_path = base_path + ext
        if not os.path.exists(full_path):
            print("Missing model file: {}".format(full_path))
            return None

    print("Found all required model files:")
    for ext in required_extensions:
        full_path = base_path + ext
        print("  - {}".format(full_path))

    return base_path


def diagnose_yuv_file(yuv_file, width, height):
    """Diagnose YUV file format and dimensions."""
    if not os.path.exists(yuv_file):
        print("File not found: {}".format(yuv_file))
        return

    file_size = os.path.getsize(yuv_file)
    print("File size: {:,} bytes".format(file_size))

    # Test different formats
    formats = {
        'YUV420': width * height * 3 // 2,
        'YUV422': width * height * 2,
        'YUV444': width * height * 3
    }

    print("\nFormat analysis for {}x{}:".format(width, height))
    for fmt, frame_size in formats.items():
        num_frames = file_size // frame_size
        remainder = file_size % frame_size
        print("{}: {} complete frames (frame size: {:,} bytes, remainder: {} bytes)".format(
            fmt, num_frames, frame_size, remainder))


def extract_ctus_from_frame(frame, ctu_size=64):
    """Extract all CTUs from a frame with proper padding."""
    height, width = frame.shape
    print("Original frame dimensions: {}x{}".format(width, height))

    # Calculate padded dimensions
    padded_height = int(math.ceil(height / float(ctu_size))) * ctu_size
    padded_width = int(math.ceil(width / float(ctu_size))) * ctu_size

    print("Padded dimensions: {}x{}".format(padded_width, padded_height))

    # Create padded frame with zero padding
    padded_frame = np.zeros((padded_height, padded_width), dtype=np.uint8)
    padded_frame[:height, :width] = frame

    ctus = []
    ctu_positions = []

    # Extract CTUs
    ctu_rows = padded_height // ctu_size
    ctu_cols = padded_width // ctu_size

    print("CTU grid: {}x{} = {} CTUs".format(ctu_cols, ctu_rows, ctu_cols * ctu_rows))

    for row in range(ctu_rows):
        for col in range(ctu_cols):
            y = row * ctu_size
            x = col * ctu_size
            ctu = padded_frame[y:y + ctu_size, x:x + ctu_size]
            ctus.append(ctu)
            ctu_positions.append((x, y))

    return ctus, ctu_positions, (padded_width, padded_height)


def load_cnn_model():
    """Load CNN model for CTU analysis."""
    print("Loading CNN model...")

    # Check if model files exist
    model_path = check_model_files(CNN_MODEL_PATH)
    if model_path is None:
        raise FileNotFoundError("CNN model files not found. Check the model path and files.")

    # Configure TensorFlow for compatibility
    tf.compat.v1.disable_eager_execution()

    # Define placeholders
    x = tf.compat.v1.placeholder("float", [None, 64, 64, 1])
    y_ = tf.compat.v1.placeholder("float", [None, 16])
    qp_ph = tf.compat.v1.placeholder("float", [None, 1])
    isdrop = tf.compat.v1.placeholder("float")
    global_step = tf.compat.v1.placeholder("float")

    # Build network
    _, _, _, y_conv_64, y_conv_32, y_conv_16, _, _, _, _, _, opt_vars = nt.net(
        x, y_, qp_ph, isdrop, global_step, 0.01, 0.9, 1, 1
    )

    # Create session and restore model
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True  # Allow fallback to CPU if GPU fails
    sess = tf.compat.v1.Session(config=config)

    saver = tf.compat.v1.train.Saver(opt_vars)

    try:
        saver.restore(sess, model_path)
        print("CNN model loaded successfully from: {}".format(model_path))
    except Exception as e:
        print("Error loading model: {}".format(e))
        sess.close()
        raise

    return sess, x, qp_ph, isdrop, y_conv_64, y_conv_32, y_conv_16


def get_cnn_predictions(ctus, cnn_model_data, qp_value=32):
    """Get CNN predictions for all CTUs with batch processing."""
    sess, x, qp_ph, isdrop, y_conv_64, y_conv_32, y_conv_16 = cnn_model_data

    num_ctus = len(ctus)
    predictions = []

    print("Running CNN inference on {} CTUs with QP={}...".format(num_ctus, qp_value))

    batch_size = 16
    num_batches = int(math.ceil(num_ctus / float(batch_size)))

    for i in range(0, num_ctus, batch_size):
        end_idx = min(i + batch_size, num_ctus)
        batch_ctus = ctus[i:end_idx]
        batch_size_actual = len(batch_ctus)

        # Prepare batch data
        batch_images = np.array(batch_ctus).reshape(batch_size_actual, 64, 64, 1).astype(np.float32)
        batch_qp = np.full((batch_size_actual, 1), qp_value, dtype=np.float32)

        # Run inference
        try:
            pred_64, pred_32, pred_16 = sess.run(
                [y_conv_64, y_conv_32, y_conv_16],
                feed_dict={x: batch_images, qp_ph: batch_qp, isdrop: 0}
            )

            # Store predictions
            for j in range(batch_size_actual):
                predictions.append({
                    'pred_64': pred_64[j],
                    'pred_32': pred_32[j],
                    'pred_16': pred_16[j]
                })

        except Exception as e:
            print("Error in CNN inference batch {}: {}".format(i // batch_size + 1, e))
            # Add dummy predictions for failed batch
            for j in range(batch_size_actual):
                predictions.append({
                    'pred_64': np.zeros(16),
                    'pred_32': np.zeros(16),
                    'pred_16': np.zeros(16)
                })

        # Progress update
        if (i // batch_size + 1) % 5 == 0 or (i // batch_size + 1) == num_batches:
            print("Processed {}/{} CNN batches".format(i // batch_size + 1, num_batches))

    return predictions


def save_frame_data(frame_idx, frame, ctus, ctu_positions, padded_dims, cnn_predictions):
    """Save frame data in multiple formats for analysis."""

    # Create output directory
    output_dir = "frame_{}_analysis".format(frame_idx)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save complete frame data as numpy
    frame_data = {
        'frame_idx': frame_idx,
        'original_frame': frame,
        'ctus': ctus,
        'ctu_positions': ctu_positions,
        'padded_dims': padded_dims,
        'cnn_predictions': cnn_predictions,
        'metadata': {
            'frame_shape': frame.shape,
            'num_ctus': len(ctus),
            'ctu_grid': (padded_dims[0] // 64, padded_dims[1] // 64)
        }
    }

    np.save('{}/complete_data.npy'.format(output_dir), frame_data)
    print("Saved complete frame data to {}/complete_data.npy".format(output_dir))

    # Save original frame as image
    cv2.imwrite('{}/original_frame.png'.format(output_dir), frame)

    # Save individual CTUs
    ctu_dir = '{}/ctus'.format(output_dir)
    if not os.path.exists(ctu_dir):
        os.makedirs(ctu_dir)

    for i, (ctu, pos) in enumerate(zip(ctus, ctu_positions)):
        # Save as text file
        np.savetxt('{}/ctu_{:04d}_pos_{}_{}.txt'.format(ctu_dir, i, pos[0], pos[1]), ctu, fmt='%d')
        # Save as image
        cv2.imwrite('{}/ctu_{:04d}_pos_{}_{}.png'.format(ctu_dir, i, pos[0], pos[1]), ctu)

    # Save CNN predictions summary
    with open('{}/cnn_predictions_summary.txt'.format(output_dir), 'w') as f:
        f.write("Frame {} CNN Predictions Summary\n".format(frame_idx))
        f.write("Total CTUs: {}\n\n".format(len(cnn_predictions)))

        for i, pred in enumerate(cnn_predictions):
            pos = ctu_positions[i]
            f.write("CTU {:04d} at position ({}, {}):\n".format(i, pos[0], pos[1]))
            f.write("  64x64 prediction: {}... (mean: {:.4f})\n".format(pred['pred_64'][:5], pred['pred_64'].mean()))
            f.write("  32x32 prediction: {}... (mean: {:.4f})\n".format(pred['pred_32'][:5], pred['pred_32'].mean()))
            f.write("  16x16 prediction: {}... (mean: {:.4f})\n\n".format(pred['pred_16'][:5], pred['pred_16'].mean()))

    print("Saved analysis data to {}/".format(output_dir))


def extract_and_analyze():
    """Main function to extract frames and perform CNN analysis."""
    print("=== HEVC Frame Extraction and CNN Analysis ===")

    # First, diagnose the YUV file
    print("\n1. Diagnosing YUV file...")
    diagnose_yuv_file(YUV_FILE_PATH, FRAME_WIDTH, FRAME_HEIGHT)

    # Initialize frame extractor
    print("\n2. Initializing frame extractor...")
    try:
        extractor = FrameExtractor(YUV_FILE_PATH, FRAME_WIDTH, FRAME_HEIGHT, YUV_FORMAT)
    except Exception as e:
        print("Error initializing frame extractor: {}".format(e))
        return

    # Load CNN model
    print("\n3. Loading CNN model...")
    try:
        cnn_model = load_cnn_model()
    except Exception as e:
        print("Error loading CNN model: {}".format(e))
        return

    # Process each frame
    for frame_idx in FRAME_INDICES:
        print("\n4. Analyzing frame {}...".format(frame_idx))

        # Extract frame
        frame = extractor.get_frame(frame_idx)
        if frame is None:
            print("Skipping frame {} due to extraction error".format(frame_idx))
            continue

        # Save frame as image for debugging
        extractor.save_frame_as_image(frame, frame_idx)

        # Extract CTUs
        print("Extracting CTUs from frame {}...".format(frame_idx))
        ctus, ctu_positions, padded_dims = extract_ctus_from_frame(frame)
        print("Extracted {} CTUs".format(len(ctus)))

        # Get CNN predictions
        print("Running CNN analysis on frame {}...".format(frame_idx))
        cnn_predictions = get_cnn_predictions(ctus, cnn_model)

        # Save all data
        print("Saving analysis data for frame {}...".format(frame_idx))
        save_frame_data(frame_idx, frame, ctus, ctu_positions, padded_dims, cnn_predictions)

        print("Frame {} analysis complete!".format(frame_idx))

    # Cleanup
    cnn_model[0].close()
    print("\n=== Analysis Complete! ===")
    print("Check the generated directories for:")
    print("- debug_frames/: Original extracted frames as PNG")
    print("- frame_X_analysis/: Complete analysis data for each frame")


# Test just frame extraction without CNN (for debugging)
def test_frame_extraction_only():
    """Test frame extraction without CNN to verify YUV parsing is working."""
    print("=== Testing Frame Extraction Only ===")

    print("1. Diagnosing YUV file...")
    diagnose_yuv_file(YUV_FILE_PATH, FRAME_WIDTH, FRAME_HEIGHT)

    print("\n2. Testing frame extraction...")
    try:
        extractor = FrameExtractor(YUV_FILE_PATH, FRAME_WIDTH, FRAME_HEIGHT, YUV_FORMAT)

        for frame_idx in FRAME_INDICES:
            print("\nExtracting frame {}...".format(frame_idx))
            frame = extractor.get_frame(frame_idx)
            if frame is not None:
                extractor.save_frame_as_image(frame, frame_idx)
                print("Successfully extracted and saved frame {}".format(frame_idx))
            else:
                print("Failed to extract frame {}".format(frame_idx))

    except Exception as e:
        print("Error during frame extraction test: {}".format(e))


if __name__ == "__main__":
    # For testing frame extraction only (recommended first):
    # test_frame_extraction_only()

    # For full analysis:
    extract_and_analyze()
