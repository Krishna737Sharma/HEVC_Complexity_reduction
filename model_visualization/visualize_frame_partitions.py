import cv2
import numpy as np
import os
import glob


def draw_partitions_on_ctu(image, predictions, x_offset, y_offset, size=64, color=(0, 255, 0), thickness=1):
    """Draw partition lines on a CTU."""
    if size < 16:
        return

    should_split = False
    if size == 64:
        # For 64x64, check if prediction > 0.5
        if len(predictions['pred_64']) > 0 and predictions['pred_64'][0] > 0.5:
            should_split = True
    elif size == 32:
        # For 32x32, determine which quadrant and check corresponding prediction
        quad_idx = (((y_offset % 64) // 32) % 2) * 2 + (((x_offset % 64) // 32) % 2)
        if quad_idx < len(predictions['pred_32']) and predictions['pred_32'][quad_idx] > 0.5:
            should_split = True
    elif size == 16:
        # For 16x16, determine which 16x16 block and check corresponding prediction
        cu_idx = ((y_offset % 64) // 16) * 4 + ((x_offset % 64) // 16)
        if cu_idx < len(predictions['pred_16']) and predictions['pred_16'][cu_idx] > 0.5:
            should_split = True

    if should_split:
        half_size = size // 2
        # Draw horizontal and vertical lines
        cv2.line(image, (x_offset, y_offset + half_size),
                 (x_offset + size, y_offset + half_size), color, thickness)
        cv2.line(image, (x_offset + half_size, y_offset),
                 (x_offset + half_size, y_offset + size), color, thickness)

        # Recursively draw partitions for sub-blocks
        draw_partitions_on_ctu(image, predictions, x_offset, y_offset, half_size, color, thickness)
        draw_partitions_on_ctu(image, predictions, x_offset + half_size, y_offset, half_size, color, thickness)
        draw_partitions_on_ctu(image, predictions, x_offset, y_offset + half_size, half_size, color, thickness)
        draw_partitions_on_ctu(image, predictions, x_offset + half_size, y_offset + half_size, half_size, color,
                               thickness)


def load_vit_predictions_from_numpy_safe(frame_idx):
    """Load ViT predictions from numpy file with compatibility handling."""
    vit_dir = 'frame_{}_vit_analysis'.format(frame_idx)
    vit_file = os.path.join(vit_dir, 'vit_predictions.npy')

    if not os.path.exists(vit_file):
        print("ViT predictions file not found: {}".format(vit_file))
        return None

    try:
        # Try loading with allow_pickle=True for compatibility
        predictions = np.load(vit_file, allow_pickle=True)
        print("Loaded {} ViT predictions from {}".format(len(predictions), vit_file))
        return predictions
    except Exception as e:
        print("Error loading ViT predictions from numpy: {}".format(e))
        return None


def parse_float_from_line(line, prefix):
    """Safely extract float values from a line with given prefix."""
    try:
        # Find the prefix in the line
        if prefix not in line:
            return None

        # Extract the part after the prefix
        value_part = line.split(prefix)[1].strip()

        # Remove common formatting characters
        value_part = value_part.replace('[', '').replace(']', '').replace('(', '').replace(')', '')

        # Split by whitespace and extract only the numeric parts
        parts = value_part.split()
        values = []

        for part in parts:
            # Stop at non-numeric parts like '(mean:'
            if '(' in part or 'mean' in part:
                break
            try:
                val = float(part)
                values.append(val)
            except ValueError:
                continue

        return np.array(values) if values else None

    except Exception as e:
        print("Error parsing line '{}': {}".format(line.strip(), e))
        return None


def load_vit_predictions_from_text_fixed(frame_idx):
    """Load ViT predictions from text file with improved parsing."""
    vit_dir = 'frame_{}_vit_analysis'.format(frame_idx)
    vit_file = os.path.join(vit_dir, 'vit_predictions.txt')

    if not os.path.exists(vit_file):
        print("ViT text file not found: {}".format(vit_file))
        return None

    predictions = []
    try:
        with open(vit_file, 'r') as f:
            lines = f.readlines()

        current_ctu = None
        pred_64 = None
        pred_32 = None
        pred_16 = None

        for line in lines:
            line = line.strip()

            if line.startswith('CTU') and 'at position' in line:
                # Save previous CTU if complete
                if current_ctu is not None and pred_64 is not None and pred_32 is not None and pred_16 is not None:
                    predictions.append({
                        'pred_64': pred_64,
                        'pred_32': pred_32,
                        'pred_16': pred_16
                    })

                # Start new CTU
                current_ctu = line
                pred_64 = None
                pred_32 = None
                pred_16 = None

            elif '64x64:' in line:
                pred_64 = parse_float_from_line(line, '64x64:')

            elif '32x32:' in line:
                pred_32 = parse_float_from_line(line, '32x32:')

            elif '16x16:' in line:
                pred_16 = parse_float_from_line(line, '16x16:')

        # Don't forget the last CTU
        if current_ctu is not None and pred_64 is not None and pred_32 is not None and pred_16 is not None:
            predictions.append({
                'pred_64': pred_64,
                'pred_32': pred_32,
                'pred_16': pred_16
            })

        print("Successfully parsed {} ViT predictions from text file".format(len(predictions)))
        return predictions

    except Exception as e:
        print("Error loading ViT predictions from text: {}".format(e))
        return None


def load_frame_data(frame_idx):
    """Load frame data from analysis directory."""
    # Try different possible locations
    possible_files = [
        'frame_{}_analysis/complete_data.npy'.format(frame_idx),
        'frame_{}_data.npy'.format(frame_idx)
    ]

    for file_path in possible_files:
        if os.path.exists(file_path):
            try:
                frame_data = np.load(file_path, allow_pickle=True).item()
                print("Loaded frame data from: {}".format(file_path))
                return frame_data
            except Exception as e:
                print("Error loading {}: {}".format(file_path, e))
                continue

    print("No frame data file found for frame {}".format(frame_idx))
    return None


def create_clean_frame_visualization(frame_idx):
    """Create clean visualization without text overlays."""
    print("Creating clean visualization for frame {}...".format(frame_idx))

    # Load frame data
    frame_data = load_frame_data(frame_idx)
    if frame_data is None:
        print("Cannot create visualization - frame data not found")
        return

    frame = frame_data['original_frame'] if 'original_frame' in frame_data else frame_data['frame']
    ctu_positions = frame_data['ctu_positions']
    cnn_predictions = frame_data['cnn_predictions']
    padded_dims = frame_data['padded_dims']

    # Load ViT predictions (try numpy first, then text)
    vit_predictions = load_vit_predictions_from_numpy_safe(frame_idx)
    if vit_predictions is None:
        vit_predictions = load_vit_predictions_from_text_fixed(frame_idx)

    if vit_predictions is None:
        print("ViT predictions not found for frame {}".format(frame_idx))
        print("Creating visualization with CNN predictions only...")

        # Create CNN-only visualization
        padded_frame = np.zeros((padded_dims[1], padded_dims[0]), dtype=np.uint8)
        padded_frame[:frame.shape[0], :frame.shape[1]] = frame
        padded_frame_color = cv2.cvtColor(padded_frame, cv2.COLOR_GRAY2BGR)

        # Original frame
        original_vis = padded_frame_color.copy()

        # CNN visualization - Red lines
        cnn_vis = padded_frame_color.copy()
        for i, (pos, pred) in enumerate(zip(ctu_positions, cnn_predictions)):
            cnn_pred_dict = {
                'pred_64': pred['pred_64'] if isinstance(pred, dict) else pred[0],
                'pred_32': pred['pred_32'] if isinstance(pred, dict) else pred[1],
                'pred_16': pred['pred_16'] if isinstance(pred, dict) else pred[2]
            }
            draw_partitions_on_ctu(cnn_vis, cnn_pred_dict, pos[0], pos[1], 64, color=(0, 0, 255), thickness=2)

        # Save CNN-only results
        h, w = frame.shape
        scale = min(1.0, 1000.0 / max(w, h))
        new_width = int(w * scale)
        new_height = int(h * scale)

        if scale != 1.0:
            original_scaled = cv2.resize(original_vis[:h, :w], (new_width, new_height), interpolation=cv2.INTER_AREA)
            cnn_scaled = cv2.resize(cnn_vis[:h, :w], (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            original_scaled = original_vis[:h, :w]
            cnn_scaled = cnn_vis[:h, :w]

        # Create output directory
        output_dir = 'visualization_output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cv2.imwrite('{}/frame_{}_original.png'.format(output_dir, frame_idx), original_scaled)
        cv2.imwrite('{}/frame_{}_cnn_only.png'.format(output_dir, frame_idx), cnn_scaled)

        print("CNN-only visualization saved for frame {}".format(frame_idx))
        return

    print("Frame shape: {}, Padded dims: {}".format(frame.shape, padded_dims))
    print("CTU positions: {}, CNN predictions: {}, ViT predictions: {}".format(
        len(ctu_positions), len(cnn_predictions), len(vit_predictions)))

    # Create base images
    padded_frame = np.zeros((padded_dims[1], padded_dims[0]), dtype=np.uint8)
    padded_frame[:frame.shape[0], :frame.shape[1]] = frame
    padded_frame_color = cv2.cvtColor(padded_frame, cv2.COLOR_GRAY2BGR)

    # Original frame (no partitions)
    original_vis = padded_frame_color.copy()

    # CNN visualization - Red lines
    cnn_vis = padded_frame_color.copy()
    for i, (pos, pred) in enumerate(zip(ctu_positions, cnn_predictions)):
        # Convert CNN predictions to compatible format
        cnn_pred_dict = {
            'pred_64': pred['pred_64'] if isinstance(pred, dict) else pred[0],
            'pred_32': pred['pred_32'] if isinstance(pred, dict) else pred[1],
            'pred_16': pred['pred_16'] if isinstance(pred, dict) else pred[2]
        }
        draw_partitions_on_ctu(cnn_vis, cnn_pred_dict, pos[0], pos[1], 64, color=(0, 0, 255), thickness=2)

    # ViT visualization - Blue lines
    vit_vis = padded_frame_color.copy()
    for i, (pos, pred) in enumerate(zip(ctu_positions, vit_predictions)):
        if i < len(vit_predictions):
            draw_partitions_on_ctu(vit_vis, pred, pos[0], pos[1], 64, color=(255, 0, 0), thickness=2)

    # Overlay both models - Red (CNN) + Blue (ViT)
    overlay_vis = padded_frame_color.copy()
    for i, pos in enumerate(ctu_positions):
        if i < len(cnn_predictions):
            cnn_pred_dict = {
                'pred_64': cnn_predictions[i]['pred_64'] if isinstance(cnn_predictions[i], dict) else
                cnn_predictions[i][0],
                'pred_32': cnn_predictions[i]['pred_32'] if isinstance(cnn_predictions[i], dict) else
                cnn_predictions[i][1],
                'pred_16': cnn_predictions[i]['pred_16'] if isinstance(cnn_predictions[i], dict) else
                cnn_predictions[i][2]
            }
            draw_partitions_on_ctu(overlay_vis, cnn_pred_dict, pos[0], pos[1], 64, color=(0, 0, 255),
                                   thickness=1)  # CNN - Red

        if i < len(vit_predictions):
            draw_partitions_on_ctu(overlay_vis, vit_predictions[i], pos[0], pos[1], 64, color=(255, 0, 0),
                                   thickness=1)  # ViT - Blue

    # Crop back to original frame size
    h, w = frame.shape
    original_cropped = original_vis[:h, :w]
    cnn_vis_cropped = cnn_vis[:h, :w]
    vit_vis_cropped = vit_vis[:h, :w]
    overlay_vis_cropped = overlay_vis[:h, :w]

    # Scale for better viewing (reduce if too large)
    scale = min(1.0, 1000.0 / max(w, h))  # Scale down if larger than 1000px
    new_width = int(w * scale)
    new_height = int(h * scale)

    if scale != 1.0:
        original_scaled = cv2.resize(original_cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cnn_scaled = cv2.resize(cnn_vis_cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)
        vit_scaled = cv2.resize(vit_vis_cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)
        overlay_scaled = cv2.resize(overlay_vis_cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print("Scaled images from {}x{} to {}x{}".format(w, h, new_width, new_height))
    else:
        original_scaled = original_cropped
        cnn_scaled = cnn_vis_cropped
        vit_scaled = vit_vis_cropped
        overlay_scaled = overlay_vis_cropped

    # Create output directory
    output_dir = 'visualization_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save individual clean images
    cv2.imwrite('{}/frame_{}_original.png'.format(output_dir, frame_idx), original_scaled)
    cv2.imwrite('{}/frame_{}_cnn.png'.format(output_dir, frame_idx), cnn_scaled)
    cv2.imwrite('{}/frame_{}_vit.png'.format(output_dir, frame_idx), vit_scaled)
    cv2.imwrite('{}/frame_{}_overlay.png'.format(output_dir, frame_idx), overlay_scaled)

    # Create 2x2 grid without any text
    try:
        top_row = cv2.hconcat([original_scaled, cnn_scaled])
        bottom_row = cv2.hconcat([vit_scaled, overlay_scaled])
        grid = cv2.vconcat([top_row, bottom_row])
        cv2.imwrite('{}/frame_{}_clean_grid.png'.format(output_dir, frame_idx), grid)
    except Exception as e:
        print("Error creating grid image: {}".format(e))

    print("Clean images saved for frame {}:".format(frame_idx))
    print("- {}/frame_{}_original.png (No partitions)".format(output_dir, frame_idx))
    print("- {}/frame_{}_cnn.png (RED lines = CNN partitions)".format(output_dir, frame_idx))
    print("- {}/frame_{}_vit.png (BLUE lines = ViT partitions)".format(output_dir, frame_idx))
    print("- {}/frame_{}_overlay.png (RED + BLUE overlaid)".format(output_dir, frame_idx))
    print("- {}/frame_{}_clean_grid.png (2x2 grid: Original|CNN / ViT|Overlay)".format(output_dir, frame_idx))


def visualize_all_frames_clean():
    """Create clean visualizations for all frames."""
    print("=== CLEAN FRAME VISUALIZATION ===")
    print("COLOR CODING:")
    print("- RED lines = CNN model partitions")
    print("- BLUE lines = ViT model partitions")
    print("- No text overlays on images")
    print("=====================================\n")

    # Find all available frame analysis directories (avoid duplicates)
    analysis_dirs = glob.glob('frame_*_analysis')
    available_frames = set()

    for dir_name in analysis_dirs:
        # Skip ViT analysis directories for frame detection
        if 'vit_analysis' in dir_name:
            continue
        try:
            frame_idx = int(dir_name.split('_')[1])
            available_frames.add(frame_idx)
        except (ValueError, IndexError):
            continue

    # Also check for old-style data files
    data_files = glob.glob('frame_*_data.npy')
    for file_name in data_files:
        try:
            frame_idx = int(file_name.split('_')[1])
            available_frames.add(frame_idx)
        except (ValueError, IndexError):
            continue

    if not available_frames:
        print("No frame data found!")
        print("Expected files:")
        print("- frame_X_analysis/complete_data.npy")
        print("- frame_X_data.npy")
        return

    available_frames = sorted(list(available_frames))
    print("Found frames: {}".format(available_frames))

    for frame_idx in available_frames:
        print("\n" + "=" * 50)
        create_clean_frame_visualization(frame_idx)
        print("=" * 50)

    print("\nAll visualizations complete!")
    print("Check the 'visualization_output' directory for results.")


def check_file_structure():
    """Check and report the current file structure."""
    print("=== FILE STRUCTURE CHECK ===")

    # Check for frame analysis directories
    analysis_dirs = glob.glob('frame_*_analysis')
    regular_analysis = [d for d in analysis_dirs if 'vit_analysis' not in d]
    vit_analysis = [d for d in analysis_dirs if 'vit_analysis' in d]

    print("Frame analysis directories found: {}".format(len(regular_analysis)))
    for d in sorted(regular_analysis):
        print("  - {}".format(d))
        complete_data = os.path.join(d, 'complete_data.npy')
        if os.path.exists(complete_data):
            print("    ✓ complete_data.npy found")
        else:
            print("    ✗ complete_data.npy missing")

    print("\nViT analysis directories found: {}".format(len(vit_analysis)))
    for d in sorted(vit_analysis):
        print("  - {}".format(d))
        vit_numpy = os.path.join(d, 'vit_predictions.npy')
        vit_txt = os.path.join(d, 'vit_predictions.txt')
        if os.path.exists(vit_numpy):
            print("    ✓ vit_predictions.npy found")
        elif os.path.exists(vit_txt):
            print("    ✓ vit_predictions.txt found")
        else:
            print("    ✗ ViT predictions missing")

    # Check for old-style files
    old_files = glob.glob('frame_*_data.npy')
    if old_files:
        print("\nOld-style data files found: {}".format(len(old_files)))
        for f in sorted(old_files):
            print("  - {}".format(f))


if __name__ == "__main__":
    # First check file structure
    check_file_structure()
    print("\n")

    # Then create visualizations
    visualize_all_frames_clean()
