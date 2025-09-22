import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import data_info as di
import net_CTU64 as nt # Import the network definition
import glob # Import the glob library

# Enable TensorFlow 1.x compatibility behavior
tf.compat.v1.disable_eager_execution()

# --- 1. Configuration: SET THESE VALUES ---

# Paths to your data and model
YUV_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/AI_YUV&Info/Info&YUV/AI_YUV'
INFO_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/AI_YUV&Info/Info&YUV/AI_Info'
MODEL_CHECKPOINT_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/ETH-CNN_Training_AI/Models/model_20181226_045506_1000000_qp32.dat'

# Details of the frame to predict
VIDEO_FILENAME = 'IntraTrain_768x512'
FRAME_TO_VISUALIZE = 10
# The QP must match the model you are loading
QP_TO_INSPECT = 32

# --- 2. Helper Functions ---
def read_YUV420_frame(fid, width, height, frame_index):
    frame_bytes = (width * height * 3) // 2
    fid.seek(frame_index * frame_bytes)
    Y_buf = fid.read(width * height)
    if not Y_buf:
        return None
    return np.frombuffer(Y_buf, dtype=np.uint8).reshape([height, width])

def read_info_frame(fid, width, height, frame_index):
    num_units_wide = width // 16
    num_units_high = height // 16
    frame_bytes = num_units_wide * num_units_high
    fid.seek(frame_index * frame_bytes)
    info_buf = fid.read(frame_bytes)
    if not info_buf:
        return None
    return np.frombuffer(info_buf, dtype=np.uint8).reshape([num_units_high, num_units_wide])

def draw_cu_partitions(ax, label_data, x_offset, y_offset, color='cyan'):
    label_grid = label_data.reshape(4, 4)
    def draw_rect(x, y, size, lw=1.2):
        rect = patches.Rectangle((x - 0.5, y - 0.5), size, size, linewidth=lw, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    if np.all(label_grid == 0):
        draw_rect(x_offset, y_offset, 64)
        return
    for q_r in range(2):
        for q_c in range(2):
            sub_grid = label_grid[q_r*2:(q_r+1)*2, q_c*2:(q_c+1)*2]
            if np.all(sub_grid == 1):
                draw_rect(x_offset + q_c*32, y_offset + q_r*32, 32)
            else:
                for r in range(2):
                    for c in range(2):
                        label = sub_grid[r, c]
                        x_16, y_16 = x_offset + q_c*32 + c*16, y_offset + q_r*32 + r*16
                        if label == 2:
                            draw_rect(x_16, y_16, 16, lw=0.8)
                        elif label == 3:
                            draw_rect(x_16, y_16, 8, lw=0.8)
                            draw_rect(x_16 + 8, y_16, 8, lw=0.8)
                            draw_rect(x_16, y_16 + 8, 8, lw=0.8)
                            draw_rect(x_16 + 8, y_16 + 8, 8, lw=0.8)

def decode_cnn_prediction_to_depth_map(pred_64, pred_32, pred_16):
    d_64 = (pred_64 > 0.5).astype(int).flatten()
    d_32 = (pred_32 > 0.5).astype(int).flatten()
    d_16 = (pred_16 > 0.5).astype(int).flatten()
    depth_map = np.zeros(16, dtype=int)
    if d_64[0] == 1:
        depth_map[:] = 1
        for i in range(4):
            if d_32[i] == 1:
                start_idx = (i//2)*8 + (i%2)*2
                depth_map[start_idx:start_idx+2] = 2
                depth_map[start_idx+4:start_idx+6] = 2
    for i in range(16):
        if d_16[i] == 1:
            depth_map[i] = 3
    return depth_map

# --- 3. Main Prediction and Visualization Logic ---

def main():
    try:
        video_index = di.YUV_NAME_LIST_FULL.index(VIDEO_FILENAME)
        width, height = di.YUV_WIDTH_LIST_FULL[video_index], di.YUV_HEIGHT_LIST_FULL[video_index]
        yuv_file = os.path.join(YUV_PATH, VIDEO_FILENAME + '.yuv')
        
        # FIX: Replace f-string with .format-style for Python 3.5
        #info_file_pattern = os.path.join(INFO_PATH, f'Info*_{VIDEO_FILENAME}_*qp{QP_TO_INSPECT}*CUDepth.dat')
        info_file_pattern = os.path.join(INFO_PATH, "Info*_%s_*qp%d*CUDepth.dat" % (VIDEO_FILENAME, QP_TO_INSPECT))

        info_files = glob.glob(info_file_pattern)
        if not info_files:
            print("Error: Could not find Info file matching pattern: %s" % info_file_pattern)
            return
        info_file = info_files[0] 

        with open(yuv_file, 'rb') as f_yuv:
            frame_pixels = read_YUV420_frame(f_yuv, width, height, FRAME_TO_VISUALIZE)
        with open(info_file, 'rb') as f_info:
            ground_truth_map = read_info_frame(f_info, width, height, FRAME_TO_VISUALIZE)
        if frame_pixels is None or ground_truth_map is None:
            print("Error: Could not read frame #%d." % FRAME_TO_VISUALIZE)
            return
    except (ValueError, FileNotFoundError, IndexError) as e:
        print("Error loading data: %s" % str(e))
        return

    # --- Load Model Once ---
    print("Loading TensorFlow model from '%s'..." % MODEL_CHECKPOINT_PATH)
    tf.compat.v1.reset_default_graph()
    x = tf.compat.v1.placeholder("float", [None, 64, 64, 1])
    qp = tf.compat.v1.placeholder("float", [None, 1])
    # The net_CTU64 version of net() requires more placeholders
    if 'net_CTU64' in nt.__name__:
        y_ = tf.compat.v1.placeholder("float", [None, 16])
        isdrop = tf.compat.v1.placeholder("float")
        global_step = tf.compat.v1.placeholder("float")
        _, _, _, y_conv_64, y_conv_32, y_conv_16, _, _, _, _, _, _ = nt.net(x, y_, qp, isdrop, global_step, 0.01, 0.9, 1, 1)
    # The net_CNN version of net() requires fewer placeholders
    else:
        y_ = tf.compat.v1.placeholder("float", [None, 16])
        isdrop = tf.compat.v1.placeholder("float")
        y_flat_64, y_flat_32, y_flat_16, y_conv_64, y_conv_32, y_conv_16, opt_vars_all = nt.net(x, y_, qp, isdrop)

    model_variables = tf.compat.v1.trainable_variables()
    saver = tf.compat.v1.train.Saver(var_list=model_variables)
    
    sess = tf.compat.v1.Session()
    saver.restore(sess, MODEL_CHECKPOINT_PATH)
    print("Model restored successfully.")

    # --- Prepare for Loop ---
    full_predicted_map = np.zeros_like(ground_truth_map)
    
    # Pass the raw QP value. The model will normalize it internally.
    qp_tensor = np.array([float(QP_TO_INSPECT)]).reshape(1, 1)
    
    print("\nProcessing frame %d block by block..." % FRAME_TO_VISUALIZE)
    # --- Loop through frame and predict for each CTU ---
    for y_ctu in range(height // 64):
        for x_ctu in range(width // 64):
            x_start, y_start = x_ctu * 64, y_ctu * 64
            image_patch = frame_pixels[y_start:y_start+64, x_start:x_start+64]
            # Pass the raw image pixel values. The model normalizes them internally.
            image_tensor = image_patch.astype(np.float32).reshape(1, 64, 64, 1)
            feed_dict = {x: image_tensor, qp: qp_tensor}
            if 'net_CTU64' in nt.__name__:
                feed_dict.update({isdrop: 0.0, global_step: 0.0})
            pred_64, pred_32, pred_16 = sess.run(
                [y_conv_64, y_conv_32, y_conv_16],
                feed_dict=feed_dict
            )
            predicted_label = decode_cnn_prediction_to_depth_map(pred_64, pred_32, pred_16)
            map_x_start, map_y_start = x_ctu * 4, y_ctu * 4
            full_predicted_map[map_y_start:map_y_start+4, map_x_start:map_x_start+4] = predicted_label.reshape(4,4)
    
    sess.close()
    print("Frame processing complete.")

    # --- Save Text File of Prediction ---
    txt_filename = 'cnn_full_frame_prediction_qp%d.txt' % QP_TO_INSPECT
    np.savetxt(txt_filename, full_predicted_map, fmt='%d')
    print("✅ Predicted partition map saved to '%s'" % txt_filename)

    # --- Create Side-by-Side Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width / 50, height / 50))
    
    ax1.imshow(frame_pixels, cmap='gray')
    for y_ctu in range(height // 64):
        for x_ctu in range(width // 64):
            label = ground_truth_map[y_ctu*4:(y_ctu+1)*4, x_ctu*4:(x_ctu+1)*4].flatten()
            draw_cu_partitions(ax1, label, x_offset=x_ctu*64, y_offset=y_ctu*64, color='yellow')
    ax1.set_title("Ground Truth Partitions")
    ax1.axis('off')

    ax2.imshow(frame_pixels, cmap='gray')
    for y_ctu in range(height // 64):
        for x_ctu in range(width // 64):
            label = full_predicted_map[y_ctu*4:(y_ctu+1)*4, x_ctu*4:(x_ctu+1)*4].flatten()
            draw_cu_partitions(ax2, label, x_offset=x_ctu*64, y_offset=y_ctu*64, color='cyan')
    ax2.set_title("CNN Model Prediction")
    ax2.axis('off')
    
    output_filename = 'cnn_full_frame_comparison_qp%d.png' % QP_TO_INSPECT
    fig.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    
    print("✅ Comparison visualization saved as '%s'" % output_filename)

if __name__ == "__main__":
    main()
