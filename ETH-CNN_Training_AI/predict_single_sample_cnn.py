import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import data_info as di
import net_CTU64 as nt

# Enable TensorFlow 1.x compatibility behavior
tf.compat.v1.disable_eager_execution()

# --- 1. Configuration: SET THESE VALUES ---

# Path to your .dat file
DAT_FILE_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/ETH-CNN_Training_AI/Data/AI_Valid_143925.dat_shuffled'
# Path to your trained TensorFlow model checkpoint
MODEL_CHECKPOINT_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/ETH-CNN_Training_AI/Models/model_20181226_045506_1000000_qp32.dat' 
# The index of the sample you want to predict (from your example script)
SAMPLE_INDEX = 50 
# The QP to use for prediction (MUST match the loaded model)
QP_TO_INSPECT = 32

# --- Data Structure Constants ---
IMAGE_SIZE = 64
IMAGE_BYTES = IMAGE_SIZE * IMAGE_SIZE
NUM_LABEL_BYTES = 16
NUM_SAMPLE_LENGTH = 4992

# --- 2. Helper Functions ---

def draw_cu_partitions(ax, label_data, x_offset, y_offset, color='cyan'):
    label_grid = label_data.reshape(4, 4)
    def draw_rect(x, y, size, lw=1.2):
        rect = patches.Rectangle((x - 0.5, y - 0.5), size, size, linewidth=lw, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    if np.all(label_grid == 0):
        draw_rect(x_offset, y_offset, 64); return
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
                            draw_rect(x_16, y_16, 16)
                        elif label == 3:
                            draw_rect(x_16, y_16, 8); draw_rect(x_16 + 8, y_16, 8)
                            draw_rect(x_16, y_16 + 8, 8); draw_rect(x_16 + 8, y_16 + 8, 8)

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
    # --- Read the Single Sample ---
    try:
        with open(DAT_FILE_PATH, 'rb') as f:
            start_byte = SAMPLE_INDEX * NUM_SAMPLE_LENGTH
            f.seek(start_byte)
            sample_bytes = f.read(NUM_SAMPLE_LENGTH)
            if len(sample_bytes) < NUM_SAMPLE_LENGTH:
                print("Error: Could not read a full sample at index {}.".format(SAMPLE_INDEX))
                return
        data_array = np.frombuffer(sample_bytes, dtype=np.uint8)
    except FileNotFoundError:
        print("Error: Data file not found at '{}'.".format(DAT_FILE_PATH)); return
        
    # --- Extract Ground Truth and Image ---
    image_patch = data_array[0:IMAGE_BYTES].reshape(IMAGE_SIZE, IMAGE_SIZE)
    labels_start_index = IMAGE_BYTES + 64
    label_offset = QP_TO_INSPECT * NUM_LABEL_BYTES
    ground_truth_label = data_array[labels_start_index + label_offset : labels_start_index + label_offset + NUM_LABEL_BYTES]

    # --- Prepare for Prediction ---
    image_tensor = image_patch.astype(np.float32).reshape(1, 64, 64, 1)
    qp_tensor = np.array([float(QP_TO_INSPECT)]).reshape(1, 1)

    # --- Load Model and Run Prediction ---
    print("Loading TensorFlow model from '{}'...".format(MODEL_CHECKPOINT_PATH))
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        x = tf.compat.v1.placeholder("float", [None, 64, 64, 1])
        y_ = tf.compat.v1.placeholder("float", [None, 16]) # Dummy
        qp = tf.compat.v1.placeholder("float", [None, 1])
        
        _, _, _, y_conv_64, y_conv_32, y_conv_16, _, _, _, _, _, _ = nt.net(x, y_, qp, 0.0, 0.0, 0.01, 0.9, 1, 1)

        model_variables = tf.compat.v1.trainable_variables()
        saver = tf.compat.v1.train.Saver(var_list=model_variables)
        
        try:
            saver.restore(sess, MODEL_CHECKPOINT_PATH)
            print("Model restored successfully.")
        except Exception as e:
            print("Error restoring model: {}".format(e)); return
        
        pred_64, pred_32, pred_16 = sess.run(
            [y_conv_64, y_conv_32, y_conv_16],
            feed_dict={x: image_tensor, qp: qp_tensor}
        )

    # --- Decode and Display Results ---
    predicted_label = decode_cnn_prediction_to_depth_map(pred_64, pred_32, pred_16)
    
    print("\n--- Results for Sample #{} at QP={} ---".format(SAMPLE_INDEX, QP_TO_INSPECT))
    print("Ground Truth Label: {}".format(ground_truth_label))
    print("Predicted Label:    {}".format(predicted_label))
    
    # --- Create Side-by-Side Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(image_patch, cmap='gray')
    draw_cu_partitions(ax1, ground_truth_label, x_offset=0, y_offset=0, color='yellow')
    ax1.set_title("Ground Truth Partitions")
    ax1.axis('off')

    ax2.imshow(image_patch, cmap='gray')
    draw_cu_partitions(ax2, predicted_label, x_offset=0, y_offset=0, color='cyan')
    ax2.set_title("CNN Model Prediction")
    ax2.axis('off')
    
    output_filename = 'cnn_pred_sample_{}_qp_{}.png'.format(SAMPLE_INDEX, QP_TO_INSPECT)
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print("\n\xE2\x9C\x85 Comparison visualization saved as '{}'".format(output_filename))

if __name__ == "__main__":
    main()