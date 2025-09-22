import tensorflow as tf
import numpy as np
import os
import math

# --- Necessary components from your uploaded files ---

# From: net_CTU64.py
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_EXT_FEATURES = 1
NUM_LABEL_BYTES = 16
NUM_CONVLAYER1_FILTERS = 16
NUM_CONVLAYER2_FILTERS = 24
NUM_CONVLAYER3_FILTERS = 32
NUM_CONV2_FLAT_S_FILTERS = 8 * 8 * NUM_CONVLAYER2_FILTERS
NUM_CONV2_FLAT_M_FILTERS = 4 * 4 * NUM_CONVLAYER2_FILTERS
NUM_CONV2_FLAT_L_FILTERS = 2 * 2 * NUM_CONVLAYER2_FILTERS
NUM_CONV3_FLAT_S_FILTERS = 4 * 4 * NUM_CONVLAYER3_FILTERS
NUM_CONV3_FLAT_M_FILTERS = 2 * 2 * NUM_CONVLAYER3_FILTERS
NUM_CONV3_FLAT_L_FILTERS = 1 * 1 * NUM_CONVLAYER3_FILTERS
NUM_CONVLAYER_FLAT_FILTERS = (NUM_CONV2_FLAT_S_FILTERS + NUM_CONV2_FLAT_M_FILTERS +
                              NUM_CONV2_FLAT_L_FILTERS + NUM_CONV3_FLAT_S_FILTERS +
                              NUM_CONV3_FLAT_M_FILTERS + NUM_CONV3_FLAT_L_FILTERS)
NUM_DENLAYER1_FEATURES_64 = 64
NUM_DENLAYER2_FEATURES_64 = 48
NUM_DENLAYER1_FEATURES_32 = 128
NUM_DENLAYER2_FEATURES_32 = 96
NUM_DENLAYER1_FEATURES_16 = 256
NUM_DENLAYER2_FEATURES_16 = 192

# --- Helper functions from net_CTU64.py and train_CNN_CTU64.py ---
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def aver_pool(x, k_width):
    return tf.nn.avg_pool(x, ksize=[1, k_width, k_width, 1], strides=[1, k_width, k_width, 1], padding='SAME')

def activate(x, acti_mode):
    if acti_mode == 5: # Leaky ReLU
        return tf.nn.leaky_relu(x)
    elif acti_mode == 2: # Sigmoid
        return tf.nn.sigmoid(x)
    return x

def zero_mean_norm_local(x, x_width, kernel_width):
    w_norm = tf.constant(1.0/(kernel_width*kernel_width), tf.float32, shape=[kernel_width, kernel_width,1,1])
    x_mean_reduced = tf.nn.conv2d(x, w_norm, [1, kernel_width, kernel_width, 1], 'VALID')
    x_mean_expanded = tf.image.resize_nearest_neighbor(x_mean_reduced, [x_width, x_width])
    return x - x_mean_expanded

def non_overlap_conv(x, k_width, num_filters_in, num_filters_out, acti_mode):
    w_conv = weight_variable([k_width, k_width, num_filters_in, num_filters_out])
    b_conv = bias_variable([num_filters_out])
    h_conv = tf.nn.conv2d(x, w_conv, strides=[1, k_width, k_width, 1], padding='VALID') + b_conv
    return activate(h_conv, acti_mode)

def full_connect(x, num_filters_in, num_filters_out, acti_mode, keep_prob=1, name_w=None, name_b=None):
    w_fc = weight_variable([num_filters_in, num_filters_out], name_w)
    b_fc = bias_variable([num_filters_out], name_b)
    h_fc = tf.matmul(x, w_fc) + b_fc
    return activate(h_fc, acti_mode)

def build_net(x, qp):
    x = tf.scalar_mul(1.0 / 255.0, x)
    qp_norm = tf.scalar_mul(1 / 51.0, qp)
    x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

    acti_mode_conv = 5
    h_image_L = zero_mean_norm_local(aver_pool(x_image, 4), 16, 16)
    h_conv1_L = non_overlap_conv(h_image_L, 4, NUM_CHANNELS, NUM_CONVLAYER1_FILTERS, acti_mode=acti_mode_conv)
    h_conv2_L = non_overlap_conv(h_conv1_L, 2, NUM_CONVLAYER1_FILTERS, NUM_CONVLAYER2_FILTERS, acti_mode=acti_mode_conv)
    h_conv3_L = non_overlap_conv(h_conv2_L, 2, NUM_CONVLAYER2_FILTERS, NUM_CONVLAYER3_FILTERS, acti_mode=acti_mode_conv)

    h_image_M = zero_mean_norm_local(aver_pool(x_image, 2), 32, 16)
    h_conv1_M = non_overlap_conv(h_image_M, 4, NUM_CHANNELS, NUM_CONVLAYER1_FILTERS, acti_mode=acti_mode_conv)
    h_conv2_M = non_overlap_conv(h_conv1_M, 2, NUM_CONVLAYER1_FILTERS, NUM_CONVLAYER2_FILTERS, acti_mode=acti_mode_conv)
    h_conv3_M = non_overlap_conv(h_conv2_M, 2, NUM_CONVLAYER2_FILTERS, NUM_CONVLAYER3_FILTERS, acti_mode=acti_mode_conv)

    h_image_S = zero_mean_norm_local(x_image, 64, 16)
    h_conv1_S = non_overlap_conv(h_image_S, 4, NUM_CHANNELS, NUM_CONVLAYER1_FILTERS, acti_mode=acti_mode_conv)
    h_conv2_S = non_overlap_conv(h_conv1_S, 2, NUM_CONVLAYER1_FILTERS, NUM_CONVLAYER2_FILTERS, acti_mode=acti_mode_conv)
    h_conv3_S = non_overlap_conv(h_conv2_S, 2, NUM_CONVLAYER2_FILTERS, NUM_CONVLAYER3_FILTERS, acti_mode=acti_mode_conv)

    h_conv3_S_flat = tf.reshape(h_conv3_S, [-1, NUM_CONV3_FLAT_S_FILTERS])
    h_conv3_M_flat = tf.reshape(h_conv3_M, [-1, NUM_CONV3_FLAT_M_FILTERS])
    h_conv3_L_flat = tf.reshape(h_conv3_L, [-1, NUM_CONV3_FLAT_L_FILTERS])
    h_conv2_S_flat = tf.reshape(h_conv2_S, [-1, NUM_CONV2_FLAT_S_FILTERS])
    h_conv2_M_flat = tf.reshape(h_conv2_M, [-1, NUM_CONV2_FLAT_M_FILTERS])
    h_conv2_L_flat = tf.reshape(h_conv2_L, [-1, NUM_CONV2_FLAT_L_FILTERS])

    h_conv_flat = tf.concat(values=[h_conv3_S_flat, h_conv3_M_flat, h_conv3_L_flat, h_conv2_S_flat, h_conv2_M_flat, h_conv2_L_flat], axis=1)

    acti_mode_fc = 5
    h_fc1_64 = full_connect(h_conv_flat, NUM_CONVLAYER_FLAT_FILTERS, NUM_DENLAYER1_FEATURES_64, acti_mode=acti_mode_fc, name_w='h_fc1__64__w', name_b='h_fc1__64__b')
    h_fc1_64 = tf.concat([h_fc1_64, qp_norm], axis=1)
    h_fc2_64 = full_connect(h_fc1_64, NUM_DENLAYER1_FEATURES_64 + NUM_EXT_FEATURES, NUM_DENLAYER2_FEATURES_64, acti_mode=acti_mode_fc, name_w='h_fc2__64__w', name_b='h_fc2__64__b')
    h_fc2_64 = tf.concat([h_fc2_64, qp_norm], axis=1)
    y_conv_flat_64 = full_connect(h_fc2_64, NUM_DENLAYER2_FEATURES_64 + NUM_EXT_FEATURES, 1, acti_mode=2, name_w='y_conv_flat__64__w', name_b='y_conv_flat__64__b')

    h_fc1_32 = full_connect(h_conv_flat, NUM_CONVLAYER_FLAT_FILTERS, NUM_DENLAYER1_FEATURES_32, acti_mode=acti_mode_fc, name_w='h_fc1__32__w', name_b='h_fc1__32__b')
    h_fc1_32 = tf.concat([h_fc1_32, qp_norm], axis=1)
    h_fc2_32 = full_connect(h_fc1_32, NUM_DENLAYER1_FEATURES_32 + NUM_EXT_FEATURES, NUM_DENLAYER2_FEATURES_32, acti_mode=acti_mode_fc, name_w='h_fc2__32__w', name_b='h_fc2__32__b')
    h_fc2_32 = tf.concat([h_fc2_32, qp_norm], axis=1)
    y_conv_flat_32 = full_connect(h_fc2_32, NUM_DENLAYER2_FEATURES_32 + NUM_EXT_FEATURES, 4, acti_mode=2, name_w='y_conv_flat__32__w', name_b='y_conv_flat__32__b')

    h_fc1_16 = full_connect(h_conv_flat, NUM_CONVLAYER_FLAT_FILTERS, NUM_DENLAYER1_FEATURES_16, acti_mode=acti_mode_fc, name_w='h_fc1__16__w', name_b='h_fc1__16__b')
    h_fc1_16 = tf.concat([h_fc1_16, qp_norm], axis=1)
    h_fc2_16 = full_connect(h_fc1_16, NUM_DENLAYER1_FEATURES_16 + NUM_EXT_FEATURES, NUM_DENLAYER2_FEATURES_16, acti_mode=acti_mode_fc, name_w='h_fc2__16__w', name_b='h_fc2__16__b')
    h_fc2_16 = tf.concat([h_fc2_16, qp_norm], axis=1)
    y_conv_flat_16 = full_connect(h_fc2_16, NUM_DENLAYER2_FEATURES_16 + NUM_EXT_FEATURES, 16, acti_mode=2, name_w='y_conv_flat__16__w', name_b='y_conv_flat__16__b')

    return y_conv_flat_64, y_conv_flat_32, y_conv_flat_16

# --- Main Inference Script ---

# --- Configuration ---
QP_VALUE = 32
NUM_SAMPLES_TO_TEST = 1000
DATA_FILE = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/ETH-CNN_Training_AI/Data/AI_Valid_143925.dat_shuffled'
MODEL_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/ETH-CNN_Training_AI/Models/model_20181226_045506_1000000_qp32.dat'
NUM_SAMPLE_LENGTH = 4992 # From your data structure analysis
IMAGE_DATA_SIZE = 4096
LABEL_BLOCK_START = 4160
THRESHOLDS = [0.5, 1.5, 2.5]

# --- Functions to interpret labels ---
def get_ground_truth_splits(label_16_bytes):
    depths = np.array(label_16_bytes).reshape(4, 4)
    split_64 = 1 if np.mean(depths) > THRESHOLDS[0] else 0
    splits_32 = []
    if split_64:
        for i in range(2):
            for j in range(2):
                sub_block = depths[i*2:(i+1)*2, j*2:(j+1)*2]
                split = 1 if np.mean(sub_block) > THRESHOLDS[1] else 0
                splits_32.append(split)
    else:
        splits_32 = [0, 0, 0, 0]
    splits_16 = []
    flat_depths = depths.flatten()
    for i in range(16):
        parent_32_index = (i // 8) * 2 + ((i % 4) // 2)
        if split_64 and splits_32[parent_32_index]:
            split = 1 if flat_depths[i] > THRESHOLDS[2] else 0
            splits_16.append(split)
        else:
            splits_16.append(0)
    return split_64, splits_32, splits_16

def get_predicted_splits(pred_64, pred_32, pred_16, threshold=0.5):
    split_64 = 1 if pred_64[0][0] > threshold else 0
    splits_32 = [1 if p > threshold else 0 for p in pred_32[0]]
    splits_16 = [1 if p > threshold else 0 for p in pred_16[0]]
    return split_64, splits_32, splits_16

# --- Main execution ---
def main():
    print("--- Starting Inference Script ---")
    print("Model: {}".format(MODEL_PATH))
    print("Data File: {}".format(DATA_FILE))
    print("QP Value: {}".format(QP_VALUE))
    print("Samples to Test: {}\n".format(NUM_SAMPLES_TO_TEST))

    # --- 1. Build TensorFlow Graph ---
    print("1. Building TensorFlow graph...")
    tf.reset_default_graph()
    x_placeholder = tf.placeholder("float", [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    qp_placeholder = tf.placeholder("float", [None, NUM_EXT_FEATURES])
    y_conv_64, y_conv_32, y_conv_16 = build_net(x_placeholder, qp_placeholder)
    saver = tf.train.Saver()
    print("Graph built successfully.\n")

    # --- 2. Start Session and Load Model ---
    with tf.Session() as sess:
        print("2. Loading model weights from {}...".format(MODEL_PATH))
        try:
            saver.restore(sess, MODEL_PATH)
            print("Model restored successfully.\n")
        except Exception as e:
            print("Error restoring model: {}".format(e))
            print("Please ensure the model files exist at the specified path and are compatible with TF1.x.")
            return

        # --- 3. Read Data and Perform Inference ---
        print("3. Starting inference loop...")
        total_correct_64 = 0
        total_correct_32 = 0
        total_correct_16 = 0
        samples_processed = 0

        try:
            with open(DATA_FILE, 'rb') as f:
                while samples_processed < NUM_SAMPLES_TO_TEST:
                    sample_bytes = f.read(NUM_SAMPLE_LENGTH)
                    if not sample_bytes or len(sample_bytes) != NUM_SAMPLE_LENGTH:
                        print("\nReached end of data file.")
                        break

                    image_data = np.frombuffer(sample_bytes[:IMAGE_DATA_SIZE], dtype=np.uint8)
                    image_data = image_data.reshape(1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS).astype(np.float32)
                    qp_data = np.array([[QP_VALUE]]).astype(np.float32)
                    label_start = LABEL_BLOCK_START + QP_VALUE * NUM_LABEL_BYTES
                    label_end = label_start + NUM_LABEL_BYTES
                    ground_truth_label_bytes = np.frombuffer(sample_bytes[label_start:label_end], dtype=np.uint8)

                    pred_64, pred_32, pred_16 = sess.run(
                        [y_conv_64, y_conv_32, y_conv_16],
                        feed_dict={x_placeholder: image_data, qp_placeholder: qp_data}
                    )

                    gt_split_64, gt_splits_32, gt_splits_16 = get_ground_truth_splits(ground_truth_label_bytes)
                    pred_split_64, pred_splits_32, pred_splits_16 = get_predicted_splits(pred_64, pred_32, pred_16)

                    print("--- Sample {} ---".format(samples_processed + 1))
                    print("  GT 64->32 Split:  {} \t| Prediction: {}".format(gt_split_64, pred_split_64))
                    print("  GT 32->16 Splits: {} \t| Prediction: {}".format(gt_splits_32, pred_splits_32))
                    print("  GT 16->8 Splits:  {} \t| Prediction: {}".format(gt_splits_16, pred_splits_16))

                    if gt_split_64 == pred_split_64:
                        total_correct_64 += 1
                    if gt_split_64 == 1:
                        correct_32 = sum(1 for gt, pred in zip(gt_splits_32, pred_splits_32) if gt == pred)
                        total_correct_32 += (correct_32 / 4.0)
                    elif gt_split_64 == 0:
                        total_correct_32 += 1.0 if sum(pred_splits_32) == 0 else 0

                    valid_16_parents = sum(gt_splits_32)
                    if gt_split_64 == 1 and valid_16_parents > 0:
                        correct_16, num_valid_16 = 0, 0
                        for i in range(16):
                            parent_32_index = (i // 8) * 2 + ((i % 4) // 2)
                            if gt_splits_32[parent_32_index] == 1:
                                num_valid_16 += 1
                                if gt_splits_16[i] == pred_splits_16[i]:
                                    correct_16 += 1
                        if num_valid_16 > 0:
                            total_correct_16 += (correct_16 / num_valid_16)
                    else:
                        total_correct_16 += 1.0 if sum(pred_splits_16) == 0 else 0

                    samples_processed += 1
        except FileNotFoundError:
            print("ERROR: Data file not found at {}".format(DATA_FILE))
            return
        except Exception as e:
            print("An error occurred during processing: {}".format(e))

        if samples_processed > 0:
            accuracy_64 = (total_correct_64 / samples_processed) * 100
            accuracy_32 = (total_correct_32 / samples_processed) * 100
            accuracy_16 = (total_correct_16 / samples_processed) * 100

            print("\n--- Final Results ---")
            print("Total Samples Processed: {}".format(samples_processed))
            print("Accuracy for 64x64 split decision: {:.2f}%".format(accuracy_64))
            print("Average Accuracy for 32x32 split decisions: {:.2f}%".format(accuracy_32))
            print("Average Accuracy for 16x16 split decisions: {:.2f}%".format(accuracy_16))
            overall_accuracy = (accuracy_64 + accuracy_32 + accuracy_16) / 3.0
            print("Overall (average of L1, L2, L3) Accuracy: {:.2f}%".format(overall_accuracy))

if __name__ == '__main__':
    main()
