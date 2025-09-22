import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import glob
import data_info as di # Uses the data_info.py file you provided

# ==============================================================================
# --- 1. Configuration: SET THESE VALUES ---
# ==============================================================================

YUV_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/AI_YUV&Info/Info&YUV/AI_YUV'
INFO_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/AI_YUV&Info/Info&YUV/AI_Info'
VIDEO_FILENAME = 'IntraValid_4928x3264'
FRAME_TO_EVALUATE = 0
QP_VALUE = 32
MODEL_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/ETH-CNN_Training_AI/Models/model_20181226_045506_1000000_qp32.dat'

# ==============================================================================

# ==============================================================================
# --- 2. Helper Functions (Adapted from previous scripts) ---
# ==============================================================================

class FrameYUV:
    def __init__(self, Y, U, V):
        self._Y = Y

def read_YUV420_frame(fid, width, height, frame_index):
    frame_bytes = (width * height * 3) // 2
    fid.seek(frame_index * frame_bytes)
    Y_buf = fid.read(width * height)
    if not Y_buf: return None
    Y = np.frombuffer(Y_buf, dtype=np.uint8).reshape([height, width])
    return FrameYUV(Y, None, None)

def read_info_frame(fid, width, height, frame_index):
    num_units_wide = width // 16
    num_units_high = height // 16
    frame_bytes = num_units_wide * num_units_high
    fid.seek(frame_index * frame_bytes)
    info_buf = fid.read(frame_bytes)
    if not info_buf: return None
    info = np.frombuffer(info_buf, dtype=np.uint8).reshape([num_units_high, num_units_wide])
    return info

THRESHOLDS = [0.5, 1.5, 2.5]
def get_ground_truth_splits_from_ctu(ctu_label_map):
    depths = np.array(ctu_label_map).reshape(4, 4)
    split_64 = 1 if np.mean(depths) > THRESHOLDS[0] else 0
    splits_32 = []
    if split_64:
        for i in range(2):
            for j in range(2):
                sub_block = depths[i*2:(i+1)*2, j*2:(j+1)*2]
                splits_32.append(1 if np.mean(sub_block) > THRESHOLDS[1] else 0)
    else:
        splits_32 = [0, 0, 0, 0]
    splits_16 = []
    flat_depths = depths.flatten()
    for i in range(16):
        parent_32_index = (i // 8) * 2 + ((i % 4) // 2)
        if split_64 and splits_32[parent_32_index]:
            splits_16.append(1 if flat_depths[i] > THRESHOLDS[2] else 0)
        else:
            splits_16.append(0)
    return split_64, splits_32, splits_16

def get_predicted_splits(pred_64, pred_32, pred_16, threshold=0.5):
    split_64 = 1 if pred_64[0][0] > threshold else 0
    splits_32 = [1 if p > threshold else 0 for p in pred_32[0]]
    splits_16 = [1 if p > threshold else 0 for p in pred_16[0]]
    return split_64, splits_32, splits_16

def convert_splits_to_depth_map(pred_64, pred_32, pred_16):
    depth_map = np.zeros((4, 4), dtype=np.uint8)
    if pred_64 == 0:
        return depth_map
    for q in range(4):
        q_r, q_c = q // 2, q % 2
        if pred_32[q] == 0:
            depth_map[q_r*2 : q_r*2+2, q_c*2 : q_c*2+2] = 1
        else:
            for b in range(4):
                b_r, b_c = b // 2, b % 2
                block_16_index = q * 4 + b
                if pred_16[block_16_index] == 0:
                    depth_map[q_r*2+b_r, q_c*2+b_c] = 2
                else:
                    depth_map[q_r*2+b_r, q_c*2+b_c] = 3
    return depth_map

# ==============================================================================
# --- 3. Main Evaluation Logic ---
# ==============================================================================
def evaluate_single_frame():
    # --- Find video info and file paths ---
    try:
        video_index = di.YUV_NAME_LIST_FULL.index(VIDEO_FILENAME)
        width = di.YUV_WIDTH_LIST_FULL[video_index]
        height = di.YUV_HEIGHT_LIST_FULL[video_index]
    except ValueError:
        print("Error: Video '{}' not found in data_info.py.".format(VIDEO_FILENAME))
        return

    yuv_file_path = os.path.join(YUV_PATH, VIDEO_FILENAME + '.yuv')
    info_file_pattern = os.path.join(INFO_PATH, 'Info*_{}_{}qp{}*CUDepth.dat'.format(VIDEO_FILENAME, '*', QP_VALUE))
    info_files = glob.glob(info_file_pattern)

    if not info_files:
        print("Error: Could not find Info file for QP {}. Pattern: {}".format(QP_VALUE, info_file_pattern))
        return
    info_file_path = info_files[0]

    print("--- Evaluating Single Frame with CNN Model ---")
    print("Video: {}, Frame: {}, QP: {}".format(VIDEO_FILENAME, FRAME_TO_EVALUATE, QP_VALUE))
    print("Model: {}\n".format(MODEL_PATH))

    # --- Read the full frame data ---
    try:
        with open(yuv_file_path, 'rb') as fid_yuv:
            frame_yuv = read_YUV420_frame(fid_yuv, width, height, FRAME_TO_EVALUATE)
        with open(info_file_path, 'rb') as fid_info:
            full_cu_depth_map = read_info_frame(fid_info, width, height, FRAME_TO_EVALUATE)
        if frame_yuv is None or full_cu_depth_map is None:
            print("Error: Frame #{} not found in the files.".format(FRAME_TO_EVALUATE))
            return
    except FileNotFoundError as e:
        print("Error: File not found. Please check paths.\n{}".format(e))
        return

    tf.reset_default_graph()
    x = tf.placeholder("float", [None, 64, 64, 1])
    qp = tf.placeholder("float", [None, 1])
    from net_CTU64 import net as build_actual_net
    isdrop_placeholder = tf.placeholder("float")
    _, _, _, y_conv_64_op, y_conv_32_op, y_conv_16_op, _, _, _, _, _, _ = build_actual_net(
        x, tf.placeholder("float", [None, 16]), qp, isdrop_placeholder,
        tf.placeholder("float"), 0.01, 0.9, 1, 1
    )
    model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(model_vars)

    with tf.Session() as sess:
        try:
            saver.restore(sess, MODEL_PATH)
            print("CNN model restored successfully.\n")
        except Exception as e:
            print("Error restoring CNN model: {}".format(e))
            return

        all_gt_splits = []
        all_pred_splits = []
        predicted_cu_depth_map = np.zeros_like(full_cu_depth_map)
        num_ctus_h, num_ctus_w = height // 64, width // 64

        for y_ctu in range(num_ctus_h):
            for x_ctu in range(num_ctus_w):
                ctu_label_map = full_cu_depth_map[y_ctu*4:(y_ctu+1)*4, x_ctu*4:(x_ctu+1)*4]
                gt_splits = get_ground_truth_splits_from_ctu(ctu_label_map)
                all_gt_splits.append(gt_splits)

                ctu_image = frame_yuv._Y[y_ctu*64:(y_ctu+1)*64, x_ctu*64:(x_ctu+1)*64]
                ctu_image_batch = ctu_image.reshape(1, 64, 64, 1).astype(np.float32)
                qp_batch = np.array([[QP_VALUE]]).astype(np.float32)
                feed_dict = {x: ctu_image_batch, qp: qp_batch, isdrop_placeholder: 0.0}
                pred_64, pred_32, pred_16 = sess.run([y_conv_64_op, y_conv_32_op, y_conv_16_op], feed_dict=feed_dict)
                pred_splits = get_predicted_splits(pred_64, pred_32, pred_16)
                all_pred_splits.append(pred_splits)
                ctu_predicted_depths = convert_splits_to_depth_map(pred_splits[0], pred_splits[1], pred_splits[2])
                predicted_cu_depth_map[y_ctu*4:(y_ctu+1)*4, x_ctu*4:(x_ctu+1)*4] = ctu_predicted_depths

    print("--- Ground Truth Partition Map ---")
    print(full_cu_depth_map)
    print("\n" + "="*50 + "\n")
    print("--- Model Predicted Partition Map ---")
    print(predicted_cu_depth_map)
    print("\n")

    total_ctus, correct_64 = len(all_gt_splits), 0
    correct_32, total_32 = 0, 0
    correct_16, total_16 = 0, 0

    for i in range(total_ctus):
        gt_64, gt_32, gt_16 = all_gt_splits[i]
        pred_64, pred_32, pred_16 = all_pred_splits[i]

        if gt_64 == pred_64: correct_64 += 1
        if gt_64 == 1:
            total_32 += 4
            correct_32 += sum(1 for gt, pred in zip(gt_32, pred_32) if gt == pred)
        for j in range(4):
            if gt_32[j] == 1:
                total_16 += 4
                start_idx, end_idx = j * 4, j * 4 + 4
                correct_16 += sum(1 for gt, pred in zip(gt_16[start_idx:end_idx], pred_16[start_idx:end_idx]) if gt == pred)

    acc_64 = (correct_64 / total_ctus) * 100 if total_ctus > 0 else 0
    acc_32 = (correct_32 / total_32) * 100 if total_32 > 0 else 100
    acc_16 = (correct_16 / total_16) * 100 if total_16 > 0 else 100

    print("--- Frame Accuracy Results ---")
    print("L1 (64x64 Split) Accuracy: {:.2f}% ({}/{}) correct CTUs".format(acc_64, correct_64, total_ctus))
    print("L2 (32x32 Split) Accuracy: {:.2f}% ({}/{}) correct blocks".format(acc_32, correct_32, total_32))
    print("L3 (16x16 Split) Accuracy: {:.2f}% ({}/{}) correct blocks".format(acc_16, correct_16, total_16))
    overall_accuracy = (acc_64 + acc_32 + acc_16) / 3.0
    print("Overall (average of L1, L2, L3) Accuracy: {:.2f}%".format(overall_accuracy))

if __name__ == "__main__":
    if not os.path.exists("net_CTU64.py"):
        print("ERROR: This script requires 'net_CTU64.py' to be in the same directory.")
    else:
        evaluate_single_frame()
