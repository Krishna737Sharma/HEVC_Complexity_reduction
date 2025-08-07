# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, shutil
import numpy as np
import random
import net_ViT as nt_vit
import math
import time

# **YOUR ACTUAL PATHS**
VIT_REPO_PATH = "/home/ai-iitkgp/Downloads/HEVC_Intra_Models-ViT"
VIT_MODEL_DIR = "ViT_2.3M"

NUM_CHANNELS = nt_vit.NUM_CHANNELS
NUM_EXT_FEATURES = nt_vit.NUM_EXT_FEATURES
NUM_LABEL_BYTES = nt_vit.NUM_LABEL_BYTES
IMAGE_SIZE = nt_vit.IMAGE_SIZE
SAVE_FILE = 'cu_depth.dat'


def print_current_line(str):
    print('\r' + str, end='')
    sys.stdout.flush()


def get_file_size(path):
    try:
        size = os.path.getsize(path)
        return size
    except Exception as err:
        print(err)


def get_Y_for_one_frame(f, frame_width, frame_height, image_size):
    y_buf = f.read(frame_width * frame_height)
    uv_temp = f.read(frame_width * frame_height // 2)
    data = np.frombuffer(y_buf, dtype=np.uint8)
    data = data.reshape(1, frame_height * frame_width)
    valid_height = math.ceil(frame_height / image_size) * image_size
    valid_width = math.ceil(frame_width / image_size) * image_size
    data = data.reshape(frame_height, frame_width)
    if valid_height > frame_height:
        data = np.concatenate((data, np.zeros((valid_height - frame_height, frame_width))), axis=0)
    if valid_width > frame_width:
        data = np.concatenate((data, np.zeros((valid_height, valid_width - frame_width))), axis=1)

    return data


def get_y_conv_on_large_data_vit(input_image, qp_seq):
    """Get ViT predictions for large batches"""
    batch_size = np.shape(input_image)[0]
    y_conv_out = np.zeros((batch_size, 21))
    sub_batch_size = 2176

    for i in range(int(math.ceil(batch_size / float(sub_batch_size)))):
        index_start = i * sub_batch_size
        index_end = (i + 1) * sub_batch_size
        if index_end > batch_size:
            index_end = batch_size

        sub_batch = input_image[index_start:index_end, :]
        y_flat_64_temp, y_flat_32_temp, y_flat_16_temp = nt_vit.net_vit_predictions(sub_batch, qp_seq)

        y_conv_out[index_start:index_end, :] = np.concatenate([y_flat_64_temp, y_flat_32_temp, y_flat_16_temp], axis=1)

        if i % 10 == 0:
            print("  Processed sub-batch {}/{}".format(i + 1, int(math.ceil(batch_size / float(sub_batch_size)))))

    return y_conv_out


def get_prob(yuv_name, image_size, save_file, qp_seq, n_frames_start, n_frames_end, frame_width, frame_height):
    n_frames = n_frames_end - n_frames_start
    prob = np.zeros((n_frames * int(math.ceil(frame_height / float(image_size))) * int(math.ceil(frame_width / float(image_size))), 21))
    f = open(yuv_name, 'rb')
    f_out = open(save_file, 'wb')

    valid_height = int(math.ceil(frame_height / float(image_size))) * image_size
    valid_width = int(math.ceil(frame_width / float(image_size))) * image_size

    for k in range(n_frames_start):
        luma = get_Y_for_one_frame(f, frame_width, frame_height, image_size)

    for k in range(n_frames):
        valid_luma = get_Y_for_one_frame(f, frame_width, frame_height, image_size)
        batch_size = (valid_height // image_size) * (valid_width // image_size)
        input_batch = np.zeros((batch_size, image_size, image_size, NUM_CHANNELS))

        index = 0
        ystart = 0
        while ystart < frame_height:
            xstart = 0
            while xstart < frame_width:
                CU_input = valid_luma[ystart: ystart + image_size, xstart: xstart + image_size]
                input_batch[index] = np.reshape(CU_input, [1, image_size, image_size, NUM_CHANNELS])
                index += 1
                xstart += image_size
            ystart += image_size

        input_batch = input_batch.astype(np.float32)

        y_conv_out = get_y_conv_on_large_data_vit(input_batch, qp_seq)
        prob[k * batch_size: (k + 1) * batch_size] = y_conv_out

        print_current_line('%s  frame %d/%d  %dx%d (ViT)' % (yuv_name, k + 1, n_frames, frame_width, frame_height))

    print('')
    prob_arr = np.reshape(prob.astype(np.float32),
                          [1, (n_frames * (valid_height // image_size) * (valid_width // image_size) * 21)])

    f_out.write(prob_arr)
    f.close()
    f_out.close()


# Main execution
assert len(sys.argv) == 5
yuv_file = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])
qp_seq = int(sys.argv[4])

print("Starting ViT-based CU depth prediction...")
print("Video: {}".format(yuv_file))
print("Resolution: {}x{}".format(width, height))
print("QP: {}".format(qp_seq))

# Initialize ViT predictor with your actual paths
model_path = os.path.join(VIT_REPO_PATH, VIT_MODEL_DIR, "best_vit_model.pth")
print("Loading ViT model from: {}".format(model_path))

nt_vit.initialize_vit_predictor(VIT_REPO_PATH, model_path)

file_bytes = get_file_size(yuv_file)
frame_bytes = width * height * 3 // 2
assert (file_bytes % frame_bytes == 0)

n_frames_start = 0
n_frames_end = file_bytes // frame_bytes

print("Total frames to process: {}".format(n_frames_end))

t1 = time.time()
get_prob(yuv_file, IMAGE_SIZE, SAVE_FILE, qp_seq, n_frames_start, n_frames_end, width, height)
t2 = time.time()
print('--------\n\nViT Prediction Time: {:.3f} sec.\n\n--------'.format(float(t2 - t1)))
