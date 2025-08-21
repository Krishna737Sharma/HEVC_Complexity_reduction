import os, sys, shutil
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, savefig
import matplotlib.patches as patches
import time
import timeit
import numpy as np
import string
import math
import input_data as input_data
import tensorflow as tf

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

# decide to run with which type of device
# 0:CPU  1:GPU (limited space)  2:GPU (unlimited space)
DEVICE_MODE = 1
if DEVICE_MODE == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # GPU is disabled
    sess = tf.Session()
elif DEVICE_MODE == 1:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    sess = tf.Session(config=config)
elif DEVICE_MODE == 2:
    sess = tf.Session()

IMAGE_SIZE = input_data.IMAGE_SIZE
NUM_CHANNELS = input_data.NUM_CHANNELS
NUM_EXT_FEATURES = input_data.NUM_EXT_FEATURES
NUM_LABEL_BYTES = input_data.NUM_LABEL_BYTES
DATA_SWITCH = input_data.DATA_SWITCH

# --- MODIFICATIONS FOR EVALUATION ---
IS_RELOAD = True  # Set to True to load the pre-trained model
ITER_TIMES = 0  # Set to 0 to skip training and run only the final evaluation
# ------------------------------------

RUN_MODE = 0  # 0: train  1: evaluate
LEARNING_RATE_INIT = 0.01  # initial learning rate
IS_RESET_LR = False  # when IS_RELOAD == True, whether to set learning rate as LEARNING_RATE_INIT
MOMENTUM_INIT = 0.9  # initial momentum

# These parameters are for training and can be ignored for evaluation
ITER_TIMES_PER_CHANGE_RATE = 250000
LEARNING_RATE_DECAY_RATIO = 0.3163

PARTLY_TUNING_MODE = input_data.PARTLY_TUNING_MODE
EVALUATE_QP_THR_LIST = input_data.EVALUATE_QP_THR_LIST
MODEL_NAME = input_data.MODEL_NAME

BATCH_SIZE = 64

NUM_TRAIN_PART = 5000
NUM_VALID_PART = 5000

ITER_TIMES_PER_PRINT = 1000
ITER_TIMES_PER_EVALUATE = 1000
ITER_TIMES_PER_CHANGE_TRAINSET = 2000
ITER_TIMES_PER_CHANGE_VALIDSET = 10000
ITER_TIMES_PER_SAVE = 50000

ITER_TIMES_PER_EVALUATE_LARGE_BATCH = 50000


def is_sep_64(y_one_sample, thr):
    depth_mean = np.mean(y_one_sample)
    if depth_mean > thr:
        return 1
    else:
        return 0


def is_sep_64_by_bit(y_one_sample):
    if (y_one_sample[0] & 4) > 0:
        return 1
    else:
        return 0


def is_sep_32(y_one_sample, thr):
    depth_mean = np.mean(y_one_sample)
    if depth_mean > thr:
        return 1
    else:
        return 0


def is_sep_32_by_bit(y_one_sample):
    if (y_one_sample[0] & 2) > 0:
        return 1
    else:
        return 0


def is_sep_16(y_one_sample, thr):
    if y_one_sample > thr:
        return 1
    else:
        return 0


def is_sep_16_by_bit(y_one_sample):
    if (y_one_sample & 1) > 0:
        return 1
    else:
        return 0


def get_class_matrices(y_truth, y_predict_64, y_predict_32, y_predict_16, thr_list):
    matrix_64 = [[0, 0], [0, 0]]
    matrix_32 = [[0, 0], [0, 0]]
    matrix_16 = [[0, 0], [0, 0]]
    assert y_truth.shape[0] == y_predict_16.shape[0]
    num_samples = y_truth.shape[0]
    index_32_list = [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]

    for i in range(num_samples):
        class_64_truth = is_sep_64(y_truth[i], input_data.DEFAULT_THR_LIST[0])
        class_64_predict = is_sep_64(y_predict_64[i], thr_list[0])
        matrix_64[class_64_truth][class_64_predict] += 1
        if class_64_truth == 1:
            for j in range(4):
                class_32_truth = is_sep_32(y_truth[i][index_32_list[j]], input_data.DEFAULT_THR_LIST[1])
                class_32_predict = is_sep_32(y_predict_32[i][j], thr_list[1])
                matrix_32[class_32_truth][class_32_predict] += 1
                if class_32_truth == 1:
                    for k in range(4):
                        class_16_truth = is_sep_16(y_truth[i][index_32_list[j][k]], input_data.DEFAULT_THR_LIST[2])
                        class_16_predict = is_sep_16(y_predict_16[i][index_32_list[j][k]], thr_list[2])
                        matrix_16[class_16_truth][class_16_predict] += 1

    return matrix_64, matrix_32, matrix_16


def get_tendency_2x2(matrix_2x2):
    if matrix_2x2[0][1] == 0 and matrix_2x2[1][0] == 0:
        return 0
    elif matrix_2x2[0][1] == 0 or matrix_2x2[1][1] == 0:
        return -100
    elif matrix_2x2[1][0] == 0 or matrix_2x2[0][0] == 0:
        return 100
    else:
        return -math.log10((matrix_2x2[0][0] / matrix_2x2[0][1]) / (matrix_2x2[1][1] / matrix_2x2[1][0]))


def get_accuracy_on_large_data_interqp(f, data_set, qp_thr_list, thr_list, plot_tag_set=None):
    range_stat = input_data.RangingStatistics(qp_thr_list, 'scalar')
    count_list, stat_index = range_stat.feed_data_list(data_set.qps, is_select=True)
    segment_names = range_stat.get_segment_names('QP')
    for i in range(len(qp_thr_list) + 1):
        if count_list[i] > 0:
            fprint(f, '------------------------------')
            fprint(f, segment_names[i])
            accuracy, _, _, _, accuracy_list, tendency_list = get_accuracy_on_large_data(f,
                                                                                         data_set.images[stat_index[i]],
                                                                                         data_set.qps[stat_index[i]],
                                                                                         data_set.labels[stat_index[i]],
                                                                                         thr_list=thr_list,
                                                                                         is_get_y_predict=True,
                                                                                         is_print=True)


def get_accuracy_on_large_data(f, images, qps, labels, thr_list, is_get_y_predict=False, is_print=False):
    length = np.shape(images)[0]
    PRE_BATCH_SIZE = 5000

    accuracy_total = np.zeros((3))
    y_predict_16 = []
    y_predict_32 = []
    y_predict_64 = []
    matrix_64_sum = [[0, 0], [0, 0]]
    matrix_32_sum = [[0, 0], [0, 0]]
    matrix_16_sum = [[0, 0], [0, 0]]

    for i in range(int(math.ceil(length / PRE_BATCH_SIZE))):
        index_start = PRE_BATCH_SIZE * i
        index_end = PRE_BATCH_SIZE * (i + 1)
        if index_end > length:
            index_end = length
        y_predict_temp_64, y_predict_temp_32, y_predict_temp_16, accuracy_temp = sess.run(
            [y_conv_64, y_conv_32, y_conv_16, accuracy_list],
            feed_dict={x: images[index_start:index_end], y_: labels[index_start:index_end],
                       qp: qps[index_start:index_end], isdrop: 0})
        if is_get_y_predict == True:
            if not isinstance(y_predict_16, np.ndarray):
                y_predict_16 = y_predict_temp_16
            else:
                y_predict_16 = np.vstack((y_predict_16, y_predict_temp_16))
            if not isinstance(y_predict_32, np.ndarray):
                y_predict_32 = y_predict_temp_32
            else:
                y_predict_32 = np.vstack((y_predict_32, y_predict_temp_32))
            if not isinstance(y_predict_64, np.ndarray):
                y_predict_64 = y_predict_temp_64
            else:
                y_predict_64 = np.vstack((y_predict_64, y_predict_temp_64))
        accuracy_total += accuracy_temp * (index_end - index_start)
        matrix_64_temp, matrix_32_temp, matrix_16_temp = get_class_matrices(labels[index_start:index_end],
                                                                            y_predict_temp_64, y_predict_temp_32,
                                                                            y_predict_temp_16, thr_list)
        matrix_64_sum = np.add(matrix_64_sum, matrix_64_temp)
        matrix_32_sum = np.add(matrix_32_sum, matrix_32_temp)
        matrix_16_sum = np.add(matrix_16_sum, matrix_16_temp)

    accuracy_64 = (matrix_64_sum[0][0] + matrix_64_sum[1][1]) / np.sum(matrix_64_sum)
    accuracy_32 = (matrix_32_sum[0][0] + matrix_32_sum[1][1]) / np.sum(matrix_32_sum)
    accuracy_16 = (matrix_16_sum[0][0] + matrix_16_sum[1][1]) / np.sum(matrix_16_sum)

    tendency_64 = get_tendency_2x2(matrix_64_sum)
    tendency_32 = get_tendency_2x2(matrix_32_sum)
    tendency_16 = get_tendency_2x2(matrix_16_sum)
    if is_print == True:
        print(matrix_64_sum)
        print(matrix_32_sum)
        print(matrix_16_sum)
        if f != None:
            fprint(f, 'accuracy = %lf, %lf, %lf' % (accuracy_64, accuracy_32, accuracy_16))
            fprint(f, 'tendency = %lf, %lf, %lf' % (tendency_64, tendency_32, tendency_16))
    accuracy_all = accuracy_total / length
    accuracy_valid = [accuracy_64, accuracy_32, accuracy_16]
    tendency_valid = [tendency_64, tendency_32, tendency_16]

    return accuracy_all, y_predict_64, y_predict_32, y_predict_16, accuracy_valid, tendency_valid


def evaluate_loss_accuracy(step, learning_rate_value):
    global train_accuracy_list, valid_accuracy_list, train_loss_list, valid_loss_list
    train_batch = data_sets.train.next_batch_random(NUM_TRAIN_PART)
    valid_batch = data_sets.validation.next_batch_random(NUM_VALID_PART)

    train_y_predict_temp_64, train_y_predict_temp_32, train_y_predict_temp_16, train_accuracy_list_c, train_loss_list_c = sess.run(
        [y_conv_64, y_conv_32, y_conv_16, accuracy_list, loss_list],
        feed_dict={x: train_batch[0], y_: train_batch[1], qp: train_batch[2], isdrop: 0})
    valid_y_predict_temp_64, valid_y_predict_temp_32, valid_y_predict_temp_16, valid_accuracy_list_c, valid_loss_list_c = sess.run(
        [y_conv_64, y_conv_32, y_conv_16, accuracy_list, loss_list],
        feed_dict={x: valid_batch[0], y_: valid_batch[1], qp: valid_batch[2], isdrop: 0})

    print("%s step %d: loss=[[%.3f %.3f %.3f] [%.3f %.3f %.3f]], accu=[[%.3f %.3f %.3f] [%.3f %.3f %.3f]], lr=%g" % (
        get_time_str(), step, train_loss_list_c[0], train_loss_list_c[1], train_loss_list_c[2], valid_loss_list_c[0],
        valid_loss_list_c[1], valid_loss_list_c[2], train_accuracy_list_c[0], train_accuracy_list_c[1],
        train_accuracy_list_c[2], valid_accuracy_list_c[0], valid_accuracy_list_c[1], valid_accuracy_list_c[2],
        learning_rate_value))
    train_accuracy_list.append(list(train_accuracy_list_c))
    valid_accuracy_list.append(list(valid_accuracy_list_c))
    train_loss_list.append(list(train_loss_list_c))
    valid_loss_list.append(list(valid_loss_list_c))
    step_list.append(step)

    matrix_64, matrix_32, matrix_16 = get_class_matrices(train_batch[1], train_y_predict_temp_64,
                                                         train_y_predict_temp_32, train_y_predict_temp_16,
                                                         [0.5, 0.5, 0.5])
    train_tendency_64 = get_tendency_2x2(matrix_64)
    train_tendency_32 = get_tendency_2x2(matrix_32)
    train_tendency_16 = get_tendency_2x2(matrix_16)
    matrix_64, matrix_32, matrix_16 = get_class_matrices(valid_batch[1], valid_y_predict_temp_64,
                                                         valid_y_predict_temp_32, valid_y_predict_temp_16,
                                                         [0.5, 0.5, 0.5])
    valid_tendency_64 = get_tendency_2x2(matrix_64)
    valid_tendency_32 = get_tendency_2x2(matrix_32)
    valid_tendency_16 = get_tendency_2x2(matrix_16)
    print(
        'tendency = [[%.3f, %.3f, %.3f] [%.3f, %.3f, %.3f]]' % (train_tendency_64, train_tendency_32, train_tendency_16,
                                                                valid_tendency_64, valid_tendency_32,
                                                                valid_tendency_16))
    train_tendency_list_c = [train_tendency_64, train_tendency_32, train_tendency_16]
    valid_tendency_list_c = [valid_tendency_64, valid_tendency_32, valid_tendency_16]
    train_tendency_list.append(list(train_tendency_list_c))
    valid_tendency_list.append(list(valid_tendency_list_c))


def reload_loss_and_accuracy():
    # In evaluation mode, we don't need to reload past metrics.
    print("Evaluation mode: Skipping reload of historical loss and accuracy.")
    return 0


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))


def fprint(f, str_to_print):
    print(str_to_print)
    f.write(str_to_print + '\r\n')


def evaluate():
    log_filename = 'Models/evaluation_log_%s.dat' % get_time_str()
    f = open(log_filename, 'w+')

    # Changed from f-string for Python 3.5 compatibility
    print("Starting evaluation... Log will be saved to {}".format(log_filename))

    thr_list = [0.5, 0.5, 0.5]

    # Changed from f-string for Python 3.5 compatibility
    fprint(f, "\n--- Evaluating on Training Set ---")
    get_accuracy_on_large_data_interqp(f, data_sets.train, EVALUATE_QP_THR_LIST, thr_list)

    # Changed from f-string for Python 3.5 compatibility
    fprint(f, "\n--- Evaluating on Validation Set ---")
    get_accuracy_on_large_data_interqp(f, data_sets.validation, EVALUATE_QP_THR_LIST, thr_list)

    # Changed from f-string for Python 3.5 compatibility
    fprint(f, "\n--- Evaluating on Test Set ---")
    get_accuracy_on_large_data_interqp(f, data_sets.test, EVALUATE_QP_THR_LIST, thr_list)

    f.close()

    # Changed from f-string for Python 3.5 compatibility
    print("Evaluation finished. Results saved to {}".format(log_filename))


# placeholders to create the model

x = tf.placeholder("float", [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
y_ = tf.placeholder("float", [None, NUM_LABEL_BYTES])
qp = tf.placeholder("float", [None, NUM_EXT_FEATURES])
isdrop = tf.placeholder("float")
global_step = tf.placeholder("float")

y_flat_64, y_flat_32, y_flat_16, y_conv_64, y_conv_32, y_conv_16, total_loss, loss_list, learning_rate_current, train_step, accuracy_list, opt_vars_all = input_data.nt.net(
    x, y_, qp, isdrop, global_step, LEARNING_RATE_INIT, MOMENTUM_INIT, ITER_TIMES_PER_CHANGE_RATE,
    LEARNING_RATE_DECAY_RATIO)

saver = tf.train.Saver(opt_vars_all, write_version=tf.train.SaverDef.V2)
data_sets = input_data.read_data_sets()

train_accuracy_list = list()
valid_accuracy_list = list()
train_loss_list = list()
valid_loss_list = list()
train_tendency_list = list()
valid_tendency_list = list()
step_list = list()

sess.run(tf.global_variables_initializer())
iter_times_last = 0
if IS_RELOAD == True:
    print("Reloading model from 'Models/model.dat'")
    saver.restore(sess, "/home/ai-iitkgp/PycharmProjects/HEVC-CNN/HEVC-Complexity-Reduction/ETH-CNN_Training_AI/Models/model_20181226_192018_1000000_qp37.dat")
    iter_times_last = reload_loss_and_accuracy()
else:
    # This branch will be skipped since IS_RELOAD is True
    print("Starting from scratch (this should not happen in evaluation mode)")

print('iter_times_last = %d' % iter_times_last)

# The training loop will be skipped because ITER_TIMES is 0
for i in range(ITER_TIMES):
    step = i + iter_times_last + 1

    batch = data_sets.train.next_batch_random(BATCH_SIZE)
    if IS_RESET_LR == True:
        feed_step = i + 1
    else:
        feed_step = step
    learning_rate_value, _ = sess.run([learning_rate_current, train_step],
                                      feed_dict={x: batch[0], qp: batch[2], y_: batch[1], isdrop: 1,
                                                 global_step: feed_step})

# Run the final evaluation
evaluate()

print("\nScript finished.")