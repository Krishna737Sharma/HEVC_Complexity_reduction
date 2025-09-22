import os
import numpy as np
import tensorflow as tf
import input_data as input_data  # your input_data.py
import net_CTU64 as nt          # your net_CTU64.py

# Configure device mode: 0=CPU, 1=GPU limited, 2=GPU unlimited
DEVICE_MODE = 1
if DEVICE_MODE == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess = tf.Session()
elif DEVICE_MODE == 1:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    sess = tf.Session(config=config)
else:
    sess = tf.Session()

# Set necessary constants from input_data and net_CTU64
IMAGE_SIZE = input_data.IMAGE_SIZE
NUM_CHANNELS = input_data.NUM_CHANNELS
NUM_EXT_FEATURES = input_data.NUM_EXT_FEATURES
NUM_LABEL_BYTES = input_data.NUM_LABEL_BYTES

# Set DATA_SWITCH globally before loading to select which datasets to load (0=full)
input_data.DATA_SWITCH = 0

# Define TensorFlow placeholders
x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name='x')
y_ = tf.placeholder(tf.float32, [None, NUM_LABEL_BYTES], name='y_')
qp = tf.placeholder(tf.float32, [None, NUM_EXT_FEATURES], name='qp')
isdrop = tf.placeholder(tf.float32, name='isdrop')
global_step = tf.placeholder(tf.float32, name='global_step')

# Build model graph from net_CTU64.py
(y_flat_64, y_flat_32, y_flat_16,
 y_conv_64, y_conv_32, y_conv_16,
 total_loss, loss_list, learning_rate_current,
 train_step, accuracy_list, opt_vars_all) = nt.net(
    x, y_, qp, isdrop, global_step,
    learning_rate_init=0.01,
    momentum=0.9,
    decay_step=250000,
    decay_rate=0.3163)

# TensorFlow Saver for restoring trained model variables
saver = tf.train.Saver(opt_vars_all)

# Load datasets
data_sets = input_data.read_data_sets()

def evaluate_dataset(name, dataset):
    batch_size = 5000
    num_samples = dataset.num_examples
    all_acc = []

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        images = dataset.images[start:end]
        labels = dataset.labels[start:end]
        qps = dataset.qps[start:end]

        feed_dict = {x: images, y_: labels, qp: qps, isdrop: 0, global_step: 0}
        accuracy_val = sess.run(accuracy_list, feed_dict=feed_dict)
        all_acc.append(accuracy_val)

        print("{} samples {}-{} accuracy (64x64, 32x32, 16x16): {:.4f}, {:.4f}, {:.4f}".format(
            name, start, end-1, accuracy_val[0], accuracy_val[1], accuracy_val[2]))

    avg_acc = np.mean(all_acc, axis=0)
    print("\n{} set average accuracy - 64x64: {:.4f}, 32x32: {:.4f}, 16x16: {:.4f}\n".format(
        name, avg_acc[0], avg_acc[1], avg_acc[2]))
    return avg_acc

with sess.as_default():
    # Set this path to your actual TensorFlow checkpoint prefix (exclude extensions like .index)
    checkpoint_path = 'Models/model_20181226_045506_1000000_qp32.dat'

    # Check if checkpoint exists (TensorFlow checkpoints typically have .index and .data files)
    if not (os.path.exists(checkpoint_path + '.index') or os.path.exists(checkpoint_path)):
        raise FileNotFoundError("Checkpoint prefix '{}' file(s) not found.".format(checkpoint_path))

    saver.restore(sess, checkpoint_path)
    print("Model restored from '{}'".format(checkpoint_path))

    # Evaluate train, validation, and test datasets
    evaluate_dataset("Train", data_sets.train)
    evaluate_dataset("Validation", data_sets.validation)
    evaluate_dataset("Test", data_sets.test)
