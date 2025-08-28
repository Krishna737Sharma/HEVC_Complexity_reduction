import sys
import os

sys.path.append('/home/ai-iitkgp/PycharmProjects/HEVC-CNN/HEVC-Complexity-Reduction')  # Update this path
import input_data as input_data
import net_CTU64 as nt
import tensorflow as tf
import numpy as np

# Configuration
MODEL_PATH = '/home/ai-iitkgp/PycharmProjects/HEVC-CNN/HEVC-Complexity-Reduction/ETH-CNN_Training_AI/Models/model_20181226_045506_1000000_qp32.dat'  # Update this path


def run_cnn_inference():
    # Load ground truth data
    gt_data = np.load('ground_truth.npy', allow_pickle=True).item()

    # Setup TensorFlow placeholders
    x = tf.placeholder("float", [None, 64, 64, 1])
    y_ = tf.placeholder("float", [None, 16])
    qp = tf.placeholder("float", [None, 1])
    isdrop = tf.placeholder("float")
    global_step = tf.placeholder("float")

    # Build network
    _, _, _, y_conv_64, y_conv_32, y_conv_16, _, _, _, _, _, opt_vars = nt.net(
        x, y_, qp, isdrop, global_step, 0.01, 0.9, 1, 1
    )

    # Create session and load model
    sess = tf.Session()
    saver = tf.train.Saver(opt_vars)
    saver.restore(sess, MODEL_PATH)

    # Prepare input
    image_input = gt_data['image'].reshape(1, 64, 64, 1)
    qp_input = gt_data['qp'].reshape(1, 1)

    # Run inference
    pred_64, pred_32, pred_16 = sess.run(
        [y_conv_64, y_conv_32, y_conv_16],
        feed_dict={x: image_input, qp: qp_input, isdrop: 0, y_: np.zeros((1, 16))}
    )

    # Save results
    np.save('cnn_predictions.npy', {
        'pred_64': pred_64[0],
        'pred_32': pred_32[0],
        'pred_16': pred_16[0]
    })

    sess.close()
    print("CNN predictions saved to cnn_predictions.npy")


if __name__ == "__main__":
    run_cnn_inference()
