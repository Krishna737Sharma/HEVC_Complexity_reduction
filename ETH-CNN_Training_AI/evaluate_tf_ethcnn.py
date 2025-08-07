#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for HEVC-CNN model on validation dataset only
Python 3.5 compatible
"""

import os
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import input_data as input_data

# Set matplotlib backend for headless environments
matplotlib.use('Agg')

# Configuration
DEVICE_MODE = 1  # 0:CPU, 1:GPU (limited), 2:GPU (unlimited)
MODEL_PATH = 'Models/model.dat'  # Default path to your trained model
RESULTS_DIR = 'evaluation_results'


def find_available_models():
    """Find all available trained model files"""
    models_dir = 'Models'
    available_models = []

    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.meta'):
                model_path = os.path.join(models_dir, file[:-5])  # Remove .meta extension
                available_models.append(model_path)

    return available_models


def select_model():
    """Select which model to use for evaluation"""
    # First try the default model
    if os.path.exists(MODEL_PATH + '.meta'):
        return MODEL_PATH

    # If default doesn't exist, find available models
    available_models = find_available_models()

    if not available_models:
        print("No trained model files found in Models directory!")
        print("Please ensure you have trained model files (.meta, .index, .data files)")
        return None

    print("\nAvailable trained models:")
    for i, model in enumerate(available_models):
        print("  {}: {}".format(i + 1, model))

    if len(available_models) == 1:
        selected_model = available_models[0]
        print("Using model: {}".format(selected_model))
        return selected_model

    # Let user choose if multiple models available
    while True:
        try:
            choice = input("\nSelect model number (1-{}): ".format(len(available_models)))
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_models):
                selected_model = available_models[choice_idx]
                print("Using model: {}".format(selected_model))
                return selected_model
            else:
                print("Invalid choice. Please enter a number between 1 and {}".format(len(available_models)))
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or cancelled.")
            return None


# Create results directory if it doesn't exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Configure TensorFlow session
if DEVICE_MODE == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess = tf.Session()
elif DEVICE_MODE == 1:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    sess = tf.Session(config=config)
elif DEVICE_MODE == 2:
    sess = tf.Session()

# Import model parameters
IMAGE_SIZE = input_data.IMAGE_SIZE
NUM_CHANNELS = input_data.NUM_CHANNELS
NUM_EXT_FEATURES = input_data.NUM_EXT_FEATURES
NUM_LABEL_BYTES = input_data.NUM_LABEL_BYTES
EVALUATE_QP_THR_LIST = input_data.EVALUATE_QP_THR_LIST
MODEL_NAME = input_data.MODEL_NAME


def get_time_str():
    """Get current timestamp as string"""
    return time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))


def fprint(f, text):
    """Print to both console and file"""
    print(text)
    if f is not None:
        f.write(text + '\n')
        f.flush()


def is_sep_64(y_one_sample, thr):
    """Check if 64x64 CU should be separated"""
    depth_mean = np.mean(y_one_sample)
    return 1 if depth_mean > thr else 0


def is_sep_32(y_one_sample, thr):
    """Check if 32x32 CU should be separated"""
    depth_mean = np.mean(y_one_sample)
    return 1 if depth_mean > thr else 0


def is_sep_16(y_one_sample, thr):
    """Check if 16x16 CU should be separated"""
    return 1 if y_one_sample > thr else 0


def get_class_matrices(y_truth, y_predict_64, y_predict_32, y_predict_16, thr_list):
    """
    Calculate classification matrices for 64x64, 32x32, and 16x16 CUs
    Returns confusion matrices for each CU size
    """
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


def calculate_metrics(matrix):
    """Calculate precision, recall, F1-score from confusion matrix"""
    tn, fp, fn, tp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score


def get_tendency_2x2(matrix_2x2):
    """Calculate tendency metric from 2x2 confusion matrix"""
    if matrix_2x2[0][1] == 0 and matrix_2x2[1][0] == 0:
        return 0
    elif matrix_2x2[0][1] == 0 or matrix_2x2[1][1] == 0:
        return -100
    elif matrix_2x2[1][0] == 0 or matrix_2x2[0][0] == 0:
        return 100
    else:
        import math
        return -math.log10((matrix_2x2[0][0] / matrix_2x2[0][1]) / (matrix_2x2[1][1] / matrix_2x2[1][0]))


def evaluate_dataset(f, images, qps, labels, thr_list, dataset_name):
    """Evaluate model performance on a dataset"""
    fprint(f, "\n" + "=" * 60)
    fprint(f, "Evaluating {} Dataset".format(dataset_name))
    fprint(f, "=" * 60)

    length = images.shape[0]
    batch_size = 1000  # Process in batches to avoid memory issues

    y_predict_16_all = []
    y_predict_32_all = []
    y_predict_64_all = []

    total_loss = 0
    total_accuracy = np.zeros(3)

    fprint(f, "Processing {} samples in batches of {}...".format(length, batch_size))

    # Process data in batches
    for i in range(0, length, batch_size):
        end_idx = min(i + batch_size, length)
        batch_images = images[i:end_idx]
        batch_qps = qps[i:end_idx]
        batch_labels = labels[i:end_idx]

        # Run inference
        y_pred_64, y_pred_32, y_pred_16, accuracy_batch, loss_batch = sess.run(
            [y_conv_64, y_conv_32, y_conv_16, accuracy_list, loss_list],
            feed_dict={
                x: batch_images,
                y_: batch_labels,
                qp: batch_qps,
                isdrop: 0  # No dropout during evaluation
            }
        )

        # Accumulate predictions
        if len(y_predict_16_all) == 0:
            y_predict_16_all = y_pred_16
            y_predict_32_all = y_pred_32
            y_predict_64_all = y_pred_64
        else:
            y_predict_16_all = np.vstack((y_predict_16_all, y_pred_16))
            y_predict_32_all = np.vstack((y_predict_32_all, y_pred_32))
            y_predict_64_all = np.vstack((y_predict_64_all, y_pred_64))

        # Accumulate metrics
        batch_size_actual = end_idx - i
        total_accuracy += accuracy_batch * batch_size_actual
        total_loss += np.sum(loss_batch) * batch_size_actual

        if (i // batch_size + 1) % 10 == 0:
            fprint(f, "  Processed batch {}/{}".format(i // batch_size + 1, (length + batch_size - 1) // batch_size))

    # Calculate overall metrics
    avg_accuracy = total_accuracy / length
    avg_loss = total_loss / length

    # Get confusion matrices
    matrix_64, matrix_32, matrix_16 = get_class_matrices(
        labels, y_predict_64_all, y_predict_32_all, y_predict_16_all, thr_list
    )

    # Calculate detailed metrics for each CU size
    metrics_64 = calculate_metrics(matrix_64)
    metrics_32 = calculate_metrics(matrix_32)
    metrics_16 = calculate_metrics(matrix_16)

    # Calculate tendencies
    tendency_64 = get_tendency_2x2(matrix_64)
    tendency_32 = get_tendency_2x2(matrix_32)
    tendency_16 = get_tendency_2x2(matrix_16)

    # Print results
    fprint(f, "\nOverall Results:")
    fprint(f, "Average Loss: {:.6f}".format(avg_loss))
    fprint(f, "Average Accuracy: 64x64={:.4f}, 32x32={:.4f}, 16x16={:.4f}".format(
        avg_accuracy[0], avg_accuracy[1], avg_accuracy[2]))

    fprint(f, "\nConfusion Matrices:")
    fprint(f, "64x64 CU: {}".format(matrix_64))
    fprint(f, "32x32 CU: {}".format(matrix_32))
    fprint(f, "16x16 CU: {}".format(matrix_16))

    fprint(f, "\nDetailed Metrics:")
    cu_sizes = ['64x64', '32x32', '16x16']
    all_metrics = [metrics_64, metrics_32, metrics_16]
    tendencies = [tendency_64, tendency_32, tendency_16]

    for i, (cu_size, metrics, tendency) in enumerate(zip(cu_sizes, all_metrics, tendencies)):
        acc, prec, rec, f1 = metrics
        fprint(f, "{} CU:".format(cu_size))
        fprint(f, "  Accuracy: {:.4f}".format(acc))
        fprint(f, "  Precision: {:.4f}".format(prec))
        fprint(f, "  Recall: {:.4f}".format(rec))
        fprint(f, "  F1-score: {:.4f}".format(f1))
        fprint(f, "  Tendency: {:.4f}".format(tendency))

    return {
        'avg_accuracy': avg_accuracy,
        'avg_loss': avg_loss,
        'confusion_matrices': [matrix_64, matrix_32, matrix_16],
        'detailed_metrics': all_metrics,
        'tendencies': tendencies
    }


def evaluate_by_qp_range(f, dataset, dataset_name, thr_list):
    """Evaluate performance by QP ranges"""
    fprint(f, "\n" + "=" * 60)
    fprint(f, "QP Range Analysis for {}".format(dataset_name))
    fprint(f, "=" * 60)

    range_stat = input_data.RangingStatistics(EVALUATE_QP_THR_LIST, 'scalar')
    count_list, stat_index = range_stat.feed_data_list(dataset.qps, is_select=True)
    segment_names = range_stat.get_segment_names('QP')

    qp_results = {}

    for i in range(len(EVALUATE_QP_THR_LIST) + 1):
        if count_list[i] > 0:
            fprint(f, "\n" + "-" * 30)
            fprint(f, segment_names[i])
            fprint(f, "Sample count: {}".format(count_list[i]))
            fprint(f, "-" * 30)

            # Get subset of data for this QP range
            indices = stat_index[i]
            subset_images = dataset.images[indices]
            subset_qps = dataset.qps[indices]
            subset_labels = dataset.labels[indices]

            # Evaluate this subset
            results = evaluate_dataset(f, subset_images, subset_qps, subset_labels,
                                       thr_list, "{}_{}".format(dataset_name, segment_names[i]))
            qp_results[segment_names[i]] = results

    return qp_results


def plot_results(results, save_path):
    """Create visualization of results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy comparison
    cu_sizes = ['64x64', '32x32', '16x16']
    accuracies = results['avg_accuracy']

    axes[0, 0].bar(cu_sizes, accuracies)
    axes[0, 0].set_title('Accuracy by CU Size')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)

    # Confusion matrices visualization
    matrices = results['confusion_matrices']
    for i, (cu_size, matrix) in enumerate(zip(cu_sizes, matrices)):
        ax = axes[0, 1] if i == 0 else (axes[1, 0] if i == 1 else axes[1, 1])
        im = ax.imshow(matrix, cmap='Blues')
        ax.set_title('Confusion Matrix - {}'.format(cu_size))

        # Add text annotations
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(matrix[row][col]),
                        ha='center', va='center', fontsize=12)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Split', 'Split'])
        ax.set_yticklabels(['No Split', 'Split'])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    """Main evaluation function"""
    global x, y_, qp, isdrop, y_conv_64, y_conv_32, y_conv_16, accuracy_list, loss_list

    print("Starting HEVC-CNN Model Evaluation (Validation Set Only)")
    print("Model: {}".format(MODEL_NAME))

    # Select available model
    selected_model_path = select_model()
    if selected_model_path is None:
        return

    print("Model path: {}".format(selected_model_path))

    # Create placeholders and network
    x = tf.placeholder("float", [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    y_ = tf.placeholder("float", [None, NUM_LABEL_BYTES])
    qp = tf.placeholder("float", [None, NUM_EXT_FEATURES])
    isdrop = tf.placeholder("float")
    global_step = tf.placeholder("float")

    # Build network
    (y_flat_64, y_flat_32, y_flat_16, y_conv_64, y_conv_32, y_conv_16,
     total_loss, loss_list, learning_rate_current, train_step,
     accuracy_list, opt_vars_all) = input_data.nt.net(
        x, y_, qp, isdrop, global_step, 0.01, 0.9, 250000, 0.3163
    )

    # Create saver and load model
    saver = tf.train.Saver(opt_vars_all, write_version=tf.train.SaverDef.V2)

    # Initialize session
    sess.run(tf.global_variables_initializer())

    try:
        # Load trained model
        print("Loading trained model...")
        saver.restore(sess, selected_model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print("Error loading model: {}".format(e))
        return

    # Load datasets (only validation set will be used)
    print("Loading validation dataset...")
    data_sets = input_data.read_data_sets()
    print("Validation set size: {}".format(data_sets.validation.num_examples))

    # Open results file
    timestamp = get_time_str()
    model_name_clean = os.path.basename(selected_model_path).replace('/', '_')
    results_file = os.path.join(RESULTS_DIR, 'validation_evaluation_{}_{}.txt'.format(model_name_clean, timestamp))

    with open(results_file, 'w') as f:
        fprint(f, "HEVC-CNN Model Validation Evaluation Results")
        fprint(f, "Generated on: {}".format(timestamp))
        fprint(f, "Model: {}".format(MODEL_NAME))
        fprint(f, "Model file: {}".format(selected_model_path))
        fprint(f, "Dataset: Validation Set Only")
        fprint(f, "=" * 60)

        # Evaluation thresholds
        thr_list = [0.5, 0.5, 0.5]
        fprint(f, "Evaluation thresholds: {}".format(thr_list))

        # Evaluate validation set only
        print("\nEvaluating validation dataset...")
        val_results = evaluate_dataset(f, data_sets.validation.images,
                                       data_sets.validation.qps,
                                       data_sets.validation.labels,
                                       thr_list, "Validation")

        # QP range analysis for validation set
        print("\nPerforming QP range analysis on validation set...")
        val_qp_results = evaluate_by_qp_range(f, data_sets.validation, "Validation", thr_list)

        # Summary
        fprint(f, "\n" + "=" * 60)
        fprint(f, "VALIDATION SET SUMMARY")
        fprint(f, "=" * 60)
        fprint(f, "Overall Accuracy: {}".format(val_results['avg_accuracy']))
        fprint(f, "Average Loss: {:.6f}".format(val_results['avg_loss']))
        fprint(f, "64x64 CU Accuracy: {:.4f}".format(val_results['avg_accuracy'][0]))
        fprint(f, "32x32 CU Accuracy: {:.4f}".format(val_results['avg_accuracy'][1]))
        fprint(f, "16x16 CU Accuracy: {:.4f}".format(val_results['avg_accuracy'][2]))

        # Print F1 scores summary
        f1_64 = val_results['detailed_metrics'][0][3]
        f1_32 = val_results['detailed_metrics'][1][3]
        f1_16 = val_results['detailed_metrics'][2][3]
        fprint(f, "64x64 CU F1-Score: {:.4f}".format(f1_64))
        fprint(f, "32x32 CU F1-Score: {:.4f}".format(f1_32))
        fprint(f, "16x16 CU F1-Score: {:.4f}".format(f1_16))
        fprint(f, "Average F1-Score: {:.4f}".format((f1_64 + f1_32 + f1_16) / 3))

    # Create visualization for validation results
    print("Creating result visualizations...")
    plot_results(val_results,
                 os.path.join(RESULTS_DIR, 'validation_results_{}_{}.png'.format(model_name_clean, timestamp)))

    print("\nValidation evaluation completed!")
    print("Results saved to: {}".format(results_file))
    print("Visualizations saved to: {}/".format(RESULTS_DIR))

    sess.close()


if __name__ == "__main__":
    main()
