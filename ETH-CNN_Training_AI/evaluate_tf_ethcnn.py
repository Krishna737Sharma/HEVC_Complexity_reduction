import os
import numpy as np
import tensorflow as tf
import input_data as input_data
import net_CTU64 as nt
import math
import time

# Configuration
MODEL_PATHS = [
    'Models/model_20181227_060655_1000000_qp22.dat',
    'Models/model_20181225_154616_1000000_qp27.dat',
    'Models/model_20181226_045506_1000000_qp32.dat',
    'Models/model_20181226_192018_1000000_qp37.dat'
]

RESULTS_DIR = 'evaluation_results'
BATCH_SIZE = 64


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))


def fprint(f, str_text):
    print(str_text)
    f.write(str_text + '\r\n')


def get_model_info(model_path):
    """Extract QP info from model filename"""
    if 'qp22' in model_path:
        return 22, "QP22"
    elif 'qp27' in model_path:
        return 27, "QP27"
    elif 'qp32' in model_path:
        return 32, "QP32"
    elif 'qp37' in model_path:
        return 37, "QP37"
    else:
        return None, "Unknown"


def is_sep_64(y_one_sample, thr):
    depth_mean = np.mean(y_one_sample)
    return 1 if depth_mean > thr else 0


def is_sep_32(y_one_sample, thr):
    depth_mean = np.mean(y_one_sample)
    return 1 if depth_mean > thr else 0


def is_sep_16(y_one_sample, thr):
    return 1 if y_one_sample > thr else 0


def get_class_matrices(y_truth, y_predict_64, y_predict_32, y_predict_16, thr_list):
    matrix_64 = [[0, 0], [0, 0]]
    matrix_32 = [[0, 0], [0, 0]]
    matrix_16 = [[0, 0], [0, 0]]

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


def evaluate_validation_set(sess, x, y_, qp, isdrop, y_conv_64, y_conv_32, y_conv_16,
                            accuracy_list, validation_data, thr_list, f):
    """Evaluate complete validation dataset"""
    length = validation_data.num_examples
    PRE_BATCH_SIZE = 1000

    matrix_64_sum = [[0, 0], [0, 0]]
    matrix_32_sum = [[0, 0], [0, 0]]
    matrix_16_sum = [[0, 0], [0, 0]]

    fprint(f, "COMPLETE VALIDATION SET EVALUATION")
    fprint(f, "=" * 40)
    fprint(f, "Total validation samples: {}".format(length))

    for i in range(int(math.ceil(length / float(PRE_BATCH_SIZE)))):
        index_start = PRE_BATCH_SIZE * i
        index_end = min(PRE_BATCH_SIZE * (i + 1), length)

        y_predict_temp_64, y_predict_temp_32, y_predict_temp_16, accuracy_temp = sess.run(
            [y_conv_64, y_conv_32, y_conv_16, accuracy_list],
            feed_dict={x: validation_data.images[index_start:index_end],
                       y_: validation_data.labels[index_start:index_end],
                       qp: validation_data.qps[index_start:index_end],
                       isdrop: 0})

        matrix_64_temp, matrix_32_temp, matrix_16_temp = get_class_matrices(
            validation_data.labels[index_start:index_end], y_predict_temp_64,
            y_predict_temp_32, y_predict_temp_16, thr_list)

        matrix_64_sum = np.add(matrix_64_sum, matrix_64_temp)
        matrix_32_sum = np.add(matrix_32_sum, matrix_32_temp)
        matrix_16_sum = np.add(matrix_16_sum, matrix_16_temp)

    # Calculate final accuracies
    accuracy_64 = (matrix_64_sum[0][0] + matrix_64_sum[1][1]) / float(np.sum(matrix_64_sum)) if np.sum(
        matrix_64_sum) > 0 else 0
    accuracy_32 = (matrix_32_sum[0][0] + matrix_32_sum[1][1]) / float(np.sum(matrix_32_sum)) if np.sum(
        matrix_32_sum) > 0 else 0
    accuracy_16 = (matrix_16_sum[0][0] + matrix_16_sum[1][1]) / float(np.sum(matrix_16_sum)) if np.sum(
        matrix_16_sum) > 0 else 0

    tendency_64 = get_tendency_2x2(matrix_64_sum)
    tendency_32 = get_tendency_2x2(matrix_32_sum)
    tendency_16 = get_tendency_2x2(matrix_16_sum)

    # Print detailed results
    fprint(f, '')
    fprint(f, 'DETAILED RESULTS:')
    fprint(f, '-' * 30)
    fprint(f, 'Confusion Matrix 64x64: {}'.format(matrix_64_sum))
    fprint(f, 'Confusion Matrix 32x32: {}'.format(matrix_32_sum))
    fprint(f, 'Confusion Matrix 16x16: {}'.format(matrix_16_sum))
    fprint(f, '')
    fprint(f, 'Accuracy Results:')
    fprint(f, '  64x64 CU Accuracy: {:.6f}'.format(accuracy_64))
    fprint(f, '  32x32 CU Accuracy: {:.6f}'.format(accuracy_32))
    fprint(f, '  16x16 CU Accuracy: {:.6f}'.format(accuracy_16))
    fprint(f, '')
    fprint(f, 'Tendency Results:')
    fprint(f, '  64x64 CU Tendency: {:.6f}'.format(tendency_64))
    fprint(f, '  32x32 CU Tendency: {:.6f}'.format(tendency_32))
    fprint(f, '  16x16 CU Tendency: {:.6f}'.format(tendency_16))
    fprint(f, '')
    fprint(f, 'Overall Weighted Accuracy: {:.6f}'.format((accuracy_64 + accuracy_32 + accuracy_16) / 3.0))

    return {
        'samples': length,
        'accuracy_64': accuracy_64,
        'accuracy_32': accuracy_32,
        'accuracy_16': accuracy_16,
        'tendency_64': tendency_64,
        'tendency_32': tendency_32,
        'tendency_16': tendency_16,
        'overall_accuracy': (accuracy_64 + accuracy_32 + accuracy_16) / 3.0,
        'matrix_64': matrix_64_sum,
        'matrix_32': matrix_32_sum,
        'matrix_16': matrix_16_sum
    }


def evaluate_dataset_for_all_models(data_switch, dataset_name):
    """Evaluate all models on a specific dataset and save to one file"""

    # Create results directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Find available models
    available_models = []
    for model_path in MODEL_PATHS:
        if os.path.exists(model_path + '.index'):
            available_models.append(model_path)

    if not available_models:
        print("No models found! Please check the model paths.")
        return

    # Create results file for this dataset
    timestamp = get_time_str()
    results_file = '{}/{}_dataset_evaluation_{}.txt'.format(
        RESULTS_DIR, dataset_name.replace(" ", "_").lower(), timestamp)

    f = open(results_file, 'w+')

    # File header
    fprint(f, "HEVC CU PARTITION MODEL EVALUATION RESULTS")
    fprint(f, "=" * 60)
    fprint(f, "Dataset: {}".format(dataset_name))
    fprint(f, "Evaluation Type: Complete Validation Set")
    fprint(f, "Dataset Switch: {}".format(data_switch))
    fprint(f, "Timestamp: {}".format(timestamp))
    fprint(f, "Models Evaluated: {}".format(len(available_models)))
    fprint(f, "=" * 60)
    fprint(f, "")

    all_results = []

    # Evaluate each model
    for model_index, model_path in enumerate(available_models):
        model_qp, model_name = get_model_info(model_path)

        print("\n" + ">" * 60)
        print("EVALUATING {} on {} (Model {}/{})".format(model_name, dataset_name, model_index + 1,
                                                         len(available_models)))
        print(">" * 60)

        # Reset TensorFlow graph
        tf.reset_default_graph()

        # Set dataset switch
        input_data.DATA_SWITCH = data_switch

        # Load dataset
        print("Loading {} dataset...".format(dataset_name))
        data_sets = input_data.read_data_sets()

        # Create TensorFlow session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess = tf.Session(config=config)

        # Create placeholders and model
        x = tf.placeholder("float", [None, input_data.IMAGE_SIZE, input_data.IMAGE_SIZE, input_data.NUM_CHANNELS])
        y_ = tf.placeholder("float", [None, input_data.NUM_LABEL_BYTES])
        qp = tf.placeholder("float", [None, input_data.NUM_EXT_FEATURES])
        isdrop = tf.placeholder("float")
        global_step = tf.placeholder("float")

        # Build network
        y_flat_64, y_flat_32, y_flat_16, y_conv_64, y_conv_32, y_conv_16, total_loss, loss_list, learning_rate_current, train_step, accuracy_list, opt_vars_all = nt.net(
            x, y_, qp, isdrop, global_step, 0.01, 0.9, 250000, 0.3163)

        # Initialize and load model
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(opt_vars_all)

        print("Loading model weights...")
        saver.restore(sess, model_path)

        # Write model header in file
        fprint(f, "MODEL {}: {} (QP={})".format(model_index + 1, model_name, model_qp if model_qp else "Unknown"))
        fprint(f, "=" * 60)
        fprint(f, "Model Path: {}".format(model_path))
        fprint(f, "Training QP: {}".format(model_qp if model_qp else "Unknown"))
        fprint(f, "")

        # Evaluation thresholds
        thr_list = [0.5, 0.5, 0.5]

        # Evaluate validation set
        results = evaluate_validation_set(
            sess, x, y_, qp, isdrop, y_conv_64, y_conv_32, y_conv_16,
            accuracy_list, data_sets.validation, thr_list, f)

        # Add model info to results
        results['model_path'] = model_path
        results['model_qp'] = model_qp
        results['model_name'] = model_name
        results['dataset_name'] = dataset_name

        all_results.append(results)

        sess.close()

        fprint(f, "")
        fprint(f, "=" * 60)
        fprint(f, "")

        print("Model {} evaluation completed.".format(model_name))

    # Write overall summary at the end of file
    fprint(f, "OVERALL PERFORMANCE SUMMARY")
    fprint(f, "=" * 60)
    fprint(f, "Dataset: {}".format(dataset_name))
    fprint(f, "Total Models Evaluated: {}".format(len(all_results)))
    fprint(f, "")

    fprint(f, "SUMMARY TABLE:")
    fprint(f, "-" * 80)
    fprint(f, "Model\t\tQP\tSamples\t\tAcc_64\t\tAcc_32\t\tAcc_16\t\tOverall")
    fprint(f, "-" * 80)

    for result in all_results:
        fprint(f, "{}\t\t{}\t{}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(
            result['model_name'],
            result['model_qp'] if result['model_qp'] else "?",
            result['samples'],
            result['accuracy_64'],
            result['accuracy_32'],
            result['accuracy_16'],
            result['overall_accuracy']
        ))

    fprint(f, "-" * 80)
    fprint(f, "")

    # Best performing model analysis
    best_model = max(all_results, key=lambda x: x['overall_accuracy'])
    worst_model = min(all_results, key=lambda x: x['overall_accuracy'])

    fprint(f, "PERFORMANCE ANALYSIS:")
    fprint(f, "-" * 40)
    fprint(f, "Best Performing Model:")
    fprint(f, "  Model: {} (QP={})".format(best_model['model_name'], best_model['model_qp']))
    fprint(f, "  Overall Accuracy: {:.6f}".format(best_model['overall_accuracy']))
    fprint(f, "  64x64 Accuracy: {:.6f}".format(best_model['accuracy_64']))
    fprint(f, "  32x32 Accuracy: {:.6f}".format(best_model['accuracy_32']))
    fprint(f, "  16x16 Accuracy: {:.6f}".format(best_model['accuracy_16']))
    fprint(f, "")
    fprint(f, "Worst Performing Model:")
    fprint(f, "  Model: {} (QP={})".format(worst_model['model_name'], worst_model['model_qp']))
    fprint(f, "  Overall Accuracy: {:.6f}".format(worst_model['overall_accuracy']))
    fprint(f, "  64x64 Accuracy: {:.6f}".format(worst_model['accuracy_64']))
    fprint(f, "  32x32 Accuracy: {:.6f}".format(worst_model['accuracy_32']))
    fprint(f, "  16x16 Accuracy: {:.6f}".format(worst_model['accuracy_16']))
    fprint(f, "")
    fprint(f, "Performance Range:")
    fprint(f,
           "  Accuracy Range: {:.6f} - {:.6f}".format(worst_model['overall_accuracy'], best_model['overall_accuracy']))
    fprint(f, "  Accuracy Difference: {:.6f}".format(best_model['overall_accuracy'] - worst_model['overall_accuracy']))
    fprint(f, "")

    # Average performance across all models
    avg_accuracy_64 = sum([r['accuracy_64'] for r in all_results]) / len(all_results)
    avg_accuracy_32 = sum([r['accuracy_32'] for r in all_results]) / len(all_results)
    avg_accuracy_16 = sum([r['accuracy_16'] for r in all_results]) / len(all_results)
    avg_overall = sum([r['overall_accuracy'] for r in all_results]) / len(all_results)

    fprint(f, "AVERAGE PERFORMANCE ACROSS ALL MODELS:")
    fprint(f, "-" * 40)
    fprint(f, "Average 64x64 Accuracy: {:.6f}".format(avg_accuracy_64))
    fprint(f, "Average 32x32 Accuracy: {:.6f}".format(avg_accuracy_32))
    fprint(f, "Average 16x16 Accuracy: {:.6f}".format(avg_accuracy_16))
    fprint(f, "Average Overall Accuracy: {:.6f}".format(avg_overall))
    fprint(f, "")
    fprint(f, "Dataset Statistics:")
    fprint(f, "  Total Validation Samples: {:,}".format(all_results[0]['samples']))
    fprint(f, "  Evaluation Method: Complete Validation Set")
    fprint(f, "  QP Coverage: All available QP ranges in dataset")
    fprint(f, "")
    fprint(f, "=" * 60)
    fprint(f, "Evaluation completed at: {}".format(time.strftime('%Y-%m-%d %H:%M:%S')))

    f.close()

    print("\nEvaluation for {} completed!".format(dataset_name))
    print("Results saved to: {}".format(results_file))

    return all_results, results_file


def main():
    """
    Main evaluation function - evaluates all models on both datasets separately
    """
    print("HEVC CU PARTITION MODEL EVALUATION - COMPLETE VALIDATION SETS")
    print("=" * 80)

    # Evaluate Full Dataset
    print("\nEVALUATING FULL DATASET")
    print("=" * 40)
    full_results, full_file = evaluate_dataset_for_all_models(0, "Full Dataset")

    # Evaluate Demo Dataset
    print("\nEVALUATING DEMO DATASET")
    print("=" * 40)
    demo_results, demo_file = evaluate_dataset_for_all_models(1, "Demo Dataset")

    # Final summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("✓ Full Dataset evaluation completed")
    print("  - Results file: {}".format(full_file))
    print("  - Models evaluated: {}".format(len(full_results) if full_results else 0))
    print("✓ Demo Dataset evaluation completed")
    print("  - Results file: {}".format(demo_file))
    print("  - Models evaluated: {}".format(len(demo_results) if demo_results else 0))
    print("\n✓ Complete validation sets were used for evaluation")
    print("✓ Separate text files created for each dataset")
    print("✓ Performance summaries added to end of each file")
    print("✓ All available models evaluated on both datasets")


if __name__ == "__main__":
    main()