# ETH-CNN: Hierarchical CTU Split Prediction for Video Coding

## Overview

This project implements an ETH-CNN (Enhanced Three-branch Hierarchical Convolutional Neural Network) for predicting CTU (Coding Tree Unit) split decisions in video compression. The model uses a multi-scale hierarchical approach to predict split decisions at different levels (64x64, 32x32, and 16x16) simultaneously.

## Architecture

The ETH-CNN model features:
- **Three-branch architecture** for multi-scale feature extraction
- **Hierarchical CTU processing** at 64x64, 32x32, and 16x16 levels  
- **Mean normalization** at different block levels for each branch
- **Multi-scale convolutional layers** with LeakyReLU activation
- **Fully connected layers** with QP (Quantization Parameter) integration
- **Sigmoid output** for binary split prediction

### Model Structure

```
Input: CTU (64x64) + QP
├── Branch 1: 64x64 → Mean norm → Downsample to 16x16 → Conv layers → FC → 1 output
├── Branch 2: 64x64 → Mean norm → Downsample to 32x32 → Conv layers → FC → 4 outputs  
└── Branch 3: 64x64 → Mean norm → Keep 64x64 → Conv layers → FC → 16 outputs
```

## Files Description

### Core Files
- `ETH_CNN.py`: Main training script with complete model implementation
- `Model_Evaluation_ETH_CNN.py`: Model evaluation and testing script
- `file_reader_3.py`: Utility for reading binary data files

### Key Components

#### Data Processing
- **StreamingDataset**: Custom PyTorch Dataset for efficient data loading
- **Hierarchical label processing**: Converts labels to multi-scale format
- **Data normalization**: CTU values normalized to [0,1], QP normalized to [0,1]

#### Model Architecture
- **ETH_CNN class**: Main model with three-branch design
- **Convolutional layers**: 3 conv layers per branch with increasing channels (16→24→32)
- **Fully connected layers**: QP-integrated FC layers with dropout (0.5, 0.2)

#### Training Features
- **Custom loss function**: Multi-level binary cross-entropy with mask handling
- **Accuracy calculation**: Level-wise and combined accuracy metrics
- **Checkpoint system**: Automatic model saving and resuming
- **Wandb integration**: Experiment tracking and visualization
- **Dynamic dataset loading**: Periodic dataset shuffling for better generalization

## Requirements

```python
torch >= 1.9.0
torchsummary
numpy
pandas
matplotlib
scikit-learn
wandb
onnxruntime
```

## Dataset Format

The model expects binary data files with the following structure per sample:
- **CTU data**: 4096 bytes (64x64 image)
- **Metadata**: 64 bytes
- **Labels**: 832 bytes (52 QP levels × 16 bytes each)

## Usage

### Training

```python
python ETH_CNN.py
```

Key training parameters:
- **Batch size**: 64
- **Learning rate**: 0.01 (SGD with momentum 0.9)
- **Scheduler**: Exponential decay (γ=0.3163)
- **Epochs**: 100,000
- **Early stopping**: Patience-based with dataset refreshing

### Evaluation

```python
python Model_Evaluation_ETH_CNN.py
```

## Configuration

### Data Paths
```python
train_file_path = "/path/to/AI_Train_1668975.dat_shuffled"
validation_file_path = "/path/to/AI_Valid_98175.dat_shuffled"  
test_file_path = "/path/to/AI_Test_196350.dat_shuffled"
```

### Model Parameters
```python
IMAGE_SIZE = 64
NUM_CHANNELS = 1
BATCH_SIZE = 64
SELECT_QP_LIST = [22, 27, 32, 37]  # QP values used during training
```

## Training Strategy

1. **Multi-scale normalization**: Different mean removal strategies for each branch
2. **Hierarchical loss**: Combined loss from all three prediction levels
3. **Dynamic data loading**: Periodic dataset shuffling to prevent overfitting
4. **Checkpointing**: Regular model state saving with best model tracking
5. **Early stopping**: Patience-based stopping with model restoration

## Model Output

The model produces three outputs:
- **Branch 1**: 1 value (64x64 level split decision)
- **Branch 2**: 4 values (32x32 level split decisions) 
- **Branch 3**: 16 values (16x16 level split decisions)

Each output represents the probability of splitting at the corresponding block level.

## Loss Function

Custom hierarchical loss combining:
- **Level 64 loss**: Standard binary cross-entropy
- **Level 32 loss**: Masked binary cross-entropy (only valid blocks)
- **Level 16 loss**: Masked binary cross-entropy (only valid blocks)

## Accuracy Metrics

- **Combined accuracy**: Average of all three levels
- **Level-wise accuracy**: Individual accuracy for each hierarchical level
- **Masked accuracy**: Only considers valid blocks for levels 32 and 16

## Checkpoints

The training process saves:
- **Model state**: Complete model parameters
- **Optimizer state**: Training state for resuming
- **Training metrics**: Loss and accuracy histories
- **Best model**: Automatically saved when validation improves

## Visualization

Training generates plots for:
- Combined accuracy (train vs validation)
- Level-wise training accuracy  
- Level-wise training loss
- Level-wise validation accuracy
- Level-wise validation loss

## Hardware Requirements

- **GPU**: CUDA-capable GPU recommended
- **Memory**: Minimum 8GB RAM
- **Storage**: Sufficient space for large binary dataset files

## Notes

- The model uses QP integration at fully connected layers for rate-distortion optimization
- Hierarchical pooling operations convert single labels to multi-scale targets
- The three-branch design captures features at different spatial resolutions
- Early stopping with dataset refreshing helps prevent overfitting on large datasets

## Citation

If you use this implementation, please cite the original ETH-CNN paper and methodology.
