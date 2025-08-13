# HEVC_Complexity_reduction
# CPH-Intra Dataset Preparation for Intra Prediction Models

This repository contains the complete data preparation pipeline for training intra prediction models using the CPH-Intra dataset.

## Overview

The CPH-Intra dataset is a comprehensive dataset for training and evaluating intra prediction models in video coding. This guide provides step-by-step instructions for downloading, extracting, and preparing the dataset for machine learning applications.

## Dataset Structure

The CPH-Intra dataset consists of two main components:

### 1. Raw Video Files (YUV Format)
- **IntraTest**: Test sequences in different resolutions
- **IntraTrain**: Training sequences in different resolutions  
- **IntraValid**: Validation sequences in different resolutions

Available resolutions:
- 768×512
- 1536×1024
- 2880×1920
- 4928×3264

### 2. Metadata Files
- **CUDepth.dat**: Coding Unit depth information
- **Index.dat**: Indexing information for CU partitions

## Installation & Setup

### Method 1: Download from Original Source

#### Step 1: Download the Dataset
Download the following files:
- `info.rar`
- `yuv_all.part1.rar` through `yuv_all.part7.rar`

#### Step 2: Extract Multi-part RAR Archive

**On Windows:**
1. Install WinRAR or 7-Zip
2. Place all `.partXX.rar` files in the same directory
3. Right-click on `yuv_all.part1.rar` → Select "Extract Here"
4. The tool will automatically combine all parts

**On Linux/Mac:**
```bash
# Install unrar
sudo apt install unrar  # Ubuntu/Debian
# or
brew install unrar     # macOS

# Extract starting from part1
unrar x yuv_all.part1.rar
```

#### Step 3: Extract Info Files
Extract `info.rar` separately to get the metadata files.

### Method 2: Download via Kaggle Hub (Recommended)

```python
import kagglehub

# Download the raw dataset
path = kagglehub.dataset_download("krishnasharma737/cph-intra-dataset")
print("Path to dataset files:", path)
```

### Method 3: Download Pre-processed Dataset

```python
import kagglehub

# Download pre-processed training datasets
path = kagglehub.dataset_download("krishnasharma737/cph-training-datasets")
print("Path to dataset files:", path)
```

## Data Preparation Pipeline

### File Organization

After extraction, organize files into two separate folders:

```
dataset/
├── yuv_files/
│   ├── IntraTest_768x512.yuv
│   ├── IntraTest_1536x1024.yuv
│   ├── IntraTest_2880x1920.yuv
│   ├── IntraTest_4928x3264.yuv
│   ├── IntraTrain_768x512.yuv
│   ├── IntraTrain_1536x1024.yuv
│   ├── IntraTrain_2880x1920.yuv
│   ├── IntraTrain_4928x3264.yuv
│   ├── IntraValid_768x512.yuv
│   ├── IntraValid_1536x1024.yuv
│   ├── IntraValid_2880x1920.yuv
│   └── IntraValid_4928x3264.yuv
└── info_files/
    ├── Info_20170810_191501_AI_IntraTest_768x512_qp22_nf50_CUDepth.dat
    ├── Info_20170810_191501_AI_IntraTest_768x512_qp22_nf50_Index.dat
    └── ... (73+ more info files)
```

### Processing the Dataset

1. **Configure Paths**: Update the paths in `extract_data_ai.py` to point to your YUV and DAT file directories

2. **Run Data Extraction**:
```bash
python extract_data_ai.py
```

3. **Output Files**: The script generates 6 processed dataset files:

**Original Format:**
- `AI_Train_2446725.dat` - Training dataset (2,446,725 samples)
- `AI_Valid_143925.dat` - Validation dataset (143,925 samples)
- `AI_Test_287850.dat` - Test dataset (287,850 samples)

**Shuffled Format:**
- `AI_Train_2446725.dat_shuffled` - Shuffled training dataset
- `AI_Valid_143925.dat_shuffled` - Shuffled validation dataset
- `AI_Test_287850.dat_shuffled` - Shuffled test dataset

## Dataset Statistics

| Split | Samples | Description |
|-------|---------|-------------|
| Train | 2,446,725 | Training samples for model learning |
| Validation | 143,925 | Validation samples for hyperparameter tuning |
| Test | 287,850 | Test samples for final evaluation |

## Quality Parameters (QP) Coverage

The dataset includes sequences encoded with different Quantization Parameters:
- QP 22 (High quality)
- QP 27 (Medium-high quality)
- QP 32 (Medium quality)
- QP 37 (Lower quality)

## Usage

After preparing the dataset, you can load the processed files for training your intra prediction models:

```python
import numpy as np

# Load training data
train_data = np.fromfile('AI_Train_2446725.dat_shuffled', dtype=np.uint8)

# Load validation data
valid_data = np.fromfile('AI_Valid_143925.dat_shuffled', dtype=np.uint8)

# Load test data
test_data = np.fromfile('AI_Test_287850.dat_shuffled', dtype=np.uint8)
```

## Requirements

- Python 3.x
- NumPy
- Kaggle Hub (for direct download)
- WinRAR/7-Zip/unrar (for manual extraction)

---

**Note**: The dataset files are large (several GB). Ensure you have sufficient storage space and a stable internet connection for downloading.
