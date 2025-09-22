# HEVC Deep Learning Dataset Generation, Training and Inference (Intra Prediction Mode)

## Overview

This project generates datasets and trains deep neural networks to accelerate HEVC encoding by predicting optimal Coding Unit (CU) partitions.

The main purpose is to generate a dataset where each sample consists of:
- An input feature: A 64x64 pixel patch from the luminance (Y) channel of a video frame
- A set of labels: The corresponding Coding Unit (CU) depth partitions for that same 64x64 patch, as determined by the HEVC encoder for different Quantization Parameters (QPs)

This dataset is used to train a neural network to predict the optimal CU partition for a given 64x64 block, aiming to speed up the HEVC encoding process.

## Dataset Structure

### Single Data Sample (4992 bytes)

Each training sample is structured as a 4992-byte block:

```
buf_sample = (np.ones((4096 + 64 + 16 * 52,)) * 255).astype(np.uint8)
```

This equals 4096 + 64 + 832 = 4992 bytes, structured as follows:

| Byte Range | Size (bytes) | Content | Description |
|------------|--------------|---------|-------------|
| 0 - 4095 | 4096 | **Image Patch (Feature)** | Raw pixel data from 64x64 Y-channel patch, flattened into 1D array |
| 4096 - 4159 | 64 | **Padding/Unused Space** | Initialized to 255, not explicitly filled with data |
| 4160 - 4991 | 832 | **CU Depth Labels** | Structured as sparse array for QPs 0-51 |

### Label Data Structure

The 832-byte label section is structured like a sparse array where the QP value determines the position:
- Total size: 52 * 16 = 832 bytes (accommodates QPs from 0 to 51)
- For each QP, the offset is calculated as: `i_start_in_buf = 4096 + 64 + qp_list[i_qp] * 16`

For QPs [22, 27, 32, 37], the 16-byte CU depth data is written at:
- QP 22: Position 4160 + (22 * 16) = 4512
- QP 27: Position 4160 + (27 * 16) = 4592
- QP 32: Position 4160 + (32 * 16) = 4672
- QP 37: Position 4160 + (37 * 16) = 4752

Each 16-byte chunk represents CU depth information for a 64x64 block, corresponding to a 4x4 grid of depth values (since the smallest CU is 8x8 and stored for 16x16 units).

## Data Extraction Pipeline

### Step 1: File Identification (get_file_list)

For each video sequence, the system needs:
- `.yuv file`: Raw uncompressed video data from YUV_PATH_ORI directory
- `.dat files`: CU depth information from INFO_PATH directory
  - For each video and each QP (22, 27, 32, 37): `Info*...CUDepth.dat` file

### Step 2: Reading Video and Info Frames (generate_data loop)

For each video sequence:
1. Open the `.yuv` file
2. Open all four corresponding `.dat` files (one for each QP)
3. For each frame:
   - Read Y, U, V components using `read_YUV420_frame` (only Y component is used)
   - Read CU depth map from each `.dat` file using `read_info_frame`

### Step 3: Creating Samples from Frames (write_data)

For each frame:
1. Treat frame as grid of non-overlapping 64x64 blocks
2. For every 64x64 block:
   - **Extract Feature**: Take 64x64 pixel patch from Y-channel (4096 bytes)
   - **Extract Labels**: For each QP, extract 4x4 grid of depth values from CU depth map (16 bytes per QP)
   - **Assemble Sample**: Create 4992-byte structure

Sample assembly:
```python
buf_sample[0:4096] = np.reshape(patch_Y, (4096,))  # Image data
# Skip padding bytes 4096-4159
# Label data at calculated positions for each QP
```

### Step 4: Final Aggregation and Shuffling

1. Concatenate all 4992-byte samples from all blocks, frames, and videos into one large binary file
2. Use `shuffle_samples` function to randomize sample order for training

## Training Pipeline

### Data Loading

The training system loads data with specific structure:
- Images: 64x64 pixel patches reshaped to (64, 64, 1)
- Labels: 16-byte CU depth information reshaped to 4x4 matrix
- QPs: Quantization parameter values

Sample loading from dataset:
```python
labels[i,:] = data[i, 4160+qps[i,0]*NUM_LABEL_BYTES:4160+(qps[i,0]+1)*NUM_LABEL_BYTES]
```

Where `NUM_LABEL_BYTES = 16`.

### Network Architecture (ETH-CNN)

The ETH-CNN generates predictions at three hierarchical levels:
- 64x64 CU split decision
- 32x32 CU split decisions  
- 16x16 CU split decisions

The 4x4 CU depth matrix is converted to hierarchical split decisions where each depth value represents:
- `0`: 64x64 CU (depth 0)
- `1`: 32x32 CU (depth 1) 
- `2`: 16x16 CU (depth 2)
- `3`: 8x8 CU (depth 3)

## Testing Pipeline

### Integration Process

The modified HEVC encoder (HM-16.5_Test_AI) integrates CNN predictions:

1. **Pre-Processing**: Before encoding the first frame, HM invokes `video_to_cu_depth.py` via command line with parameters (YUV file name, frame width, frame height, QP)

2. **Prediction Generation**: The Python program reads the YUV file and predicts CU partition probability for all frames, saving results in `cu_depth.dat`

3. **Encoding with Predictions**: HM encodes frames according to predicted CU partition probability from `cu_depth.dat`, skipping redundant RD cost checking

### Modified Source Files

Four C++ files in HM 16.5 have been modified:
- `source/App/TAppEncoder/TAppEncCfg.cpp` - Invokes Python program
- `source/Lib/TLibCommon/TComPic.h` - Data structures
- `source/Lib/TLibEncoder/TEncGOP.cpp` - GOP encoding integration
- `source/Lib/TLibEncoder/TEncCu.cpp` - CU decision integration

Python files for prediction:
- `video_to_cu_depth.py` - Main prediction script
- `net_CNN.py` - Network architecture for inference

### Threshold Configuration

Thresholds are set in `Thr_info.txt` with format: `[ᾱ₁ α₁ ᾱ₂ α₂ ᾱ₃ α₃]`

Example: `[0.5 0.5 0.5 0.5 0.5 0.5]`

## Training Data Requirements

The system requires:
- 12 YUV files for training data
- 96 `Info_XX.dat` files (optional, pre-provided in `AI_Info/` folder)

These are generated by compressing 12 YUV files with encoder `HM-16.5_Extract_Data/bin/TAppEncoderStatic` at 4 QPs to extract:
- `str_XX.bin` files
- `Info_XX.dat` files  
- `log_XX.txt` files

## Configuration Variables

In `Extract_Data/extract_data_AI.py`:
```python
YUV_PATH_ORI = "path/to/yuv/files"     # Directory containing YUV files
INFO_PATH = "AI_Info"                   # Directory containing CU depth files
```

## Usage Instructions

### Data Extraction
```bash
cd Extract_Data
python extract_data_AI.py
```
This creates training, validation and test data files. Each data file is shuffled during execution with sample size of 4992 bytes.

### Training
```bash
cd ETH-CNN_Training_AI
python train_CNN_CTU64.py
```
Follow instructions in `ETH-CNN_Training_AI/readme.txt` for detailed configuration.

### Testing
1. Install TensorFlow (versions ≥ 1.8.0)
2. Navigate to `HM-16.5_Test_AI/Release`
3. Set thresholds in `Thr_info.txt`
4. Run `TAppEncoderStatic` (Linux) or `TAppEncoder.exe` (Windows)

Example scripts: `RUN_AI.sh` and `RUN_AI.bat`

## Technical Notes

- Python 3 is assumed as default. If not, edit line 2319 of `TAppEncCfg.cpp` and change "python" to "python3"
- Linux 64-bit platform is recommended for high-resolution video sequences
- YUV file paths should be shorter than 900 characters due to command line length limitations
- The system processes frames as YUV 4:2:0 format
- Sample length calculation: `NUM_SAMPLE_LENGTH = (64 * 64 * 1) + 64 + (52) * 16 = 4992 bytes`
