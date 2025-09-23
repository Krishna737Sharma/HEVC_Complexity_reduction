# Vision Transformer (ViT) for HEVC Intra Coding Unit Prediction

## Overview

This project implements a Vision Transformer (ViT) model to replace CNN-based approaches for predicting optimal Coding Unit (CU) partitions in HEVC intra encoding. The ViT model processes 64x64 pixel patches from video frames and predicts hierarchical CU split decisions to accelerate the encoding process.

## Dataset Structure and Processing

### Raw Data Format (4992 bytes per sample)

Each training sample represents one 64x64 CTU and contains three main sections:

```
[Image Data: 4096 bytes] [Padding: 64 bytes] [Label Data: 832 bytes]
    |                         |                      |
    v                         v                      v
64x64 luma pixels        Unused space        CU depths for all QPs (0-51)
```
### Note:
The model was trained on a self-prepared 4K resolution dataset derived from the CPH dataset.

### Label Processing
The model uses hierarchical label generation:
```python
# Convert CU depths to hierarchical split decisions
y_image_16 = F.relu(y_image - 2)  # 16x16 splits
y_image_32 = F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 1) - \
             F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 2)  # 32x32 splits
y_image_64 = F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=4) - 0) - \
             F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=4) - 1)  # 64x64 splits
```

Final output: 21-dimensional vector (1 + 4 + 16 for 64x64, 32x32, and 16x16 split decisions)

## Model Architecture

```
Input: [Batch, 1, 64, 64] image + [Batch] QP values
    ↓
[PatchEmbed] → Convert to [Batch, 64, 196] patches
    ↓
[Add CLS Token + Position Embedding] → [Batch, 65, 196]
    ↓
[5 × CustomTransformerEncoderLayer] → Process with attention + QP integration
    ↓
[Classification Head] → [Batch, 21] partition predictions
    ↓
Output: Sigmoid activation → probabilities for each split decision
```

### Vision Transformer Configuration
```python
VisionTransformer(
    img_size=64,           # Input image size
    patch_size=8,          # Patch size (creates 8x8 = 64 patches)
    in_chans=1,            # Single channel (luma only)
    num_classes=21,        # Hierarchical output classes
    embed_dim=196,         # Embedding dimension
    depth=5,               # Number of transformer layers
    num_heads=4,           # Multi-head attention heads
    mlp_ratio=4.0,         # MLP expansion ratio
    dropout=0.1            # Dropout rate
)
```

### Key Components

1. **PatchEmbed**: Converts 64x64 image into 64 patches of 8x8 pixels each
2. **CustomTransformerEncoderLayer**: Integrates QP information into transformer layers
3. **QP Integration**: Quantization Parameter is normalized (qp/51.0) and concatenated with features

### QP-Integrated Transformer Layer
```python
# QP is concatenated with features in MLP layers
self.linear1 = nn.Linear(d_model + 1, dim_feedforward)  # +1 for QP
self.linear2 = nn.Linear(dim_feedforward + 1, d_model)  # +1 for QP

# Forward pass with QP concatenation
src_cat = torch.cat([src, qp_exp], dim=-1)
```

## Training Configuration

### Dataset Parameters
```python
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_LABEL_BYTES = 16
SELECT_QP_LIST = [22, 27, 32, 37]
BATCH_SIZE = 64
```

### Training Setup
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.3163)
criterion = nn.BCELoss()
```

### Dataset Sizes
- Training: 1,668,975 samples (subset of 80,000 used)
- Validation: 98,175 samples (subset of 60,000 used)
- Test: 287,850 samples

## Performance Evaluation

### Accuracy Calculation
The model uses hierarchical accuracy calculation:
```python
def calculate_accuracy_repo(y_flat_64, y_conv_flat_64, y_flat_32, y_conv_flat_32, 
                           y_flat_valid_32, y_flat_16, y_conv_flat_16, y_flat_valid_16):
    # 64x64 accuracy
    accuracy_64 = torch.mean((torch.round(y_conv_flat_64) == torch.round(y_flat_64)).float()) * 100
    
    # 32x32 accuracy (weighted by valid mask)
    accuracy_32 = torch.sum(y_flat_valid_32 * correct_prediction_valid_32) / 
                  (torch.sum(y_flat_valid_32) + epsilon) * 100
    
    # 16x16 accuracy (weighted by valid mask)
    accuracy_16 = torch.sum(y_flat_valid_16 * correct_prediction_valid_16) / 
                  (torch.sum(y_flat_valid_16) + epsilon) * 100
    
    avg_acc = (accuracy_64 + accuracy_32 + accuracy_16) / 3
```

## Performance Results

### BD-Rate Comparison
BD-Rate (Bjøntegaard Delta Rate) measures bitrate savings at equivalent quality. Lower values indicate better compression efficiency.

| Video Sequence | Resolution | Frames | CNN BD-Rate (%) | ViT BD-Rate (%) | Improvement |
|---|---|---|---|---|---|
| IntraValid_4928x3264.yuv | 4928×3264 | 25 | 2.12 | **2.04** | 0.08% better |
| Rush_Hour.yuv | 3840×2160 | 250 | 5.88 | **5.03** | 0.85% better |
| Netflix_FoodMarket2_4096x2160.yuv | 4096×2160 | 150 | 6.45 | **5.22** | 1.23% better |
| Scarf.yuv | 3840×2160 | 100 | 4.02 | **3.53** | 0.49% better |
| Construction_Field.yuv | High-resolution | 7 | 6.49 | **4.89** | 1.60% better |

**Key Findings:**
- ViT consistently outperforms CNN across all test sequences
- Improvements range from 0.08% to 1.60% BD-Rate reduction
- Larger improvements on complex sequences (Construction_Field, Netflix_FoodMarket2)
- Average improvement: ~0.85% BD-Rate reduction

**Average Improvement**: ViT achieves better BD-Rate performance compared to CNN across all test sequences.

### Rate-Distortion Curves

The following plots show bitrate vs PSNR for different encoding methods. Lower bitrate at same PSNR indicates better performance.

<img width="700" height="500" alt="rd_curve_Netflix_FoodMarket2_4096x2160_150f" src="https://github.com/user-attachments/assets/cd94dac7-8bde-42fd-8d06-9ddd896e0601" />
<img width="700" height="500" alt="rd_curve_Construction_Field_7f" src="https://github.com/user-attachments/assets/0291fa16-425d-4377-aa19-85bc98b5dd8f" />


## Integration with HEVC Encoder

### Model Loading for Inference
```python
class ViTPredictor:
    def __init__(self, model_path):
        self.model = VisionTransformer(
            img_size=64, patch_size=8, in_chans=1, num_classes=21,
            embed_dim=196, depth=5, num_heads=4
        )
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
```

### Prediction Pipeline
1. **Initialization**: Load trained ViT model (`net_ViT.initialize_vit_predictor()`)
2. **Frame Processing**: Extract 64x64 patches from YUV frames
3. **Batch Prediction**: Process patches in batches of 1024
4. **Output Generation**: Save predictions to `cu_depth.dat`

### Modified HEVC Integration Files
- `net_ViT.py`: ViT model definition and prediction interface
- `video_to_cu_depth.py`: Main prediction script called by HEVC encoder
- HEVC encoder files (same as CNN version):
  - `TAppEncCfg.cpp`: Invokes Python prediction
  - `TEncGOP.cpp`, `TEncCu.cpp`: Use predictions for RD optimization

## Usage Instructions

### Training
```python
# Training script with ViT model
model = VisionTransformer(
    img_size=IMAGE_SIZE,
    patch_size=8,
    in_chans=NUM_CHANNELS,
    num_classes=21,
    embed_dim=196,
    depth=5,
    num_heads=4
).to(device)

# Training loop with QP integration
for batch in train_loader:
    qp_batch, ctu_batch, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, target = batch
    inputs = ctu_batch.to(device).unsqueeze(1)
    qp_tensor = qp_batch.to(device)
    outputs = model(inputs, qp_tensor)
    loss = criterion(outputs, target)
```

### Testing
```python
# Load trained model for testing
checkpoint = torch.load("best_vit_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Run inference
with torch.no_grad():
    outputs = model(inputs, qp_tensor)
```

### Integration with HEVC Encoder
```bash
# The HEVC encoder calls the Python script:
python video_to_cu_depth.py input.yuv width height qp

# Which generates cu_depth.dat for encoding optimization
```

## Model Parameters

**Total Parameters**: ~2.3M (significantly larger than CNN baseline)
**Inference Device**: CPU (for integration with HEVC encoder)
**Memory Requirements**: ~50MB for model storage

## Key Features

1. **QP-Aware Architecture**: Quantization parameter integrated into transformer layers
2. **Hierarchical Prediction**: Multi-scale CU split decisions (64x64, 32x32, 16x16)
3. **Efficient Integration**: Direct model loading eliminates subprocess overhead
4. **Improved Performance**: Better BD-Rate compared to CNN baseline
5. **Batch Processing**: Optimized inference for large video sequences

## File Structure

```
├── net_ViT.py                    # ViT model definition and inference
├── video_to_cu_depth.py         # Main prediction script
├── train_vit.py                 # Training script
├── test_vit.py                  # Testing script  
├── best_vit_model.pth           # Trained model weights
└── Thr_info.txt                 # Threshold configuration
```

## Constants and Configuration

```python
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_LABEL_BYTES = 16
NUM_SAMPLE_LENGTH = 4992  # 4096 + 64 + 832
SELECT_QP_LIST = [22, 27, 32, 37]
DEVICE = torch.device('cpu')  # CPU inference for encoder integration
```
