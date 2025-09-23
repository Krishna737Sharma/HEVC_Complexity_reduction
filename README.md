# Accelerated HEVC: Deep Learning Approaches for Fast Video Encoding

## Overview

This repository contains multiple implementations of deep learning-based approaches to accelerate High Efficiency Video Coding (HEVC) encoding by predicting optimal Coding Unit (CU) partitions. The project builds upon the official ITU-T H.265 | ISO/IEC 23008-2 reference software (HM-16.5) and integrates various neural network architectures to reduce encoding complexity while maintaining compression efficiency.

## Reference Software Foundation

The reference software included in this package is the official implementation for Rec. ITU-T H.265 | ISO/IEC 23008-2 High efficiency video coding (HEVC), jointly developed by:

- **ITU-T Video Coding Experts Group (VCEG)** - Question 6 of ITU-T Study Group 16
- **ISO/IEC Moving Picture Experts Group (MPEG)** - Working Group 11 of Subcommittee 29 of ISO/IEC Joint Technical Committee 1

The reference software provides both encoder and decoder functionality, serving as a foundation for establishing conformance, testing interoperability, and demonstrating HEVC capabilities. A comprehensive software manual with usage instructions can be found in the `doc` subdirectory.

## Repository Structure

This repository is organized into several specialized branches, each focusing on different aspects of HEVC acceleration:

### Main Branch
- **HM-16.5-official**: Contains the unmodified official HEVC reference software
- **Documentation**: Complete software manual and usage instructions in the `doc` directory

### Specialized Implementation Branches

#### 1. [ETH-CNN_Pytorch](https://github.com/Krishna737Sharma/Accelerated_HEVC/tree/ETH-CNN_Pytorch)
**Enhanced Three-branch Hierarchical Convolutional Neural Network**

- **Architecture**: Multi-scale three-branch CNN with hierarchical processing
- **Features**: 
  - 64x64, 32x32, and 16x16 level predictions
  - QP-integrated fully connected layers
  - Custom hierarchical loss function
  - Wandb experiment tracking
- **Performance**: Baseline CNN performance with comprehensive training pipeline
- **Framework**: PyTorch implementation with advanced data loading and checkpointing

#### 2. [HEVC_ViT](https://github.com/Krishna737Sharma/Accelerated_HEVC/tree/HEVC_ViT)
**Vision Transformer for HEVC Encoding**

- **Architecture**: Custom Vision Transformer with QP integration
- **Key Features**:
  - 8x8 patch embedding for 64x64 CTU blocks
  - 5-layer transformer encoder with 4 attention heads
  - QP-aware transformer layers
  - 21-dimensional hierarchical output
- **Performance**: **0.85% average BD-Rate improvement** over CNN baseline
- **Innovation**: First ViT application to HEVC CU partition prediction

#### 3. [HEVC_CNN](https://github.com/Krishna737Sharma/Accelerated_HEVC/tree/HEVC_CNN)
**Classical CNN Implementation**

- **Architecture**: Traditional CNN approach for CU partition prediction
- **Purpose**: Baseline implementation and comparison reference
- **Integration**: Direct HEVC encoder integration with C++ modifications
- **Features**: Optimized inference pipeline for real-time encoding

#### 4. [CPH-Intra-Dataset-Preparation](https://github.com/Krishna737Sharma/Accelerated_HEVC/tree/CPH-Intra-Dataset-Preparation-for-Intra-Prediction-Models)
**Dataset Preparation Pipeline**

- **Dataset**: CPH-Intra dataset processing and preparation
- **Features**:
  - Multi-resolution support (768×512 to 4928×3264)
  - Automated download from multiple sources (Google Drive, Dropbox, Baidu)
  - Pre-processed dataset availability via Kaggle Hub
- **Output**: 2.8M+ training samples in standardized format
- **Utilities**: Complete data extraction and shuffling pipeline

## Performance Comparison

### BD-Rate Results (Lower is Better)

| Video Sequence | Resolution | CNN BD-Rate (%) | ViT BD-Rate (%) | Improvement |
|---|---|---|---|---|
| IntraValid_4928x3264 | 4928×3264 | 2.12 | **2.04** | 0.08% |
| Rush_Hour | 3840×2160 | 5.88 | **5.03** | 0.85% |
| Netflix_FoodMarket2 | 4096×2160 | 6.45 | **5.22** | 1.23% |
| Scarf | 3840×2160 | 4.02 | **3.53** | 0.49% |
| Construction_Field | High-res | 6.49 | **4.89** | 1.60% |

**Key Findings:**
- Vision Transformer consistently outperforms traditional CNN approaches
- Average improvement of **0.85% BD-Rate reduction** across test sequences
- Larger improvements observed on complex, high-resolution content
- Maintained encoding speed while improving compression efficiency

## Technical Approach

### Problem Statement
HEVC encoding involves computationally expensive Rate-Distortion (RD) optimization for determining optimal CU partitions. Traditional encoders evaluate all possible partition configurations, resulting in significant computational overhead.

### Solution Strategy
1. **Feature Extraction**: Extract 64x64 luminance patches from video frames
2. **Neural Prediction**: Use deep learning models to predict optimal CU splits
3. **Encoder Integration**: Modify HEVC reference software to utilize predictions
4. **RD Optimization**: Skip unnecessary RD evaluations based on model confidence

### Data Pipeline
```
Raw YUV Video → 64x64 Patches → Feature Extraction → Neural Network → CU Predictions → HEVC Encoder
```

### Integration Architecture
```
HEVC Encoder (C++) ←→ Python Prediction Interface ←→ Trained Models (PyTorch)
```

## Getting Started

### Prerequisites
```bash
# Required software
- Python 3.7+
- PyTorch 1.9.0+
- OpenCV
- NumPy, Matplotlib
- CMake (for HEVC compilation)
- GCC/Visual Studio (platform dependent)
```

### Quick Start

1. **Clone Repository**:
```bash
git clone https://github.com/Krishna737Sharma/Accelerated_HEVC.git
cd Accelerated_HEVC
```

2. **Choose Implementation Branch**:
```bash
# For Vision Transformer (recommended)
git checkout HEVC_ViT

# For Enhanced CNN
git checkout ETH-CNN_Pytorch

# For dataset preparation
git checkout CPH-Intra-Dataset-Preparation-for-Intra-Prediction-Models
```

3. **Follow Branch-Specific Instructions**: Each branch contains detailed README files with setup and usage instructions.

## Dataset Information

### CPH-Intra Dataset
- **Training Samples**: 2,446,725
- **Validation Samples**: 143,925  
- **Test Samples**: 287,850
- **Resolutions**: 768×512, 1536×1024, 2880×1920, 4928×3264
- **QP Range**: 22, 27, 32, 37
- **Format**: 4992 bytes per sample (4096 image + 832 label + 64 padding)

### Data Availability
- **Kaggle Hub**: Pre-processed datasets available for direct download
- **Original Sources**: Google Drive, Dropbox, Baidu Cloud options
- **Processing Tools**: Complete extraction and preparation pipeline included

## Model Architectures

### ETH-CNN (Enhanced Three-branch CNN)
```
Input [64×64] → Three Parallel Branches → Hierarchical Predictions
├── Branch 1: 64×64 → 16×16 → Conv → FC → 1 output (64×64 split)
├── Branch 2: 64×64 → 32×32 → Conv → FC → 4 outputs (32×32 splits)
└── Branch 3: 64×64 → 64×64 → Conv → FC → 16 outputs (16×16 splits)
```

### Vision Transformer (ViT)
```
Input [64×64] → Patch Embedding [8×8 patches] → Transformer Layers → Classification Head
├── Patch Embedding: 64 patches of 8×8 pixels
├── Transformer: 5 layers, 4 heads, QP integration
└── Output: 21-dimensional hierarchical prediction
```

## Applications

### Research Applications
- Video coding acceleration research
- Deep learning in multimedia systems
- Rate-distortion optimization studies
- Transformer applications in signal processing

### Industrial Applications
- Real-time video streaming optimization
- Mobile device encoding acceleration
- Cloud-based video processing
- Broadcasting and media production

## Evaluation Metrics

### Compression Efficiency
- **BD-Rate (Bjøntegaard Delta Rate)**: Bitrate savings at equivalent quality
- **BD-PSNR**: Quality improvement at equivalent bitrate
- **Rate-Distortion Curves**: Comprehensive quality vs bitrate analysis

### Computational Complexity
- **Encoding Time Reduction**: Percentage decrease in encoding time
- **Model Inference Time**: Neural network prediction overhead
- **Memory Usage**: Runtime memory consumption analysis

## Contributing

We welcome contributions to improve the implementations and extend the research:

1. **Bug Reports**: Submit issues for any problems encountered
2. **Feature Requests**: Propose new architectures or improvements
3. **Performance Optimizations**: Contribute speed or accuracy improvements
4. **Documentation**: Help improve documentation and examples

### Development Guidelines
- Follow branch-specific coding standards
- Include comprehensive testing for new features
- Document all significant changes
- Maintain backward compatibility where possible

## License

This project builds upon the HEVC reference software and follows its licensing terms. Additional implementations are provided for research and educational purposes. Please refer to individual branch licenses for specific terms.

## Citation

If you use this work in your research, please cite the relevant papers and acknowledge the HEVC reference software:

```bibtex
@misc{accelerated_hevc_2024,
  title={Accelerated HEVC: Deep Learning Approaches for Fast Video Encoding},
  author={Krishna Sharma and Contributors},
  year={2024},
  publisher={GitHub},
  url={https://github.com/Krishna737Sharma/Accelerated_HEVC}
}
```

## Acknowledgments

- **HEVC Reference Software Team**: For the foundational HM-16.5 implementation
- **CPH-Intra Dataset Authors**: For providing the comprehensive training dataset
- **Research Community**: For ongoing contributions and feedback
- **Open Source Contributors**: For improvements and bug fixes

## Support

For questions, issues, or collaboration opportunities:

- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Join technical discussions in the repository
- **Documentation**: Refer to branch-specific README files for detailed information

---

**Note**: This repository contains implementations for research and educational purposes. For production use, please ensure appropriate testing and validation for your specific requirements.
