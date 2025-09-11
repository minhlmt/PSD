# PSD (Parameter Space Distillation) - Usage Guide

## Overview

This project implements the PSD (Parameter Space Distillation) algorithm using BigGAN to generate synthetic data for training classification models. The project uses the MMGeneration framework and supports datasets such as CIFAR-10 and SVHN.

## System Requirements

- Python 3.7+
- CUDA 11.8+ (recommended)
- GPU with at least 8GB VRAM

## Installation Guide

### Step 1: Install PyTorch and Basic Libraries

```bash
# Install PyTorch with CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install MMCV
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

# Install MMClassification
pip install mmcls
```

### Step 2: Clone MMGeneration

```bash
# Clone MMGeneration repository
git clone https://github.com/open-mmlab/mmgeneration.git

# Install MMGeneration
cd mmgeneration
pip install -e .
cd ..
```

### Step 3: Create Directories and Download Pretrained Models

```bash
# Create directories for pretrained models
mkdir -p pretrain_model/cifar/biggan
mkdir -p pretrain_model/cifar/convnet
mkdir -p pretrain_model/cifar/resnet18
```

**Download pretrained models from Google Drive:**

- Access: https://drive.google.com/drive/folders/1_S26VJMOYjFDZdiyMPusG6eG0Sb7YBHl
- Download the following files to their respective directories:
  - `biggan_cifar10.pth` → `pretrain_model/cifar/biggan/`
  - `convnet_cifar10.pth` → `pretrain_model/cifar/convnet/`
  - `resnet18_cifar10.pth` → `pretrain_model/cifar/resnet18/`

### Step 4: Install Additional Libraries

```bash
# Install required libraries
pip install einops
pip install matplotlib
pip install pillow
```

## Directory Structure

```
PSD/
├── main_PSD.py              # Main file for running training
├── model_biggan.py          # BigGAN model configuration
├── networks.py              # Neural network definitions
├── utils.py                 # Utility functions
├── augment.py               # Data augmentation
├── pretrain_model/          # Directory containing pretrained models
│   └── cifar/
│       ├── biggan/
│       ├── convnet/
│       └── resnet18/
├── data/                    # Dataset directory (auto-created)
├── results/                 # Results directory (auto-created)
└── logs/                    # Logs directory (auto-created)
```

## Usage

### Basic Training

```bash
python main_PSD.py --data cifar10 --ipc 50 --epochs 300
```

### Important Parameters

- `--data`: Dataset to use (`cifar10`, `svhn`)
- `--ipc`: Images per class (1, 10, or 50)
- `--epochs`: Number of training epochs (default: 300)
- `--batch_size`: Batch size (default: 256)
- `--eval_model`: Model for evaluation (`convnet`, `resnet18`)
- `--weight_biggan`: Path to BigGAN pretrained model
- `--weight_convnet`: Path to ConvNet pretrained model
- `--weight_resnet`: Path to ResNet pretrained model

### Example with Custom Parameters

```bash
python main_PSD.py \
    --data cifar10 \
    --ipc 50 \
    --epochs 300 \
    --batch_size 128 \
    --eval_model convnet resnet18 \
    --weight_biggan ./pretrain_model/cifar/biggan/biggan_cifar10.pth \
    --weight_convnet ./pretrain_model/cifar/convnet/convnet_cifar10.pth \
    --weight_resnet ./pretrain_model/cifar/resnet18/resnet18_cifar10.pth \
    --output_dir ./results \
    --logs_dir ./logs
```

## Results

After running, you will see:

1. **Generated images**: Saved in `results/outputs/img_{epoch}.png`
2. **Model checkpoints**: Saved in `results/model_dict_{model}.pth`
3. **Training logs**: Saved in `logs/logs.txt`

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:

   - Reduce `--batch_size` (e.g., 64, 32)
   - Reduce `--ipc` (use 1, 10, or 50)

2. **ModuleNotFoundError: No module named 'mmgen'**:

   - Ensure MMGeneration is installed correctly
   - Check PYTHONPATH

3. **FileNotFoundError: pretrained model**:

   - Check pretrained model paths
   - Ensure correct files are downloaded from Google Drive
   - **IMPORTANT**: Code will crash if pretrained models are not found

4. **ImportError: No module named 'einops'**:

   ```bash
   pip install einops
   ```

5. **ImportError: No module named 'augment'**:

   - File `augment.py` must be in the same directory as `main_PSD.py`
   - Check directory structure

6. **RuntimeError: Error(s) in loading state_dict**:
   - Pretrained model is incompatible with model architecture
   - Check pretrained model versions

### Installation Check

```bash
# Check PyTorch and CUDA
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Check MMGeneration
python -c "from mmgen.models import build_model; print('MMGeneration OK')"

# Check other dependencies
python -c "import einops, matplotlib, PIL; print('All dependencies OK')"
```

### Quick Test Before Running

```bash
# Test import all modules
python -c "
from main_PSD import main
from model_biggan import model_biggan
from utils import load_data, diffaug
from augment import DiffAug
print('All imports successful!')
"

# Check pretrained models (if downloaded)
python -c "
import os
models = [
    './pretrain_model/cifar/biggan/biggan_cifar10.pth',
    './pretrain_model/cifar/convnet/convnet_cifar10.pth',
    './pretrain_model/cifar/resnet18/resnet18_cifar10.pth'
]
for model in models:
    if os.path.exists(model):
        print(f'✓ {model} exists')
    else:
        print(f'✗ {model} missing')
"
```

## Advanced Configuration

### Change Augmentation Strategy

```bash
python main_PSD.py --aug_type "color_crop_cutout_flip" --data cifar10
```

### Use Different Dataset

```bash
# SVHN
python main_PSD.py --data svhn --ipc 50
```

### Customize Model Evaluation

```bash
python main_PSD.py --eval_model convnet resnet18 alexnet --data cifar10
```

## Important Notes

1. **GPU Memory**: Project requires GPU with at least 8GB VRAM
2. **Training Time**: Each epoch may take 10-30 minutes depending on hardware
3. **Dataset**: CIFAR-10 and SVHN will be automatically downloaded if not available
4. **Pretrained Models**: Must download all pretrained models before running

## Issues Found in Code

### ⚠️ Issues to Fix:

1. **Missing error handling for pretrained models**:

   - Code will crash if pretrained model files are not found
   - No file existence check before loading

2. **Hardcoded paths**:

   - Some paths are hardcoded in the code
   - Need to ensure correct directory structure

3. **Missing dependencies**:
   - Code imports `einops` but it's not in requirements
   - Need to install: `pip install einops`

### 🔧 Recommended Improvements:

1. **Add error handling**:

   ```python
   if not os.path.exists(args.weight_biggan):
       raise FileNotFoundError(f"BigGAN model not found: {args.weight_biggan}")
   ```

2. **Check dependencies**:

   ```python
   try:
       from mmgen.models import build_model
   except ImportError:
       raise ImportError("MMGeneration not installed. Please install it first.")
   ```

3. **Add validation for pretrained models**:
   - Check model architecture compatibility
   - Validate state_dict keys

## Contact

If you encounter issues, please check:

1. Logs in the `logs/` directory
2. Ensure all dependencies are installed correctly
3. Check pretrained model paths
