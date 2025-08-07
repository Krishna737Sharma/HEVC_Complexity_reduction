#!/usr/bin/env python3
# ===============================================================
#  ViT Evaluation on ETH-CNN Dataset (Compatible with Training)
# ===============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random
import os

# ==================== Constants (Same as Training) ====================
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_LABEL_BYTES = 16
NUM_SAMPLE_LENGTH = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + 64 + (51 + 1) * NUM_LABEL_BYTES
SELECT_QP_LIST = [22, 27, 32, 37]

# Use validation dataset path
VAL_FILE = "/home/ai-iitkgp/Downloads/HEVC_Intra_Models-ViT/ViT_2.3M/Data/AI_Valid_5000.dat_shuffled"


# ==================== StreamingDataset (Same as Training) ====================
class StreamingDataset(Dataset):
    def __init__(self, file_path, max_samples):
        self.file_path = file_path
        self.max_samples = max_samples

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        with open(self.file_path, 'rb') as file_reader:
            offset = idx * NUM_SAMPLE_LENGTH
            file_reader.seek(offset)
            data = np.frombuffer(file_reader.read(NUM_SAMPLE_LENGTH), dtype=np.uint8)

            # Process image
            image = data[:4096].astype(np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

            # Process QP (randomly select from list)
            qp = np.random.choice(SELECT_QP_LIST, size=1)[0]

            # Process label using QP index
            label = np.zeros((NUM_LABEL_BYTES,))
            qp_index = int(qp)
            label[:] = data[4160 + qp_index * NUM_LABEL_BYTES: 4160 + (qp_index + 1) * NUM_LABEL_BYTES]

            # Convert image and QP to tensors
            ctu_tensor = torch.tensor(image, dtype=torch.float32).squeeze(2)
            qp_tensor = torch.tensor(float(qp), dtype=torch.float32)

            # Scale tensors
            ctu_tensor /= 255.0
            qp_tensor /= 51.0

            # Convert label to hierarchical output
            y_image = torch.tensor(label, dtype=torch.float32).view(1, 4, 4)

            # Hierarchical pooling/activation steps
            y_image_16 = F.relu(y_image - 2)
            y_image_32 = F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 1) - \
                         F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 2)
            y_image_64 = F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=4) - 0) - \
                         F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=4) - 1)
            y_image_valid_32 = F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 0) - \
                               F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 1)
            y_image_valid_16 = F.relu(y_image - 1) - F.relu(y_image - 2)

            # Flatten the hierarchical outputs
            y_flat_16 = y_image_16.view(-1)
            y_flat_32 = y_image_32.view(-1)
            y_flat_64 = y_image_64.view(-1)
            y_flat_valid_32 = y_image_valid_32.view(-1)
            y_flat_valid_16 = y_image_valid_16.view(-1)

            # Concatenate hierarchical outputs into one target vector (21-dim)
            target = torch.cat((y_flat_64, y_flat_32, y_flat_16), dim=0)

            return qp_tensor, ctu_tensor, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, target


def create_subset_dataloader(file_path, total_samples, subset_size, batch_size, shuffle=True):
    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    full_dataset = StreamingDataset(file_path, total_samples)
    subset_indices = random.sample(range(total_samples), subset_size)
    return DataLoader(
        Subset(full_dataset, subset_indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )


# ==================== Model Architecture (Same as Training) ====================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=1, embed_dim=196):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model + 1, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward + 1, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, src, qp):
        # Self-attention
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        B = src.shape[1]
        seq_len = src.shape[0]
        qp_exp = qp.view(1, B, 1).expand(seq_len, B, 1)

        # First linear layer with QP concatenation
        src_cat = torch.cat([src, qp_exp], dim=-1)
        src2 = self.linear1(src_cat)
        src2 = self.activation(src2)
        src2 = self.dropout1(src2)

        # Second linear layer with QP concatenation
        qp_exp2 = qp.view(1, B, 1).expand(seq_len, B, 1)
        src2_cat = torch.cat([src2, qp_exp2], dim=-1)
        src2 = self.linear2(src2_cat)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class VisionTransformer(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=1, num_classes=21,
                 embed_dim=196, depth=5, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        dim_feedforward = int(embed_dim * mlp_ratio)
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                          dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu")
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x, qp):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = x.transpose(0, 1)
        for layer in self.encoder_layers:
            x = layer(x, qp)
        x = x[0]  # Use the CLS token
        x = self.norm(x)
        logits = self.head(x)
        out = torch.sigmoid(logits)
        return out


# ==================== Accuracy Function (Same as Training) ====================
def calculate_accuracy_repo(y_flat_64, y_conv_flat_64, y_flat_32, y_conv_flat_32, y_flat_valid_32,
                            y_flat_16, y_conv_flat_16, y_flat_valid_16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_flat_64 = y_flat_64.to(device, non_blocking=True)
    y_conv_flat_64 = y_conv_flat_64.to(device, non_blocking=True)
    y_flat_32 = y_flat_32.to(device, non_blocking=True)
    y_conv_flat_32 = y_conv_flat_32.to(device, non_blocking=True)
    y_flat_valid_32 = y_flat_valid_32.to(device, non_blocking=True)
    y_flat_16 = y_flat_16.to(device, non_blocking=True)
    y_conv_flat_16 = y_conv_flat_16.to(device, non_blocking=True)
    y_flat_valid_16 = y_flat_valid_16.to(device, non_blocking=True)
    epsilon = 1e-12

    correct_prediction_64 = torch.round(y_conv_flat_64) == torch.round(y_flat_64)
    accuracy_64 = torch.mean(correct_prediction_64.float()) * 100

    correct_prediction_valid_32 = y_flat_valid_32 * (torch.round(y_conv_flat_32) == torch.round(y_flat_32)).float()
    accuracy_32 = torch.sum(y_flat_valid_32 * correct_prediction_valid_32) / (
                torch.sum(y_flat_valid_32) + epsilon) * 100

    correct_prediction_valid_16 = y_flat_valid_16 * (torch.round(y_conv_flat_16) == torch.round(y_flat_16)).float()
    accuracy_16 = torch.sum(y_flat_valid_16 * correct_prediction_valid_16) / (
                torch.sum(y_flat_valid_16) + epsilon) * 100

    avg_acc = (accuracy_64 + accuracy_32 + accuracy_16) / 3
    return avg_acc, accuracy_64, accuracy_32, accuracy_16


# ==================== Evaluation Function ====================
def evaluate_vit():
    """Evaluate ViT on validation partition"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model with same architecture as training
    print("Initializing ViT model...")
    model = VisionTransformer(
        img_size=64,
        patch_size=8,
        in_chans=1,
        num_classes=21,
        embed_dim=196,
        depth=5,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1
    ).to(device)

    # Try to load checkpoint
    checkpoint_path = "/home/ai-iitkgp/Downloads/HEVC_Intra_Models-ViT/ViT_2.3M/best_vit_model.pth"
    print(f"Loading checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✅ Checkpoint loaded successfully!")
        epoch = checkpoint.get('epoch', 'Unknown')
        best_loss = checkpoint.get('best_loss', 'Unknown')
        print(f"📊 Model from epoch {epoch}, best loss: {best_loss}")
    except Exception as e:
        print(f"❌ Checkpoint loading failed: {e}")
        print("📊 Evaluating with random weights instead...")

    # Create validation loader (same format as ETH-CNN)
    print("Loading validation dataset...")
    BATCH_SIZE = 32
    SUB_VAL = 1000  # Same as ETH-CNN evaluation
    val_loader = create_subset_dataloader(VAL_FILE, 5000, SUB_VAL, BATCH_SIZE, shuffle=False)
    print(f"Validation samples: {SUB_VAL}")

    # Evaluation
    model.eval()
    print("Evaluating ViT on validation partition...")

    total_acc = 0.0
    total_acc_l1 = 0.0
    total_acc_l2 = 0.0
    total_acc_l3 = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            qp_batch, ctu_batch, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, target = batch

            inputs = ctu_batch.to(device).unsqueeze(1)  # [B, 1, 64, 64]
            qp_tensor = qp_batch.to(device)

            # Forward pass
            outputs = model(inputs, qp_tensor)  # [B, 21]

            # Calculate accuracy using same function as training
            avg_acc, acc64, acc32, acc16 = calculate_accuracy_repo(
                y_flat_64, outputs[:, 0:1],  # 64x64 predictions
                y_flat_32, outputs[:, 1:5], y_flat_valid_32,  # 32x32 predictions
                y_flat_16, outputs[:, 5:21], y_flat_valid_16  # 16x16 predictions
            )

            total_acc += avg_acc.item()
            total_acc_l1 += acc64.item()
            total_acc_l2 += acc32.item()
            total_acc_l3 += acc16.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches")

    # Calculate final accuracies
    final_acc = total_acc / num_batches
    final_acc_l1 = total_acc_l1 / num_batches
    final_acc_l2 = total_acc_l2 / num_batches
    final_acc_l3 = total_acc_l3 / num_batches

    return final_acc, final_acc_l1, final_acc_l2, final_acc_l3


def compare_all_models():
    """Compare all models on validation partition"""
    print("=" * 60)
    print("MODEL COMPARISON ON VALIDATION PARTITION")
    print("=" * 60)

    # Previous results
    tf_results = {
        "QP_22": 75.56, "QP_27": 74.61,
        "QP_32": 73.98, "QP_37": 72.35
    }
    tf_avg = np.mean(list(tf_results.values()))

    # Evaluate ViT
    vit_acc, vit_acc_l1, vit_acc_l2, vit_acc_l3 = evaluate_vit()

    print("\nRESULTS:")
    print("-" * 60)
    print(f"TensorFlow ETH-CNN (Pre-trained): {tf_avg:.2f}%")
    print(f"ViT (Trained checkpoint):         {vit_acc:.2f}%")

    print(f"\nViT Detailed Accuracies:")
    print(f"- Overall:  {vit_acc:.2f}%")
    print(f"- 64x64:    {vit_acc_l1:.2f}%")
    print(f"- 32x32:    {vit_acc_l2:.2f}%")
    print(f"- 16x16:    {vit_acc_l3:.2f}%")


if __name__ == "__main__":
    print("Starting ViT evaluation with training-compatible architecture...")
    compare_all_models()
