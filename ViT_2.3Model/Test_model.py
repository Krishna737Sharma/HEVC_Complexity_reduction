import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# =================================================================================
# === Configuration: You can change these values ==================================
# =================================================================================
NUM_SAMPLES_TO_TEST = 100
QP_VALUE = 32
# --- Update with the correct path to your test data file ---
DATA_FILE = "/root/myproject/HEVC_Intra_Models-ViT/Data/AI_Valid_143925.dat_shuffled"
MODEL_PATH = "/root/myproject/HEVC_Intra_Models-ViT/ViT_2.3M/best_vit_model.pth"
# =================================================================================


# =================================================================================
# === Core Model and Data Logic (Copied from your files) ==========================
# =================================================================================

# --- Constants ---
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_LABEL_BYTES = 16
NUM_SAMPLE_LENGTH = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + 64 + (51 + 1) * NUM_LABEL_BYTES
THRESHOLDS = [0.5, 1.5, 2.5] # For interpreting ground truth depth labels

# --- PatchEmbed Class ---
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

# --- Custom Transformer Encoder ---
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
        self.activation = F.gelu if activation=="gelu" else F.relu
    def forward(self, src, qp):
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        B, seq_len = src.shape[1], src.shape[0]
        qp_exp = qp.view(1, B, 1).expand(seq_len, B, 1)
        src_cat = torch.cat([src, qp_exp], dim=-1)
        src2 = self.linear1(src_cat)
        src2 = self.activation(src2)
        src2 = self.dropout1(src2)
        qp_exp2 = qp.view(1, B, 1).expand(seq_len, B, 1)
        src2_cat = torch.cat([src2, qp_exp2], dim=-1)
        src2 = self.linear2(src2_cat)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# --- Vision Transformer Model ---
class VisionTransformer(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=1, num_classes=21, embed_dim=196, depth=5, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        dim_feedforward = int(embed_dim * mlp_ratio)
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu")
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None: nn.init.zeros_(self.head.bias)
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
        x = x[0]
        x = self.norm(x)
        logits = self.head(x)
        return torch.sigmoid(logits)

# =================================================================================
# === Helper Functions for Inference ============================================
# =================================================================================

def get_sample_for_qp(file_path, index, qp_value):
    """Reads a single sample from the binary file for a SPECIFIC qp_value."""
    with open(file_path, 'rb') as f:
        offset = index * NUM_SAMPLE_LENGTH
        f.seek(offset)
        data = np.frombuffer(f.read(NUM_SAMPLE_LENGTH), dtype=np.uint8)

    # 1. Extract and process image
    image = data[:4096].astype(np.float32) / 255.0
    ctu_tensor = torch.from_numpy(image).reshape(IMAGE_SIZE, IMAGE_SIZE)

    # 2. Process QP
    qp_tensor = torch.tensor(float(qp_value) / 51.0, dtype=torch.float32)

    # 3. Extract label for the specific QP and process it hierarchically
    label_start = 4160 + qp_value * NUM_LABEL_BYTES
    label_end = label_start + NUM_LABEL_BYTES
    label = data[label_start:label_end]
    
    # --- Convert raw label to ground truth split decisions ---
    depths = np.array(label).reshape(4, 4)
    gt_split_64 = 1 if np.mean(depths) > THRESHOLDS[0] else 0
    gt_splits_32 = []
    if gt_split_64:
        for i in range(2):
            for j in range(2):
                sub_block = depths[i*2:(i+1)*2, j*2:(j+1)*2]
                gt_splits_32.append(1 if np.mean(sub_block) > THRESHOLDS[1] else 0)
    else:
        gt_splits_32 = [0, 0, 0, 0]
    
    gt_splits_16 = []
    flat_depths = depths.flatten()
    for i in range(16):
        parent_32_index = (i // 8) * 2 + ((i % 4) // 2)
        if gt_split_64 and gt_splits_32[parent_32_index]:
            gt_splits_16.append(1 if flat_depths[i] > THRESHOLDS[2] else 0)
        else:
            gt_splits_16.append(0)

    return ctu_tensor, qp_tensor, gt_split_64, gt_splits_32, gt_splits_16

def get_predicted_splits(predictions, threshold=0.5):
    """Converts the model's 21-output tensor to binary split decisions."""
    pred_split_64 = 1 if predictions[0] > threshold else 0
    pred_splits_32 = [1 if p > threshold else 0 for p in predictions[1:5]]
    pred_splits_16 = [1 if p > threshold else 0 for p in predictions[5:21]]
    return pred_split_64, pred_splits_32, pred_splits_16

def calculate_accuracy_for_sample(gt_splits, pred_splits):
    """Calculates hierarchical accuracy for a single sample."""
    gt_64, gt_32, gt_16 = gt_splits
    pred_64, pred_32, pred_16 = pred_splits

    # L1 (64x64) Accuracy
    acc_64 = 100.0 if gt_64 == pred_64 else 0.0

    # L2 (32x32) Accuracy
    if gt_64 == 1:
        correct_32 = sum(1 for gt, pred in zip(gt_32, pred_32) if gt == pred)
        acc_32 = (correct_32 / 4.0) * 100
    else: # If GT is not split, prediction must also not split
        acc_32 = 100.0 if sum(pred_32) == 0 else 0.0

    # L3 (16x16) Accuracy
    num_valid_parents = sum(gt_32)
    if gt_64 == 1 and num_valid_parents > 0:
        correct_16, num_valid_16 = 0, 0
        for i in range(16):
            parent_32_index = (i // 8) * 2 + ((i % 4) // 2)
            if gt_32[parent_32_index] == 1:
                num_valid_16 += 1
                if gt_16[i] == pred_16[i]:
                    correct_16 += 1
        acc_16 = (correct_16 / num_valid_16) * 100 if num_valid_16 > 0 else 100.0
    else: # If parents are not split, prediction must also not split
        acc_16 = 100.0 if sum(pred_16) == 0 else 0.0
        
    return acc_64, acc_32, acc_16

# =================================================================================
# === Main Inference Execution ==================================================
# =================================================================================
def main():
    print("--- ViT Model Inference Script ---")
    print(f"Model: {MODEL_PATH}")
    print(f"Data File: {DATA_FILE}")
    print(f"QP Value: {QP_VALUE}")
    print(f"Samples to Test: {NUM_SAMPLES_TO_TEST}\n")

    # --- 1. Setup device and model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Instantiate the model with the same hyperparameters used for training
    model = VisionTransformer(
        embed_dim=196, depth=5, num_heads=4 
    ).to(device)

    # --- 2. Load the trained model weights ---
    print(f"Loading model weights from {MODEL_PATH}...")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.\n")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{MODEL_PATH}'. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return

    # --- 3. Run inference loop ---
    print("Starting inference loop...")
    total_acc_l1, total_acc_l2, total_acc_l3 = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for i in range(NUM_SAMPLES_TO_TEST):
            # Get a single sample with its ground truth splits for the chosen QP
            try:
                ctu_tensor, qp_tensor, gt_64, gt_32, gt_16 = get_sample_for_qp(DATA_FILE, i, QP_VALUE)
            except FileNotFoundError:
                print(f"ERROR: Data file not found at '{DATA_FILE}'. Please check the path.")
                return

            # Prepare inputs for the model (add batch and channel dimensions)
            inputs = ctu_tensor.to(device).unsqueeze(0).unsqueeze(0) # Shape: [1, 1, 64, 64]
            qp_input = qp_tensor.to(device).unsqueeze(0)             # Shape: [1]

            # Get model prediction
            predictions = model(inputs, qp_input).squeeze(0) # Shape: [21]
            pred_64, pred_32, pred_16 = get_predicted_splits(predictions)

            # Print comparison
            print(f"--- Sample {i + 1} ---")
            print(f"  GT 64->32 Split:  {gt_64} \t| Prediction: {pred_64}")
            print(f"  GT 32->16 Splits: {gt_32} \t| Prediction: {pred_32}")
            print(f"  GT 16->8 Splits:  {gt_16} \t| Prediction: {pred_16}")

            # Calculate and accumulate accuracy
            acc64, acc32, acc16 = calculate_accuracy_for_sample((gt_64, gt_32, gt_16), (pred_64, pred_32, pred_16))
            total_acc_l1 += acc64
            total_acc_l2 += acc32
            total_acc_l3 += acc16

    # --- 4. Report final results ---
    avg_acc_l1 = total_acc_l1 / NUM_SAMPLES_TO_TEST
    avg_acc_l2 = total_acc_l2 / NUM_SAMPLES_TO_TEST
    avg_acc_l3 = total_acc_l3 / NUM_SAMPLES_TO_TEST
    avg_total_acc = (avg_acc_l1 + avg_acc_l2 + avg_acc_l3) / 3

    print("\n--- Final Results ---")
    print(f"Total Samples Processed: {NUM_SAMPLES_TO_TEST}")
    print(f"Average Accuracy (Overall): {avg_total_acc:.2f}%")
    print(f"  - L1 (64x64) Accuracy: {avg_acc_l1:.2f}%")
    print(f"  - L2 (32x32) Accuracy: {avg_acc_l2:.2f}%")
    print(f"  - L3 (16x16) Accuracy: {avg_acc_l3:.2f}%")

if __name__ == '__main__':
    main()