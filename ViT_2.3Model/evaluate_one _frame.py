import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import data_info as di # Uses the data_info.py file you provided

# ==============================================================================
# --- 1. Configuration: SET THESE VALUES ---
# ==============================================================================

# Path to the folder containing your original .yuv files
YUV_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/AI_YUV&Info/Info&YUV/AI_YUV'

# Path to the folder containing your Info_XX.dat files
INFO_PATH = '/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/AI_YUV&Info/Info&YUV/AI_Info'

# Name of the video file to evaluate (WITHOUT .yuv)
VIDEO_FILENAME = 'IntraValid_4928x3264'

# Frame number to extract from the video (0 is the first frame)
FRAME_TO_EVALUATE = 0

# QP to inspect the partitions for (e.g., 22, 27, 32, 37)
QP_VALUE = 32

# Path to the pre-trained ViT model
MODEL_PATH = '/root/myproject/HEVC_Intra_Models-ViT/ViT_2.3M/best_vit_model.pth'
# ==============================================================================


# ==============================================================================
# --- 2. Core Model and Data Logic (Copied from your files) ---
# ==============================================================================

# --- Data Reading Functions ---
class FrameYUV:
    def __init__(self, Y, U, V):
        self._Y = Y

def read_YUV420_frame(fid, width, height, frame_index):
    frame_bytes = (width * height * 3) // 2
    fid.seek(frame_index * frame_bytes)
    Y_buf = fid.read(width * height)
    if not Y_buf: return None
    Y = np.frombuffer(Y_buf, dtype=np.uint8).reshape([height, width])
    return FrameYUV(Y, None, None)

def read_info_frame(fid, width, height, frame_index):
    num_units_wide = width // 16
    num_units_high = height // 16
    frame_bytes = num_units_wide * num_units_high
    fid.seek(frame_index * frame_bytes)
    info_buf = fid.read(frame_bytes)
    if not info_buf: return None
    info = np.frombuffer(info_buf, dtype=np.uint8).reshape([num_units_high, num_units_wide])
    return info

# --- ViT Model Architecture ---
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=1, embed_dim=196):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
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

class VisionTransformer(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=1, num_classes=21, embed_dim=196, depth=5, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        dim_feedforward = int(embed_dim * mlp_ratio)
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
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

# --- Label Interpretation Functions ---
THRESHOLDS = [0.5, 1.5, 2.5]
def get_ground_truth_splits_from_ctu(ctu_label_map):
    depths = np.array(ctu_label_map).reshape(4, 4)
    split_64 = 1 if np.mean(depths) > THRESHOLDS[0] else 0
    splits_32 = []
    if split_64:
        for i in range(2):
            for j in range(2):
                sub_block = depths[i*2:(i+1)*2, j*2:(j+1)*2]
                splits_32.append(1 if np.mean(sub_block) > THRESHOLDS[1] else 0)
    else:
        splits_32 = [0, 0, 0, 0]
    splits_16 = []
    flat_depths = depths.flatten()
    for i in range(16):
        parent_32_index = (i // 8) * 2 + ((i % 4) // 2)
        if split_64 and splits_32[parent_32_index]:
            splits_16.append(1 if flat_depths[i] > THRESHOLDS[2] else 0)
        else:
            splits_16.append(0)
    return split_64, splits_32, splits_16

def get_predicted_splits(predictions_tensor, threshold=0.5):
    predictions = predictions_tensor.cpu().numpy() # Move to CPU and convert to NumPy
    pred_split_64 = 1 if predictions[0] > threshold else 0
    pred_splits_32 = [1 if p > threshold else 0 for p in predictions[1:5]]
    pred_splits_16 = [1 if p > threshold else 0 for p in predictions[5:21]]
    return pred_split_64, pred_splits_32, pred_splits_16


# ==============================================================================
# --- 3. Main Evaluation Logic ---
# ==============================================================================
def evaluate_single_frame_vit():
    # --- Find video info and file paths ---
    try:
        video_index = di.YUV_NAME_LIST_FULL.index(VIDEO_FILENAME)
        width = di.YUV_WIDTH_LIST_FULL[video_index]
        height = di.YUV_HEIGHT_LIST_FULL[video_index]
    except ValueError:
        print(f"Error: Video '{VIDEO_FILENAME}' not found in data_info.py.")
        return

    yuv_file_path = os.path.join(YUV_PATH, VIDEO_FILENAME + '.yuv')
    info_file_pattern = os.path.join(INFO_PATH, f'Info*_{VIDEO_FILENAME}_*qp{QP_VALUE}*CUDepth.dat')
    info_files = glob.glob(info_file_pattern)
    
    if not info_files:
        print(f"Error: Could not find Info file for QP {QP_VALUE}. Pattern: {info_file_pattern}")
        return
    info_file_path = info_files[0]
    
    print("--- Evaluating Single Frame with ViT Model ---")
    print(f"Video: {VIDEO_FILENAME}, Frame: {FRAME_TO_EVALUATE}, QP: {QP_VALUE}")
    print(f"Model: {MODEL_PATH}\n")

    # --- Read the full frame data ---
    try:
        with open(yuv_file_path, 'rb') as fid_yuv:
            frame_yuv = read_YUV420_frame(fid_yuv, width, height, FRAME_TO_EVALUATE)
        with open(info_file_path, 'rb') as fid_info:
            full_cu_depth_map = read_info_frame(fid_info, width, height, FRAME_TO_EVALUATE)
        if frame_yuv is None or full_cu_depth_map is None:
            print(f"Error: Frame #{FRAME_TO_EVALUATE} not found in the files.")
            return
    except FileNotFoundError as e:
        print(f"Error: File not found. Please check paths.\n{e}")
        return

    # --- Setup and load PyTorch model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    model = VisionTransformer(embed_dim=196, depth=5, num_heads=4).to(device)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print("ViT model restored successfully.\n")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{MODEL_PATH}'.")
        return
    except Exception as e:
        print(f"Error restoring ViT model: {e}")
        return
            
    all_gt_splits = []
    all_pred_splits = []
    
    with torch.no_grad():
        # --- Process frame CTU by CTU ---
        num_ctus_h = height // 64
        num_ctus_w = width // 64
        
        for y_ctu in range(num_ctus_h):
            for x_ctu in range(num_ctus_w):
                # 1. Get Ground Truth for this CTU
                ctu_label_map = full_cu_depth_map[y_ctu*4:(y_ctu+1)*4, x_ctu*4:(x_ctu+1)*4]
                gt_splits = get_ground_truth_splits_from_ctu(ctu_label_map)
                all_gt_splits.append(gt_splits)

                # 2. Get Model Prediction for this CTU
                ctu_image = frame_yuv._Y[y_ctu*64:(y_ctu+1)*64, x_ctu*64:(x_ctu+1)*64]
                
                # Convert to tensor, normalize, and add batch/channel dimensions
                ctu_tensor = torch.from_numpy(ctu_image.astype(np.float32)) / 255.0
                inputs = ctu_tensor.to(device).unsqueeze(0).unsqueeze(0) # Shape: [1, 1, 64, 64]
                
                qp_tensor = torch.tensor(float(QP_VALUE) / 51.0, dtype=torch.float32)
                qp_input = qp_tensor.to(device).unsqueeze(0) # Shape: [1]

                predictions = model(inputs, qp_input).squeeze(0)
                pred_splits = get_predicted_splits(predictions)
                all_pred_splits.append(pred_splits)

    # --- 4. Calculate and report accuracy for the whole frame ---
    total_ctus = len(all_gt_splits)
    correct_64 = 0
    correct_32, total_32 = 0, 0
    correct_16, total_16 = 0, 0

    for i in range(total_ctus):
        gt_64, gt_32, gt_16 = all_gt_splits[i]
        pred_64, pred_32, pred_16 = all_pred_splits[i]

        if gt_64 == pred_64: correct_64 += 1
        
        if gt_64 == 1:
            total_32 += 4
            correct_32 += sum(1 for gt, pred in zip(gt_32, pred_32) if gt == pred)

        for j in range(4):
            if gt_32[j] == 1:
                total_16 += 4
                start_idx, end_idx = j * 4, j * 4 + 4
                correct_16 += sum(1 for gt, pred in zip(gt_16[start_idx:end_idx], pred_16[start_idx:end_idx]) if gt == pred)
    
    acc_64 = (correct_64 / total_ctus) * 100 if total_ctus > 0 else 0
    acc_32 = (correct_32 / total_32) * 100 if total_32 > 0 else 100
    acc_16 = (correct_16 / total_16) * 100 if total_16 > 0 else 100
    
    print("--- Frame Accuracy Results ---")
    print(f"L1 (64x64 Split) Accuracy: {acc_64:.2f}% ({correct_64}/{total_ctus} correct CTUs)")
    print(f"L2 (32x32 Split) Accuracy: {acc_32:.2f}% ({correct_32}/{total_32} correct blocks)")
    print(f"L3 (16x16 Split) Accuracy: {acc_16:.2f}% ({correct_16}/{total_16} correct blocks)")
    overall_accuracy = (acc_64 + acc_32 + acc_16) / 3.0
    print("Overall (average of L1, L2, L3) Accuracy: {:.2f}%".format(overall_accuracy))

if __name__ == "__main__":
    if not os.path.exists("data_info.py"):
        print("ERROR: This script requires 'data_info.py' to be in the same directory.")
    else:
        evaluate_single_frame_vit()