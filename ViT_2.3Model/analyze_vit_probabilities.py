# analyze_vit_probabilities.py
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Apne ViT model aur data reading functions ko yahan copy-paste karein ---
# (Maine neeche sirf zaroori parts daale hain, aap poora code daal sakte hain)
from video_to_cu_depth_vit import VisionTransformer, get_y_luma_from_frame
from evaluate_frame import read_info_frame, get_ground_truth_splits_from_ctu, di

def analyze_probabilities(video_path, info_path, model_path, qp):
    """Runs the ViT model and plots the probability distributions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Video aur Info file se details nikalein ---
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    video_index = di.YUV_NAME_LIST_FULL.index(video_filename)
    width = di.YUV_WIDTH_LIST_FULL[video_index]
    height = di.YUV_HEIGHT_LIST_FULL[video_index]

    # --- Model Load Karein ---
    model = VisionTransformer().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded.")

    # --- Probabilities jama karne ke liye lists ---
    probs_when_split = {0: [], 1: [], 2: []}  # Level 64, 32, 16
    probs_when_no_split = {0: [], 1: [], 2: []}

    with open(video_path, 'rb') as fid_yuv, open(info_path, 'rb') as fid_info:
        num_frames = os.path.getsize(video_path) // ((width * height * 3) // 2)
        for frame_idx in range(num_frames):
            print(f"\rProcessing frame {frame_idx + 1}/{num_frames}", end="")
            frame_yuv = get_y_luma_from_frame(fid_yuv, width, height)
            full_cu_depth_map = read_info_frame(fid_info, width, height, frame_idx)

            if frame_yuv is None: break

            with torch.no_grad():
                for y_ctu in range(height // 64):
                    for x_ctu in range(width // 64):
                        # Ground truth nikalein
                        ctu_label_map = full_cu_depth_map[y_ctu*4:(y_ctu+1)*4, x_ctu*4:(x_ctu+1)*4]
                        gt_64, gt_32, gt_16 = get_ground_truth_splits_from_ctu(ctu_label_map)

                        # Model se prediction lein
                        ctu_image = frame_yuv[y_ctu*64:(y_ctu+1)*64, x_ctu*64:(x_ctu+1)*64]
                        inputs = torch.from_numpy(ctu_image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device) / 255.0
                        qp_tensor = torch.tensor(float(qp) / 51.0, dtype=torch.float32).unsqueeze(0).to(device)
                        preds = model(inputs, qp_tensor).squeeze(0).cpu().numpy()

                        # Probabilities ko sahi lists mein daalein
                        if gt_64 == 1: probs_when_split[0].append(preds[0])
                        else: probs_when_no_split[0].append(preds[0])

                        if gt_64 == 1:
                            for i in range(4):
                                if gt_32[i] == 1: probs_when_split[1].append(preds[1+i])
                                else: probs_when_no_split[1].append(preds[1+i])

                        for i in range(16):
                            parent_32_idx = (i // 8) * 2 + ((i % 4) // 2)
                            if gt_64 == 1 and gt_32[parent_32_idx] == 1:
                                if gt_16[i] == 1: probs_when_split[2].append(preds[5+i])
                                else: probs_when_no_split[2].append(preds[5+i])
    print("\nAnalysis complete.")

    # --- Histograms Plot Karein ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    level_names = ['64x64 Split', '32x32 Split', '16x16 Split']
    for i in range(3):
        axes[i].hist(probs_when_no_split[i], bins=50, alpha=0.7, label='Ground Truth: No Split', color='blue')
        axes[i].hist(probs_when_split[i], bins=50, alpha=0.7, label='Ground Truth: Split', color='red')
        axes[i].set_title(level_names[i])
        axes[i].set_xlabel('Predicted Probability')
        axes[i].set_ylabel('Count')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig('vit_probability_distribution.png')
    print("Histogram saved to vit_probability_distribution.png")

# --- Yahan apne tuning video aur model ki details daalein ---
if __name__ == "__main__":
    TUNING_VIDEO_PATH = "/root/myproject/HEVC_Intra_Models-ViT/IntraValid_4928x3264.yuv"
    TUNING_INFO_PATH = "/root/myproject/HEVC-CNN/HEVC-Complexity-Reduction/Info&YUV/AI_Info/Info_20170811_001556_AI_IntraTest_4928x3264_qp32_nf50_CUDepth.dat"
    MODEL_PATH = "best_vit_model.pth"
    QP_VALUE = 32 # Ek representative QP value chunein
    
    analyze_probabilities(TUNING_VIDEO_PATH, TUNING_INFO_PATH, MODEL_PATH, QP_VALUE)