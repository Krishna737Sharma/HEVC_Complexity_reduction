import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random
import os
import wandb
import matplotlib.pyplot as plt

# Global debug flag
DEBUG = False

# ==================== Constants ====================
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_LABEL_BYTES = 16  # original label (4x4 = 16)
NUM_SAMPLE_LENGTH = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + 64 + (51 + 1) * NUM_LABEL_BYTES
SELECT_QP_LIST = [22, 27, 32, 37]

if DEBUG:
    print("DEBUG: IMAGE_SIZE =", IMAGE_SIZE)
    print("DEBUG: NUM_CHANNELS =", NUM_CHANNELS)
    print("DEBUG: NUM_LABEL_BYTES =", NUM_LABEL_BYTES)
    print("DEBUG: NUM_SAMPLE_LENGTH =", NUM_SAMPLE_LENGTH)
    print("DEBUG: SELECT_QP_LIST =", SELECT_QP_LIST)


# ==================== StreamingDataset Class ====================
class StreamingDataset(Dataset):
    def __init__(self, file_path, max_samples):
        if DEBUG:
            print(f"DEBUG: Initializing StreamingDataset with file: {file_path} and max_samples: {max_samples}")
        self.file_path = file_path
        self.max_samples = max_samples

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        if DEBUG and idx == 0:
            print(f"\nDEBUG: __getitem__ called for sample index {idx}")
        with open(self.file_path, 'rb') as file_reader:
            offset = idx * NUM_SAMPLE_LENGTH
            file_reader.seek(offset)
            data = np.frombuffer(file_reader.read(NUM_SAMPLE_LENGTH), dtype=np.uint8)

            if DEBUG and idx == 0:
                print(f"DEBUG: Read data shape: {data.shape} (Expected: ({NUM_SAMPLE_LENGTH},))")

            # Process image
            image = data[:4096].astype(np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
            if DEBUG and idx == 0:
                print(f"DEBUG: Processed image shape: {image.shape} (Expected: (64, 64, 1))")

            # Process QP (randomly select from list)
            qp = np.random.choice(SELECT_QP_LIST, size=1)[0]
            if DEBUG and idx == 0:
                print(f"DEBUG: Selected QP: {qp} (Expected one of {SELECT_QP_LIST})")

            # Process label using QP index
            label = np.zeros((NUM_LABEL_BYTES,))
            qp_index = int(qp)
            label[:] = data[4160 + qp_index * NUM_LABEL_BYTES: 4160 + (qp_index + 1) * NUM_LABEL_BYTES]
            if DEBUG and idx == 0:
                print(f"DEBUG: Processed label values: {label} (Expected 16 values)")

            # Convert image and QP to tensors
            ctu_tensor = torch.tensor(image, dtype=torch.float32).squeeze(2)
            qp_tensor = torch.tensor(float(qp), dtype=torch.float32)
            if DEBUG and idx == 0:
                print(f"DEBUG: ctu_tensor shape: {ctu_tensor.shape} (Expected: (64, 64))")
                print(f"DEBUG: qp_tensor (before scaling): {qp_tensor.item()}")

            # Scale tensors
            ctu_tensor /= 255.0
            qp_tensor /= 51.0
            if DEBUG and idx == 0:
                print(f"DEBUG: ctu_tensor normalized (max value should be <=1)")
                print(f"DEBUG: qp_tensor normalized: {qp_tensor.item()} (Expected: qp/51)")

            # Convert label to hierarchical output (reshape to 1x4x4)
            y_image = torch.tensor(label, dtype=torch.float32).view(1, 4, 4)
            if DEBUG and idx == 0:
                print(f"DEBUG: y_image shape: {y_image.shape} (Expected: (1, 4, 4))")

            # Hierarchical pooling/activation steps
            y_image_16 = F.relu(y_image - 2)
            y_image_32 = F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 1) - \
                         F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 2)
            y_image_64 = F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=4) - 0) - \
                         F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=4) - 1)
            y_image_valid_32 = F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 0) - \
                               F.relu(F.avg_pool2d(y_image.permute(0, 2, 1), kernel_size=2) - 1)
            y_image_valid_16 = F.relu(y_image - 1) - F.relu(y_image - 2)
            if DEBUG and idx == 0:
                print("DEBUG: Completed hierarchical pooling/activation")

            # Flatten the hierarchical outputs
            y_flat_16 = y_image_16.view(-1)
            y_flat_32 = y_image_32.view(-1)
            y_flat_64 = y_image_64.view(-1)
            y_flat_valid_32 = y_image_valid_32.view(-1)
            y_flat_valid_16 = y_image_valid_16.view(-1)
            if DEBUG and idx == 0:
                print(f"DEBUG: y_flat_64 length: {y_flat_64.shape[0]} (Expected: 1)")
                print(f"DEBUG: y_flat_32 length: {y_flat_32.shape[0]} (Expected: 4)")
                print(f"DEBUG: y_flat_16 length: {y_flat_16.shape[0]} (Expected: 16)")

            # Concatenate hierarchical outputs into one target vector (21-dim)
            target = torch.cat((y_flat_64, y_flat_32, y_flat_16), dim=0)
            if DEBUG and idx == 0:
                print(f"DEBUG: Final target shape: {target.shape} (Expected: (21,))")

            return qp_tensor, ctu_tensor, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, target


def create_subset_dataloader(file_path, total_samples, subset_size, batch_size, shuffle=True):
    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    full_dataset = StreamingDataset(file_path, total_samples)
    subset_indices = random.sample(range(total_samples), subset_size)
    if DEBUG:
        print(f"DEBUG: Creating DataLoader with subset size: {subset_size} and batch_size: {batch_size}")
        print(f"DEBUG: Subset indices sample: {subset_indices[:5]} ...")
    return DataLoader(
        Subset(full_dataset, subset_indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    ), subset_indices


# ==================== PatchEmbed Class ====================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=1, embed_dim=196):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if DEBUG:
            print(f"DEBUG: Initialized PatchEmbed with {self.num_patches} patches and embed_dim: {embed_dim}")

    def forward(self, x):
        if DEBUG:
            print(
                f"DEBUG: PatchEmbed input shape: {x.shape} (Expected: [B, {NUM_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE}])")
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        if DEBUG:
            print(f"DEBUG: After conv: {x.shape}")
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        if DEBUG:
            print(f"DEBUG: After flatten & transpose: {x.shape} (Expected: [B, {self.num_patches}, embed_dim])")
        return x


# ==================== Custom Transformer Encoder Layer with QP Integration ====================
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # MLP layers with extra QP input:
        self.linear1 = nn.Linear(d_model + 1, dim_feedforward)  # Input: d_model+1 (e.g. 256+1)
        self.linear2 = nn.Linear(dim_feedforward + 1, d_model)  # Input: dim_feedforward+1 (e.g. 1024+1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, src, qp):
        """
        src: [seq_len, B, d_model]
        qp: [B] scalar per sample, broadcasted to [seq_len, B, 1]
        """
        # Self-attention
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        B = src.shape[1]
        seq_len = src.shape[0]
        qp_exp = qp.view(1, B, 1).expand(seq_len, B, 1)

        # First linear layer with QP concatenation
        src_cat = torch.cat([src, qp_exp], dim=-1)  # [seq_len, B, d_model+1]
        src2 = self.linear1(src_cat)  # [seq_len, B, dim_feedforward]
        src2 = self.activation(src2)
        src2 = self.dropout1(src2)

        # Second linear layer with QP concatenation
        qp_exp2 = qp.view(1, B, 1).expand(seq_len, B, 1)
        src2_cat = torch.cat([src2, qp_exp2], dim=-1)  # [seq_len, B, dim_feedforward+1]
        src2 = self.linear2(src2_cat)  # [seq_len, B, d_model]
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# ==================== Vision Transformer Model with QP Integration ====================
class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=64,
            patch_size=8,
            in_chans=1,
            num_classes=21,  # 21 binary outputs
            embed_dim=196,  # Adjust as desired (256 or 512)
            depth=5,
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        dim_feedforward = int(embed_dim * mlp_ratio)  # e.g. 256*4 = 1024
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
                                          dropout=dropout, activation="gelu")
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()
        if DEBUG:
            print(
                f"DEBUG: Initialized VisionTransformer with embed_dim: {embed_dim}, depth: {depth}, num_heads: {num_heads}")

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
        if DEBUG:
            print("DEBUG: Weights initialized for VisionTransformer")

    def forward(self, x, qp):
        """
        x: [B, 1, 64, 64] image tensor
        qp: [B] QP scalar per sample
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        if DEBUG:
            print(f"DEBUG: After patch embedding: {x.shape}")
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Transformer expects [seq_len, B, embed_dim]
        x = x.transpose(0, 1)
        for layer in self.encoder_layers:
            x = layer(x, qp)
        x = x[0]  # Use the CLS token
        x = self.norm(x)
        logits = self.head(x)
        out = torch.sigmoid(logits)
        return out


# ==================== Accuracy Calculation Function ====================
def calculate_accuracy_repo(y_flat_64, y_conv_flat_64,
                            y_flat_32, y_conv_flat_32, y_flat_valid_32,
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
    if DEBUG:
        print("DEBUG: Accuracy per branch:")
        print(f"       64-part: {accuracy_64.item():.2f}%")
        print(f"       32-part: {accuracy_32.item():.2f}%")
        print(f"       16-part: {accuracy_16.item():.2f}%")
        print(f"       Average: {avg_acc.item():.2f}%")
    return avg_acc, accuracy_64, accuracy_32, accuracy_16


# ==================== TRAINING CODE GUARD ====================
# This prevents training code from running when importing the module
if __name__ == "__main__":
    # ==================== File Paths and DataLoader Creation ====================
    train_file_path = "/home/m1/23CS60R16/Data/AI_Train_1668975.dat_shuffled"
    validation_file_path = "/home/m1/23CS60R16/Data/AI_Valid_98175.dat_shuffled"
    TRAINSET_MAXSIZE = 1668975
    VALIDSET_MAXSIZE = 98175
    BATCH_SIZE = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEBUG:
        print(f"DEBUG: Using device: {device}")

    train_loader, train_indices = create_subset_dataloader(train_file_path, TRAINSET_MAXSIZE, 80000, BATCH_SIZE,
                                                           shuffle=True)
    validation_loader, validation_indices = create_subset_dataloader(validation_file_path, VALIDSET_MAXSIZE, 60000,
                                                                     BATCH_SIZE, shuffle=False)

    # ==================== Training Setup ====================
    wandb.login(key="555f8dba02bbffeb502ffabe1112ca7f78a019b6")
    wandb.init(
        project="vit-training",
        config={
            "learning_rate": 0.01,
            "optimizer": "SGD",
            "momentum": 0.9,
            "epochs": 10000,
            "architecture": "VisionTransformer",
            "batch_size": BATCH_SIZE,
            "patch_size": 8,
            "embed_dim": 196,  # Using embed_dim 256 for this example (can be set to 512)
            "depth": 5,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "dropout": 0.1
        }
    )

    # Instantiate the model. Note: The forward method now requires both image and qp.
    model = VisionTransformer(
        img_size=IMAGE_SIZE,
        patch_size=wandb.config.patch_size,
        in_chans=NUM_CHANNELS,
        num_classes=21,
        embed_dim=wandb.config.embed_dim,
        depth=wandb.config.depth,
        num_heads=wandb.config.num_heads,
        mlp_ratio=wandb.config.mlp_ratio,
        dropout=wandb.config.dropout
    ).to(device)


    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")


    count_parameters(model)

    optimizer = optim.SGD(model.parameters(), lr=wandb.config.learning_rate, momentum=wandb.config.momentum)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.3163)
    criterion = nn.BCELoss()  # (Not used with custom loss in this code; kept for reference)

    # ==================== Training Loop ====================
    num_epochs = wandb.config.epochs
    best_loss = float('inf')
    overall_least_loss = float('inf')
    patience = 50
    patience_counter = 0
    num_patience_counter_changed = 0
    start_epoch = 0

    # Load saved model
    checkpoint_path = 'best_vit_model.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # Restore model, optimizer, scheduler, and epoch
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        overall_least_loss = checkpoint['overall_least_loss']

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_acc_l1 = 0.0
        running_acc_l2 = 0.0
        running_acc_l3 = 0.0

        if DEBUG:
            print(f"\nDEBUG: Starting epoch {epoch + 1}/{num_epochs}")

        # Change dataset after `epochs_per_dataset_change` epochs
        if num_patience_counter_changed >= 5:
            num_patience_counter_changed = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Changing validation datasets...")
            new_validation_loader, new_validation_indices = create_subset_dataloader(validation_file_path,
                                                                                     VALIDSET_MAXSIZE, 60000,
                                                                                     BATCH_SIZE, shuffle=False)
            # Log dataset changes
            print("Validation indices changed:", new_validation_indices != validation_indices)
            # Update loaders and indices
            validation_loader, validation_indices = new_validation_loader, new_validation_indices

        for batch_idx, batch in enumerate(train_loader):
            # Each batch returns: qp_tensor, ctu_tensor, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, target
            qp_batch, ctu_batch, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, target = batch
            inputs = ctu_batch.to(device).unsqueeze(1)  # [B, 1, 64, 64]
            qp_tensor = qp_batch.to(device)  # [B]
            target = target.to(device)  # [B, 21]

            optimizer.zero_grad()
            outputs = model(inputs, qp_tensor)  # [B, 21]
            loss = criterion(outputs, target)  # You can swap to your custom loss as needed.
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            avg_acc, acc64, acc32, acc16 = calculate_accuracy_repo(
                y_flat_64, outputs[:, 0:1],
                y_flat_32, outputs[:, 1:5], y_flat_valid_32,
                y_flat_16, outputs[:, 5:21], y_flat_valid_16
            )
            running_acc += avg_acc.item()
            running_acc_l1 += acc64.item()
            running_acc_l2 += acc32.item()
            running_acc_l3 += acc16.item()

            if DEBUG and batch_idx == 0:
                print(f"DEBUG: Batch {batch_idx} - Loss: {loss.item():.4f}, Avg Acc: {avg_acc.item():.2f}%")

        avg_loss = running_loss / len(train_loader)
        avg_acc = running_acc / len(train_loader)
        avg_acc_l1 = running_acc_l1 / len(train_loader)
        avg_acc_l2 = running_acc_l2 / len(train_loader)
        avg_acc_l3 = running_acc_l3 / len(train_loader)

        wandb.log(
            {"epoch": epoch + 1, "train_loss": avg_loss, "train_accuracy": avg_acc, "train_accuracy_L1": avg_acc_l1,
             "train_accuracy_L2": avg_acc_l2, "train_accuracy_L3": avg_acc_l3})
        print(f"Epoch [{epoch + 1}/{num_epochs}] Training Loss: {avg_loss:.4f} "
              f"Accuracy: {avg_acc:.2f}% | L1: {avg_acc_l1:.2f}% | L2: {avg_acc_l2:.2f}% | L3: {avg_acc_l3:.2f}%")

        if (epoch + 1) % 100 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_loss': best_loss,
                'overall_least_loss': overall_least_loss
            }, 'best_vit_model_every_100_epoch.pth')
            print(f"Checkpoint 'best_vit_model_every_100_epoch' saved at epoch {epoch + 1}")

        # Validation every 2 epochs for this example
        if (epoch + 1) % 2 == 0:
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_acc_l1 = 0.0
            val_acc_l2 = 0.0
            val_acc_l3 = 0.0
            with torch.no_grad():
                for val_batch in validation_loader:
                    qp_batch, ctu_batch, y_flat_64, y_flat_32, y_flat_16, y_flat_valid_32, y_flat_valid_16, target = val_batch
                    inputs = ctu_batch.to(device).unsqueeze(1)
                    qp_tensor = qp_batch.to(device)
                    target = target.to(device)
                    outputs = model(inputs, qp_tensor)
                    loss = criterion(outputs, target)
                    val_loss += loss.item()
                    avg_acc, acc64, acc32, acc16 = calculate_accuracy_repo(
                        y_flat_64, outputs[:, 0:1],
                        y_flat_32, outputs[:, 1:5], y_flat_valid_32,
                        y_flat_16, outputs[:, 5:21], y_flat_valid_16
                    )
                    val_acc += avg_acc.item()
                    val_acc_l1 += acc64.item()
                    val_acc_l2 += acc32.item()
                    val_acc_l3 += acc16.item()

            avg_val_loss = val_loss / len(validation_loader)
            avg_val_acc = val_acc / len(validation_loader)
            avg_val_acc_l1 = val_acc_l1 / len(validation_loader)
            avg_val_acc_l2 = val_acc_l2 / len(validation_loader)
            avg_val_acc_l3 = val_acc_l3 / len(validation_loader)

            wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss, "val_accuracy": avg_val_acc,
                       "val_accuracy_L1": avg_val_acc_l1, "val_accuracy_L2": avg_val_acc_l2,
                       "val_accuracy_L3": avg_val_acc_l3})
            print(
                f"Validation Loss: {avg_val_loss:.4f} \n Accuracy: {avg_val_acc:.2f}% | L1: {avg_val_acc_l1:.2f}% | L2: {avg_val_acc_l2:.2f}% | L3: {avg_val_acc_l3:.2f}%")

            if avg_val_loss <= best_loss:
                best_loss = avg_val_loss
                overall_least_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'best_loss': best_loss,
                    'overall_least_loss': overall_least_loss
                }, 'best_vit_model.pth')
                print(f"Checkpoint saved at epoch {epoch + 1}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    patience_counter = 0
                    num_patience_counter_changed += 1
                    # Load best model till time and change training dataset
                    checkpoint_path = 'best_vit_model.pth'
                    if os.path.exists(checkpoint_path):
                        print(f"Loading checkpoint from {checkpoint_path}...")
                        # Load checkpoint
                        checkpoint = torch.load(checkpoint_path, weights_only=False)

                        # Restore model, optimizer, scheduler, and epoch
                        model.load_state_dict(checkpoint['model_state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        start_epoch = checkpoint['epoch']
                        best_loss = checkpoint['best_loss']
                        overall_least_loss = checkpoint['overall_least_loss']

                    print("DEBUG: No improvement for several epochs. Consider early stopping or adjusting training.")

                    best_loss = float('inf')
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    print("Changing training datasets...")
                    new_train_loader, new_train_indices = create_subset_dataloader(train_file_path, TRAINSET_MAXSIZE,
                                                                                   80000, BATCH_SIZE, shuffle=True)
                    # Log dataset changes
                    print("Train indices changed:", new_train_indices != train_indices)

                    # Update loaders and indices
                    train_loader, train_indices = new_train_loader, new_train_indices

        if (epoch + 1) % 10000 == 0:
            scheduler.step()

# (Optional) Additional plotting or analysis code can be added below.
