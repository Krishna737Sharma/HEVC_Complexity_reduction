import sys
import numpy as np
import torch
import argparse
import os


def load_vit_model(model_path):
    """Load the trained ViT model"""
    # Use absolute path to your ViT model
    vit_model_path = "/home/ai-iitkgp/PycharmProjects/HEVC_Intra_Models-ViT/Vit_model.py"

    # Load the module using importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("Vit_model", vit_model_path)
    vit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vit_module)

    # Use the VisionTransformer class
    model = vit_module.VisionTransformer(
        img_size=64,
        patch_size=8,
        in_chans=1,
        num_classes=21,
        embed_dim=196,
        depth=5,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1
    )

    # Load the trained weights
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_cu_depth(model, image_patches, qp_value):
    """Predict CU depth using ViT model"""
    with torch.no_grad():
        batch_size = image_patches.shape[0]

        # Ensure correct input shape [B, H, W, C] -> [B, C, H, W]
        if len(image_patches.shape) == 4 and image_patches.shape[-1] == 1:
            # From [B, H, W, C] to [B, C, H, W]
            input_tensor = torch.FloatTensor(image_patches).permute(0, 3, 1, 2)
        else:
            # Assume [B, H, W] and add channel dimension
            input_tensor = torch.FloatTensor(image_patches).unsqueeze(1)

        # Normalize like in your training (already done in your model, but ensure consistency)
        # Your model does: x /= 255.0, so input should be 0-255 range

        # Create QP tensor - normalized like in your training
        qp_tensor = torch.full((batch_size,), qp_value / 51.0, dtype=torch.float32)

        # Get predictions - your model outputs sigmoid probabilities
        predictions = model(input_tensor, qp_tensor)  # [B, 21]

        # Convert back to numpy
        output = predictions.cpu().numpy()

        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--qp', type=int, default=32)

    args = parser.parse_args()

    try:
        # Load input data
        input_data = np.load(args.input_file)
        print(f"Loaded input data shape: {input_data.shape}")

        # Load model and predict
        model = load_vit_model(args.model_path)
        predictions = predict_cu_depth(model, input_data, args.qp)

        # Save results
        np.save(args.output_file, predictions)
        print(f"ViT predictions saved to {args.output_file}")
        print(f"Predictions shape: {predictions.shape}")

    except Exception as e:
        print(f"Error in ViT prediction: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
