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

    # Use the VisionTransformer class from the loaded module
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

        if len(image_patches.shape) == 4 and image_patches.shape[-1] == 1:
            input_tensor = torch.FloatTensor(image_patches).permute(0, 3, 1, 2)
        else:
            input_tensor = torch.FloatTensor(image_patches).unsqueeze(1)

        # --- THIS IS THE FIX ---
        # Normalize the image data to the [0, 1] range, just like in training.
        input_tensor = input_tensor / 255.0

        # Create QP tensor - normalized like in your training
        qp_tensor = torch.full((batch_size,), qp_value / 51.0, dtype=torch.float32)

        # Get predictions
        predictions = model(input_tensor, qp_tensor)

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
        input_data = np.load(args.input_file)
        model = load_vit_model(args.model_path)
        predictions = predict_cu_depth(model, input_data, args.qp)
        np.save(args.output_file, predictions)
        print(f"ViT predictions saved to {args.output_file}")

    except Exception as e:
        print(f"Error in ViT prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)