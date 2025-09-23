import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
data = pd.read_csv('/root/myproject/HEVC_Intra_Models-ViT/inference_sheet_updated.csv')

# List YUV samples of interest
samples = [
    'IntraValid_4928x3264.yuv (25f)',
    'Rush_Hour.yuv_3840x2160 (250f)',
    'Netflix_FoodMarket2_4096x2160.yuv (150f)',
    'Scarf.yuv_3840x2160 (100f)',
    'Construction_Field.yuv (7f)',
]

# For each YUV file, plot the RD curves for all methods
for yuv in samples:
    plt.figure(figsize=(7, 5))
    for method in ['HEVC', 'CNN', 'VIT']:
        # Filter rows
        sub = data[(data['Method'] == method) & (data['YUV file'] == yuv)]
        # Sort by bitrate for smooth plot
        sub = sub.sort_values('Bitrate (kbps)')
        plt.plot(
            sub['Bitrate (kbps)'],
            sub['YUV-PSNR (dB)'],
            marker='o',
            label=method
        )
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('YUV-PSNR (dB)')
    plt.title(f'RD Curve for {yuv}')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')  # Log scale is standard for RD plots
    plt.tight_layout()
    # Generate safe filename
    safe_name = yuv.replace(" ", "_").replace(".yuv", "").replace("(", "").replace(")", "").replace("/", "")
    plt.savefig(f'rd_curve_{safe_name}.png')
    plt.show()
