import pandas as pd
import bjontegaard as bd   # pip install bjontegaard

# Read CSV
data = pd.read_csv('/root/myproject/HEVC_Intra_Models-ViT/inference_sheet_updated.csv')

# YUV samples of interest
samples = [
    'IntraValid_4928x3264.yuv (25f)',
    'Rush_Hour.yuv_3840x2160 (250f)',
    'Netflix_FoodMarket2_4096x2160.yuv (150f)',
    'Scarf.yuv_3840x2160 (100f)',
    'Construction_Field.yuv (7f)',
]

def hms_to_sec(hms):
    mins, sec = hms.split('m')
    mins = int(mins.strip())
    sec = float(sec.replace('s', '').strip())
    return mins * 60 + sec

bd_rate_results = []
time_results = []

for yuv in samples:
    qp_sel = [22, 27, 32, 37]
    base = data[(data['Method'] == 'HEVC') & (data['YUV file'] == yuv) & (data['QP'].isin(qp_sel))]
    cnn  = data[(data['Method'] == 'CNN')  & (data['YUV file'] == yuv) & (data['QP'].isin(qp_sel))]
    vit  = data[(data['Method'] == 'VIT')  & (data['YUV file'] == yuv) & (data['QP'].isin(qp_sel))]

    # --- PRINT extracted data to verify correctness ---
    print(f"\n===== DATA for {yuv} =====")
    print("HEVC (QP, Bitrate, PSNR, Time):")
    print(base[['QP', 'Bitrate (kbps)', 'YUV-PSNR (dB)', 'Time (User+Sys) (s)']])
    print("CNN (QP, Bitrate, PSNR, Time):")
    print(cnn[['QP', 'Bitrate (kbps)', 'YUV-PSNR (dB)', 'Time (User+Sys) (s)']])
    print("ViT (QP, Bitrate, PSNR, Time):")
    print(vit[['QP', 'Bitrate (kbps)', 'YUV-PSNR (dB)', 'Time (User+Sys) (s)']])

    # BD-rate: ViT vs HEVC
    bd_vit = bd.bd_rate(
        base['Bitrate (kbps)'].tolist(),
        base['YUV-PSNR (dB)'].tolist(),
        vit['Bitrate (kbps)'].tolist(),
        vit['YUV-PSNR (dB)'].tolist(),
        method='akima'
    )
    # BD-rate: CNN vs HEVC
    bd_cnn = bd.bd_rate(
        base['Bitrate (kbps)'].tolist(),
        base['YUV-PSNR (dB)'].tolist(),
        cnn['Bitrate (kbps)'].tolist(),
        cnn['YUV-PSNR (dB)'].tolist(),
        method='akima'
    )
    bd_rate_results.append([yuv, bd_cnn, bd_vit])

    # Time: sum over 4 QPs (use 'Time (User+Sys) (s)')
    base_time = base['Time (User+Sys) (s)'].map(hms_to_sec).sum()
    cnn_time = cnn['Time (User+Sys) (s)'].map(hms_to_sec).sum()
    vit_time = vit['Time (User+Sys) (s)'].map(hms_to_sec).sum()
    cnn_time_saving = 100. * (base_time - cnn_time) / base_time
    vit_time_saving = 100. * (base_time - vit_time) / base_time
    time_results.append([yuv, cnn_time_saving, vit_time_saving])
# Results DataFrames
df_bd = pd.DataFrame(bd_rate_results, columns=['YUV', 'CNN_vs_HEVC_BD-Rate(%)', 'ViT_vs_HEVC_BD-Rate(%)'])
df_time = pd.DataFrame(time_results, columns=['YUV', 'CNN_time_saving(%)', 'ViT_time_saving(%)'])

print("\n=== BD-Rate Results ===")
print(df_bd)
print("\n=== Time Savings Results ===")
print(df_time)

# Optionally, save to CSV
df_bd.to_csv('bd_rate_results.csv', index=False)
df_time.to_csv('time_savings_results.csv', index=False)
