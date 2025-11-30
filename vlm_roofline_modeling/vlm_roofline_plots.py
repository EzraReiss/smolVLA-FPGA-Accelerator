import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'lines.linewidth': 2
})

FREQ = 300e6
BW_PEAK = 460e9
BW_REAL = 300e9

# Peak Compute (Ops/s)
P_FP32 = 5.41e12
P_BF16 = 5.41e12 
P_INT8 = 18.6e12
P_INT4 = 37.2e12

# Dimensions
V_LAYERS = 12
V_D = 768
V_FFN = 3072
V_ATTN_D = 768

T_LAYERS = 16
T_D = 960
T_FFN = 2560
T_Q_D = 960
T_K_D = 320
T_V_D = 320
T_OUT_D = 960

CONN_IN = 12288
CONN_OUT = 960

B = 1
S = 50 
S_V = 1024 # Updated to 1024 patches

precisions = {
    'FP32': 4,
    'BF16': 2,
    'INT8': 1,
    'INT4': 0.5
}

def calc_oi_linear(M, K, N, dtype_bytes):
    flops = 2 * M * K * N
    bytes_xfer = (K*N + M*K + M*N) * dtype_bytes
    return flops / bytes_xfer

def analyze_kernel(name, M, K, N, p_bytes):
    flops = 2 * M * K * N
    mem_weights = K * N * p_bytes
    mem_io = (M * K + M * N) * p_bytes
    total_bytes = mem_weights + mem_io
    oi = flops / total_bytes
    return {
        'Kernel': name,
        'OI': oi
    }

def plot_roofline_base(ax, title):
    x = np.logspace(-2, 3, 100)
    ceilings = [
        ('FP32/BF16', P_FP32, 'k-'),
        ('INT8', P_INT8, 'b-'),
        ('INT4', P_INT4, 'g-')
    ]
    y_mem = BW_REAL * x
    ax.loglog(x, y_mem, 'r--', label='Memory Wall (300 GB/s)')
    for name, peak, style in ceilings:
        y = np.minimum(peak, y_mem)
        ax.loglog(x, y, style, label=f'{name} Peak')
    ax.set_xlabel('Operational Intensity (Ops/Byte)')
    ax.set_ylabel('Performance (Ops/s)')
    ax.set_title(title)
    ax.grid(True, which="both", ls="-", alpha=0.5)

def plot_kernels_improved(ax, metrics_list, peak_perf):
    for i, item in enumerate(metrics_list):
        oi = item['OI']
        perf = min(peak_perf, BW_REAL * oi)
        
        ax.plot(oi, perf, 'b^', markersize=12)
        
        # Alternating offset to avoid overlap
        offset = 1.4 if i % 2 == 0 else 0.6
        va = 'bottom' if offset > 1 else 'top'
        
        ax.text(oi, perf * offset, item['Kernel'], fontsize=9, ha='center', va=va)

# --- Vision Encoder ---
p_bytes = precisions['INT8']
vision_kernels = []
vision_kernels.append(analyze_kernel('PatchEmbed', S_V, 3*16*16, V_D, p_bytes))
vision_kernels.append(analyze_kernel('Attn_Q', S_V, V_D, V_D, p_bytes))
vision_kernels.append(analyze_kernel('Attn_K', S_V, V_D, V_D, p_bytes))
vision_kernels.append(analyze_kernel('Attn_V', S_V, V_D, V_D, p_bytes))
vision_kernels.append(analyze_kernel('Attn_Out', S_V, V_D, V_D, p_bytes))
vision_kernels.append(analyze_kernel('MLP_FC1', S_V, V_D, V_FFN, p_bytes))
vision_kernels.append(analyze_kernel('MLP_FC2', S_V, V_FFN, V_D, p_bytes))
vision_kernels.append(analyze_kernel('Connector', 1, 12288, 960, p_bytes))

fig, ax = plt.subplots(figsize=(12, 8))
plot_roofline_base(ax, 'Vision Encoder Roofline (INT8)')
plot_kernels_improved(ax, vision_kernels, P_INT8)
ax.legend()
plt.tight_layout()
plt.savefig('vision_roofline.png')
print("Saved vision_roofline.png")

# --- Text Encoder ---
text_kernels = []
text_kernels.append(analyze_kernel('Attn_Q', S, T_D, T_Q_D, p_bytes))
text_kernels.append(analyze_kernel('Attn_K', S, T_D, T_K_D, p_bytes))
text_kernels.append(analyze_kernel('Attn_V', S, T_D, T_V_D, p_bytes))
text_kernels.append(analyze_kernel('Attn_Out', S, T_D, T_OUT_D, p_bytes))
text_kernels.append(analyze_kernel('MLP_Gate', S, T_D, T_FFN, p_bytes))
text_kernels.append(analyze_kernel('MLP_Up', S, T_D, T_FFN, p_bytes))
text_kernels.append(analyze_kernel('MLP_Down', S, T_FFN, T_D, p_bytes))
text_kernels.append(analyze_kernel('LM_Head', 1, T_D, 49280, p_bytes))

fig, ax = plt.subplots(figsize=(12, 8))
plot_roofline_base(ax, 'Text Encoder Roofline (INT8)')
plot_kernels_improved(ax, text_kernels, P_INT8)
ax.legend()
plt.tight_layout()
plt.savefig('text_roofline.png')
print("Saved text_roofline.png")
