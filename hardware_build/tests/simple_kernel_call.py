"""
Simple example: Send tensors to self-attention HLS kernel and get results back with timing

This demonstrates:
1. Preparing input tensors and weight matrices
2. Calling the self-attention HLS kernel
3. Measuring execution time
4. Getting results back
"""

import numpy as np
import time
import sys
from pathlib import Path

# Make project modules importable regardless of current working directory
root = Path(__file__).resolve().parents[2]
hardware_build_dir = root / "hardware_build"
cross_attention_dir = hardware_build_dir / "attention" / "cross_attention"
sys.path.insert(0, str(cross_attention_dir))
sys.path.insert(0, str(hardware_build_dir))

import allo
from allo.ir.types import float32
from attention.cross_attention import sdpa

# Try to import config
try:
    from attention.config import VLMAttentionConfig as VAC
except ImportError:
    # Fallback: define config locally
    class VAC:
        NUM_HEADS = 12
        HIDDEN_DIM = 768
        SINGLE_HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 64
        NUM_LAYERS = 12
        NUM_TOKENS = 1024


def run_kernel_with_timing():
    """
    Execute self-attention HLS kernel and measure timing.
    """
    
    print("=" * 70)
    print("Self-Attention HLS Kernel - Execution with Timing")
    print("=" * 70)
    
    # Get configuration
    L = VAC.NUM_TOKENS       # Sequence length (1024)
    D_h = VAC.SINGLE_HEAD_DIM # Head dimension (768 / 12 = 64)
    D_mlp = 3072             # MLP hidden dimension (typically 4*D_h)
    P = 4                     # Parallelism factor
    
    print(f"\n[CONFIG]")
    print(f"  Sequence length (L): {L}")
    print(f"  Head dimension (D_h): {D_h}")
    print(f"  MLP hidden dimension (D_mlp): {D_mlp}")
    print(f"  Parallelism factor (P): {P}")
    
    # Generate input tensors
    print(f"\n[INPUT PREPARATION]")
    start_prep = time.time()
    
    # Input token sequence: (L, D_h) - single head, not full embedding
    X = np.random.randn(L, D_h).astype(np.float32)
    
    # Projection weight matrices: (D_h, D_h) - single head dimension
    W_q = np.random.randn(D_h, D_h).astype(np.float32)
    W_k = np.random.randn(D_h, D_h).astype(np.float32)
    W_v = np.random.randn(D_h, D_h).astype(np.float32)
    
    # MLP weight matrices
    # FC1: D_h -> D_mlp (project up): weight shape (D_mlp, D_h)
    W_fc1 = np.random.randn(D_mlp, D_h).astype(np.float32)
    b_fc1 = np.random.randn(D_mlp).astype(np.float32)
    
    # FC2: D_mlp -> D_h (project down): weight shape (D_h, D_mlp)
    W_fc2 = np.random.randn(D_h, D_mlp).astype(np.float32)
    b_fc2 = np.random.randn(D_h).astype(np.float32)
    
    # Scale factor: 1 / sqrt(D_h)
    scale = float(1.0 / np.sqrt(D_h))
    
    # Output buffer: (L, D_h) - single head output
    out_Z = np.zeros((L, D_h), dtype=np.float32)
    
    prep_time = time.time() - start_prep
    
    print(f"  X shape: {X.shape}, dtype: {X.dtype}, size: {X.nbytes / 1024:.2f} KB")
    print(f"  W_q shape: {W_q.shape}, dtype: {W_q.dtype}, size: {W_q.nbytes / 1024:.2f} KB")
    print(f"  W_k shape: {W_k.shape}, dtype: {W_k.dtype}, size: {W_k.nbytes / 1024:.2f} KB")
    print(f"  W_v shape: {W_v.shape}, dtype: {W_v.dtype}, size: {W_v.nbytes / 1024:.2f} KB")
    print(f"  W_fc1 shape: {W_fc1.shape}, dtype: {W_fc1.dtype}, size: {W_fc1.nbytes / 1024:.2f} KB")
    print(f"  b_fc1 shape: {b_fc1.shape}, dtype: {b_fc1.dtype}, size: {b_fc1.nbytes / 1024:.2f} KB")
    print(f"  W_fc2 shape: {W_fc2.shape}, dtype: {W_fc2.dtype}, size: {W_fc2.nbytes / 1024:.2f} KB")
    print(f"  b_fc2 shape: {b_fc2.shape}, dtype: {b_fc2.dtype}, size: {b_fc2.nbytes / 1024:.2f} KB")
    print(f"  scale: {scale:.6f}")
    print(f"  out_Z shape: {out_Z.shape}, dtype: {out_Z.dtype}")
    print(f"  Preparation time: {prep_time*1000:.2f} ms")
    
    
    
    # Customize the kernel
    print(f"\n[KERNEL CUSTOMIZATION]")
    print(f"  Customizing self_attention_and_mlp kernel...")
    s = allo.customize(
        sdpa.self_attention_and_mlp,
        instantiate=[float32, L, D_h, D_mlp, P],
    )
    
    # First, generate HLS code
    print(f"\n[HLS CODE GENERATION]")
    print(f"  Generating HLS code...")
    start_codegen = time.time()
    hls_module = s.build(target="vhls")


    
    # Measure HLS codegen time
    codegen_time = time.time() - start_codegen
    print(f"  Codegen time: {codegen_time:.2f} s")

    print(f"\n[BUILD]")
    start_build = time.time()

    mod = s.build(
        target="vitis_hls",
        mode="sw_emu",
        project="self_attention_test.prj",
        configs={
            "device": "u280",    
            "frequency": 300,    
        },
    )
    build_time = time.time() - start_build
    print(f"  Build time: {build_time:.2f} s")

    # Execute kernel and time it
    print(f"\n[KERNEL EXECUTION]")
    # Make sure arrays are C-contiguous and correct dtype before calling Allo
    X_in = np.ascontiguousarray(X, dtype=np.float32)
    W_q_in = np.ascontiguousarray(W_q, dtype=np.float32)
    W_k_in = np.ascontiguousarray(W_k, dtype=np.float32)
    W_v_in = np.ascontiguousarray(W_v, dtype=np.float32)
    W_fc1_in = np.ascontiguousarray(W_fc1, dtype=np.float32)
    b_fc1_in = np.ascontiguousarray(b_fc1, dtype=np.float32)
    W_fc2_in = np.ascontiguousarray(W_fc2, dtype=np.float32)
    b_fc2_in = np.ascontiguousarray(b_fc2, dtype=np.float32)
    out_in = np.ascontiguousarray(out_Z, dtype=np.float32)

    start_kernel = time.time()
    mod(X_in, W_q_in, W_k_in, W_v_in, W_fc1_in, b_fc1_in, W_fc2_in, b_fc2_in, scale, out_in)
    kernel_time = time.time() - start_kernel
    
    print(f"  Kernel execution time: {kernel_time*1000:.2f} ms")
    
    print(f"\n[RESULTS]")
    print(f"  Output shape: {out_in.shape}")
    print(f"  Output dtype: {out_in.dtype}")
    print(f"  Output min: {out_in.min():.6f}")
    print(f"  Output max: {out_in.max():.6f}")
    print(f"  Output mean: {out_in.mean():.6f}")

    return {
        'success': True,
        'output': out_in,
        'hls_module': hls_module,
        'times': {
            'prep': prep_time,
            'codegen': codegen_time,
            'build': build_time,
            'kernel': kernel_time,
        }
    }



if __name__ == "__main__":
    result = run_kernel_with_timing()
    print("\n[SUMMARY]")
    print(f"Success: {result['success']}")
    print(f"Total times:")
    for key, val in result['times'].items():
        print(f"  {key}: {val:.4f}s")
