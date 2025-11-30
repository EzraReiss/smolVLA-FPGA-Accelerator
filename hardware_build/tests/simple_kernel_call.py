"""
Simple example: Send tensors to HLS kernel and get results back with timing

This is a straightforward script that demonstrates:
1. Preparing input tensors
2. Calling the HLS kernel
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
import cross_attention

# Try to import config
try:
    from attention.config import CrossAttentionConfig as CAC
except ImportError:
    # Fallback: define config locally
    class CAC:
        LENGTH_OF_ACTION_CHUNK = 50
        HEAD_DIM = 80
        DEFAULT_Tf = 241


def run_kernel_with_timing():
    """
    Execute cross-attention HLS kernel and measure timing.
    """
    
    print("=" * 70)
    print("Cross-Attention HLS Kernel - Execution with Timing")
    print("=" * 70)
    
    # Get configuration
    L_A = CAC.LENGTH_OF_ACTION_CHUNK
    H_D = CAC.HEAD_DIM
    L_V = CAC.DEFAULT_Tf
    
    print(f"\n[CONFIG]")
    print(f"  Action length (L_A): {L_A}")
    print(f"  Head dimension (H_D): {H_D}")
    print(f"  VLM length (L_V): {L_V}")
    
    # Generate input tensors
    print(f"\n[INPUT PREPARATION]")
    start_prep = time.time()
    
    Q = np.random.randn(L_A, H_D).astype(np.float32)
    K = np.random.randn(L_V, H_D).astype(np.float32)
    V = np.random.randn(L_V, H_D).astype(np.float32)
    # Use native Python float for scale (Allo's LLVM backend accepts int/float scalars)
    scale = float(np.sqrt(H_D))
    out_Z = np.zeros((L_A, H_D), dtype=np.float32)
    
    prep_time = time.time() - start_prep
    
    print(f"  Q shape: {Q.shape}, dtype: {Q.dtype}, size: {Q.nbytes / 1024:.2f} KB")
    print(f"  K shape: {K.shape}, dtype: {K.dtype}, size: {K.nbytes / 1024:.2f} KB")
    print(f"  V shape: {V.shape}, dtype: {V.dtype}, size: {V.nbytes / 1024:.2f} KB")

    print(f"  out_Z shape: {out_Z.shape}, dtype: {out_Z.dtype}")
    print(f"  Preparation time: {prep_time*1000:.2f} ms")
    
    
    
    # Customize the kernel
    print(f"  Customizing kernel...")
    s = allo.customize(
        cross_attention.cross_attention_fused,
        instantiate=[float32, L_A, H_D, L_V, L_V],
    )
    
    # First, generate HLS code
    print(f"\n[HLS CODE GENERATION]")
    print(f"  Generating HLS code...")
    start_codegen = time.time()
    hls_module = s.build(target="vhls")


    
    # Measure HLS codegen time
    codegen_time = time.time() - start_codegen

    start_build = time.time()

    mod = s.build(
    target="vitis_hls",
    mode="sw_emu",
    project="cross_attention_test.prj",
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
    Q_in = np.ascontiguousarray(Q, dtype=np.float32)
    K_in = np.ascontiguousarray(K, dtype=np.float32)
    V_in = np.ascontiguousarray(V, dtype=np.float32)
    out_in = np.ascontiguousarray(out_Z, dtype=np.float32)

    start_kernel = time.time()
    mod(Q_in, K_in, V_in, scale, out_in)
    kernel_time = time.time() - start_kernel

    return {
        'success': True,
        'output': out_Z,
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
    print(result)
