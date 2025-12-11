# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo
from allo.ir.types import int8, int16, int32, float32, Stream, UInt
from allo.utils import get_np_struct_type
import allo.dataflow as df
import allo.backend.hls as hls
import allo.dsl as dsl
import numpy as np

# MLP dimensions from mlp.py
M = 1024           # batch size
D_in = 768         # input feature dim
H = 3072           # hidden dim
D_out = 768        # output dim

# Systolic array tile sizes for FC1 (M x D_in) x (D_in x H)
'''
FC1 multiplies (M × K) × (K × H) → output M × H with M=1024, H=3072. Choose Rt1 such that M % Rt1 == 0 and Ct1 such that H % Ct1 == 0.
FC2 multiplies (M × K2) × (K2 × D_out) → output M × D_out with M=1024, D_out=768. Choose Rt2 with M % Rt2 == 0 and Ct2 with D_out % Ct2 == 0.
1:3
'''
Rt1, Ct1 = 1, 1    # FC1: output is M x H
K1 = D_in
M1, N1 = M, H
P0_1, P1_1 = Rt1 + 2, Ct1 + 2


# Systolic array tile sizes for FC2 (M x H) x (H x D_out)
'''
Matches 4:3 ratio (4:3), divides both dims (1024%4==0, 768%3==0).
'''
Rt2, Ct2 = 1, 1     # FC2: output is M x D_out
K2 = H
M2, N2 = M, D_out
P0_2, P1_2 = Rt2 + 2, Ct2 + 2

# Optional inner-loop unroll/vectorization factor for ReLU packing
# Set to 1 (default) to keep original behavior. Increase to unroll inner
# packing loop by `UNROLL_U` elements per iteration if you have LUT headroom.
UNROLL_U = 8
# Normalize to at least 1 to avoid zero-step ranges
if UNROLL_U < 1:
    UNROLL_U = 1

# Toggle flags
RUN_SIMULATION = False  # Set to True to run simulation
RUN_HLS_CSYN = True     # Set to True to run HLS C synthesis


@df.region()
def top():
    # ===== FC1 Streams =====
    L3_A1: Stream[UInt(Rt1 * 8), 4]
    L3_B1: Stream[UInt(Ct1 * 8), 4]
    L3_C1: Stream[UInt(Rt1 * 32), 4]  # int32 output for accumulation
    
    L2_A1: Stream[UInt(Rt1 * 8), 4][P0_1 - 1]
    L2_B1: Stream[UInt(Ct1 * 8), 4][P1_1 - 1]
    
    L1_C1: Stream[UInt(Rt1 * 32), 4][Rt1, Ct1]
    L2_C1: Stream[UInt(Rt1 * 32), 4][Ct1]
    
    fifo_A1: Stream[int8, 4][Rt1, Ct1]
    fifo_B1: Stream[int8, 4][Rt1, Ct1]
    
    # ===== Intermediate streams (FC1 output -> ReLU -> FC2 input) =====
    fc1_to_bias: Stream[int32, 16]
    bias_to_gelu: Stream[float32, 16]
    relu_to_fc2: Stream[UInt(Rt2 * 8), 16]
    
    # ===== FC2 Streams =====
    L3_A2: Stream[UInt(Rt2 * 8), 4]
    L3_B2: Stream[UInt(Ct2 * 8), 4]
    L3_C2: Stream[UInt(Rt2 * 32), 4]  # int32 output
    
    L2_A2: Stream[UInt(Rt2 * 8), 4][P0_2 - 1]
    L2_B2: Stream[UInt(Ct2 * 8), 4][P1_2 - 1]
    
    L1_C2: Stream[UInt(Rt2 * 32), 4][Rt2, Ct2]
    L2_C2: Stream[UInt(Rt2 * 32), 4][Ct2]
    
    fifo_A2: Stream[int8, 4][Rt2, Ct2]
    fifo_B2: Stream[int8, 4][Rt2, Ct2]

    # ==================== FC1 SYSTOLIC ARRAY ====================
    
    @df.kernel(mapping=[1])
    def fc1_loadA(A_Packed: int16[M1 * K1 // Rt1]):
        for mt, nt in dsl.grid(M1 // Rt1, N1 // Ct1):
            for k in range(K1):
                packed_val: UInt(Rt1 * 8) = A_Packed[mt * K1 + k]
                L3_A1.put(packed_val)

    @df.kernel(mapping=[1])
    def fc1_loadB(W1_Packed: int32[K1 * N1 // Ct1]):
        for mt, nt in dsl.grid(M1 // Rt1, N1 // Ct1):
            for k in range(K1):
                packed_val: UInt(Ct1 * 8) = W1_Packed[nt * K1 + k]
                L3_B1.put(packed_val)

    @df.kernel(mapping=[1])
    def fc1_gemm():
        # Scalar matmul kernel specialized for Rt1=1, Ct1=1 (current defaults).
        # This avoids variable-dependent type annotations and uses simple
        # scalar accumulation per tile.
        for mt, nt in dsl.grid(M1 // Rt1, N1 // Ct1):
            c = 0
            for k in range(K1):
                a_packed = L3_A1.get()
                b_packed = L3_B1.get()
                a = a_packed[0:8]
                b = b_packed[0:8]
                c += a * b

            packed_out = 0
            packed_out[0:32] = c
            L3_C1.put(packed_out)

    @df.kernel(mapping=[1])
    def fc1_store_and_bias(b1: float32[H]):
        for mt, nt in dsl.grid(M1 // Rt1, N1 // Ct1):
            for n in range(Ct1):
                packed_val = L3_C1.get()
                for m in range(Rt1):
                    int_val: int32 = packed_val[m * 32 : (m + 1) * 32]
                    fp_val: float32 = int_val
                    bias_idx = nt * Ct1 + n
                    bias_to_gelu.put(fp_val + b1[bias_idx])

    # ==================== RELU ACTIVATION ====================
    
    @df.kernel(mapping=[1])
    def apply_relu():
        # Simpler ReLU using dsl.grid over (mt, j). No Python-level unrolling.
        # For each output column j, read Rt2 values from `bias_to_gelu`,
        # clamp to int8, pack into a word and forward to `relu_to_fc2`.
        for mt, j in dsl.grid(M // Rt2, H):
            packed: UInt(Rt2 * 8) = 0
            for m in range(Rt2):
                x: float32 = bias_to_gelu.get()
                # ReLU: clamp to int8 range [0, 127]
                val: int8 = 0
                if x > 127.0:
                    val = 127
                elif x < 0.0:
                    val = 0
                else:
                    val = x
                packed[m * 8 : (m + 1) * 8] = val
            relu_to_fc2.put(packed)

    # ==================== FC2 SYSTOLIC ARRAY ====================
    
    @df.kernel(mapping=[1])
    def fc2_loadA():
        # Read ReLU-activated values and pack them
        for mt in range(M2 // Rt2):
            for k in range(K2):
                # relu_to_fc2 now provides already-packed Rt2 elements
                packed: UInt(Rt2 * 8) = relu_to_fc2.get()
                L3_A2.put(packed)

    @df.kernel(mapping=[1])
    def fc2_loadB(W2_Packed: int32[K2 * N2 // Ct2]):
        for mt, nt in dsl.grid(M2 // Rt2, N2 // Ct2):
            for k in range(K2):
                packed_val: UInt(Ct2 * 8) = W2_Packed[nt * K2 + k]
                L3_B2.put(packed_val)

    @df.kernel(mapping=[1])
    def fc2_gemm():
        # Scalar matmul kernel specialized for Rt2=1, Ct2=1 (current defaults).
        for mt, nt in dsl.grid(M2 // Rt2, N2 // Ct2):
            c = 0
            for k in range(K2):
                a_packed = L3_A2.get()
                b_packed = L3_B2.get()
                a = a_packed[0:8]
                b = b_packed[0:8]
                c += a * b

            packed_out = 0
            packed_out[0:32] = c
            L3_C2.put(packed_out)

    @df.kernel(mapping=[1])
    def fc2_store_and_bias(b2: float32[D_out], Out: float32[M, D_out]):
        for mt, nt in dsl.grid(M2 // Rt2, N2 // Ct2):
            for n in range(Ct2):
                packed_val = L3_C2.get()
                for m in range(Rt2):
                    int_val: int32 = packed_val[m * 32 : (m + 1) * 32]
                    fp_val: float32 = int_val
                    row_idx = mt * Rt2 + m
                    col_idx = nt * Ct2 + n
                    Out[row_idx, col_idx] = fp_val + b2[col_idx]


def test_mlp_cascaded_systolic():
    print("Starting MLP with Cascaded Systolic Arrays...")
    print(f"\nMLP Architecture:")
    print(f"  FC1: ({M}, {D_in}) x ({D_in}, {H}) -> ({M}, {H})")
    print(f"  GELU activation")
    print(f"  FC2: ({M}, {H}) x ({H}, {D_out}) -> ({M}, {D_out})")
    print(f"\nSystolic Array Configurations:")
    print(f"  FC1 tile: Rt1={Rt1}, Ct1={Ct1}, Array={Rt1}x{Ct1}={Rt1*Ct1} PEs")
    print(f"  FC2 tile: Rt2={Rt2}, Ct2={Ct2}, Array={Rt2}x{Ct2}={Rt2*Ct2} PEs")
    
    if RUN_HLS_CSYN and hls.is_available("vitis_hls"):
        print("\n[Running HLS C Synthesis]")
        proj_name = f"mlp_cascaded_relu_tiled_M{M}_Din{D_in}_H{H}_FC1_{Rt1}x{Ct1}_FC2_{Rt2}x{Ct2}.prj"
        modc = df.build(
            top,
            target="vitis_hls",
            mode="csyn",
            project=proj_name,
            wrap_io=True,
        )
        modc()
        print(f"\n✅ HLS C Synthesis completed!")
        print(f"  Report: {proj_name}/out.prj/solution1/syn/report/top_csynth.rpt")
    else:
        print("\n[SKIPPED] HLS C Synthesis (RUN_HLS_CSYN=False or Vitis HLS not available)")


if __name__ == "__main__":
    test_mlp_cascaded_systolic()
