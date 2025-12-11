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
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from matrix_multiplies import mm1_return

# MLP dimensions from mlp.py
M = 1024           # batch size
D_in = 768         # input feature dim
H = 3072           # hidden dim
D_out = 768        # output dim

# Dimension aliases used in loop bounds inside kernels
# Keep these simple aliases so schedulers and kernels can refer
# to M1/N1/K1/etc consistently (some earlier files used these names).
M1, N1 = M, H
K1 = D_in
K2 = H
N2 = D_out

# Default P for the concrete wrapper
DEFAULT_P = 8




# Toggle flags
RUN_SIMULATION = False  # Set to True to run simulation
RUN_HLS_CSYN = True     # Set to True to run HLS C synthesis


@df.region()
def top():
    # ===== Single Dataflow Kernel: FC1 -> ReLU -> FC2 =====
    # Row-streaming matmuls with no packing (like self-attention SDPA)
    
    # Templated parallelism factor P: number of rows processed in parallel
    # Implements SDPA-like outer-batching: outer loop over M//P, inner loop over P.
    @df.kernel()
    def mlp_cascade[P: int16](
        A: "int8[1024, 768]",
        W1: "int8[768, 3072]",
        b1: "float32[3072]",
        W2: "int8[3072, 768]",
        b2: "float32[768]",
        Out: "float32[1024, 768]"
    ):
        """
        Cascaded MLP templated on P (rows processed in parallel).
        P is a compile-time integer template parameter. This creates P
        independent accumulator chains for FC1/FC2 similar to SDPA's P-row pattern.
        """

        # Iterate over batches of P rows
        for i_outer in allo.grid(M1 // P, name="row_outer"):
            # Local buffers for FC1 outputs for the P rows in this batch
            fc1_local: "int32[P, 3072]" = 0

            # Compute FC1 for P rows in this batch
            for p in allo.grid(P, name="p"):
                i = i_outer * P + p
                for j in allo.grid(N1, name="fc1_col"):
                    acc: int32 = 0
                    for k in allo.reduction(K1, name="fc1_k"):
                        acc += A[i, k] * W1[k, j]
                    fc1_local[p, j] = acc + b1[j]

            # Apply ReLU to the P-row block
            relu_local: "int8[P, 3072]" = 0
            for p in allo.grid(P, name="p_relu"):
                for j in allo.grid(N1, name="relu_j"):
                    val: float32 = fc1_local[p, j]
                    if val > 127.0:
                        relu_local[p, j] = 127
                    elif val < 0.0:
                        relu_local[p, j] = 0
                    else:
                        relu_local[p, j] = val

            # Compute FC2 for the P-row block and write to Out
            for p in allo.grid(P, name="fc2_p"):
                i = i_outer * P + p
                for j in allo.grid(N2, name="fc2_col"):
                    acc: int32 = 0
                    for k in allo.reduction(K2, name="fc2_k"):
                        acc += relu_local[p, k] * W2[k, j]
                    Out[i, j] = acc + b2[j]

    # Concrete wrapper that instantiates the templated kernel with P=8 so
    # the build/wrap_io flow has a concrete kernel to synthesize by default.
    @df.kernel(mapping=[1])
    def mlp_cascade_inst(
        A: "int8[1024, 768]",
        W1: "int8[768, 3072]",
        b1: "float32[3072]",
        W2: "int8[3072, 768]",
        b2: "float32[768]",
        Out: "float32[1024, 768]"
    ):
        mlp_cascade[8](A, W1, b1, W2, b2, Out)


def test_mlp_cascaded_systolic():
    print("Starting MLP with Row-Streaming Dataflow (no packing)...")
    print(f"\nMLP Architecture:")
    print(f"  FC1: ({M}, {D_in}) x ({D_in}, {H}) -> ({M}, {H})")
    print(f"  ReLU activation")
    print(f"  FC2: ({M}, {H}) x ({H}, {D_out}) -> ({M}, {D_out})")
    print(f"\nDataflow Style: Row-streaming (like self-attention SDPA)")
    print(f"  FC1 row-streaming matmul (no systolic array tiling)")
    print(f"  ReLU scalar per-element (no packing)")
    print(f"  FC2 row-streaming matmul (no systolic array tiling)")
    
    if RUN_HLS_CSYN and hls.is_available("vitis_hls"):
        print("\n[Running HLS C Synthesis]")
        proj_name = f"mlp_cascaded_relu_dataflow_M{M}_Din{D_in}_H{H}.prj"
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
