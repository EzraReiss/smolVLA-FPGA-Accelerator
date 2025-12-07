import allo
import numpy as np
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4, int64
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from matrix_multiplies import mm_transpose, mm1, mm_transpose_return, mm1_return
from attention.cross_attention.softmax import softmax_baseline, softmax_return
from attention.cross_attention.sdpa import sdpa_streaming_8row as sdpa
from attention.cross_attention.sdpa_dataflow_scheduler import schedule_sdpa_streaming_4row_parallel as sdpa_schedule
from allo.customize import Partition as partition



def self_attention[
    T: (bfloat16, float32, int4, int8),
    L: int16,
    D_h: int16,
    H: int16, #num heads in parallel
    P: int16  # Parallelism factor (8 for 8-row streaming SDPA)
](
    X:   "T[H, L, D_h]",
    W_q: "T[H, D_h, D_h]",
    W_k: "T[H, D_h, D_h]",
    W_v: "T[H, D_h, D_h]",
    W_o: "T[H, D_h, D_h]",
    scale: "float32",
    out: "T[H, L, D_h]"
):
    """
    Self-attention with integrated QKV projection.
    Structure:
    - QKV Projection: Compute Q=X@W_q, K=X@W_k, V=X@W_v
    - Outer loop: L//P iterations (batch of P rows)
    - Middle loop: P (row index within batch) 
    - Inner loops: pipelined computation for each row
    """
    # ===== QKV Projection Stage =====
    Q: "T[H, L, D_h]" = 0
    K: "T[H, L, D_h]" = 0
    V: "T[H, L, D_h]" = 0
    
    # # ===== QKV Projection (manual matmul-transpose) =====
    for h1 in allo.grid(H, name="head_loop"):
        for i in allo.grid(L, name="mm_loop"):
            for j in allo.grid(D_h, name="mm_loop"):
                for k in allo.reduction(D_h, name="prj_dot_product"):
                    Q[h1, i, j] += X[h1, i, k] * W_q[h1, j, k] #standard transpose matmul - TODO: Verify if its supposed to transpose for VLM encoder
                    K[h1, i, j] += X[h1, i, k] * W_k[h1, j, k]
                    V[h1, i, j] += X[h1, i, k] * W_v[h1, j, k]
                    
            
        sdpa[T, L, D_h, P, "sdpa"](Q[h1, :, :], K[h1, :, :], V[h1, :, :], scale, out[h1, :, :])


def self_attention_2[
    T: (bfloat16, float32, int4, int8),
    L: int16, # Number of Tokens
    H: int16, # Number of Heads
    D_h: int16, # Head Embedding Length
    D_o: int16, # Output Embedding Length (H*D_h)
](
    X:   "T[H, L, D_h]",
    W_q: "T[H, D_h, D_h]",
    W_k: "T[H, D_h, D_h]",
    W_v: "T[H, D_h, D_h]",
    W_o: "T[H, D_h, D_h]",
    scale: "float32",
    out: "T[L, D_o]"
):
    pass

H: int16 = 12
P: int16 = 8
L: int16 = 1024
D_h: int16 = 64
if __name__ == "__main__":
    s1 = allo.customize(self_attention, instantiate=[int8, L, D_h, H, P])
    _, s2 = sdpa_schedule(np.int8, int8, P, mode="llvm")
    loop = s1.get_loops("self_attention")
    # s1.dataflow(loop["head_loop"]["h1"])
    s1.pipeline(loop["head_loop"]["j"])
    s1.unroll(loop["head_loop"]["j"], factor=8)
    s1.compose(s2, id="sdpa")
    W_q = np.random.randint(-128, 127, size=(H, D_h, D_h), dtype=np.int8)
    W_k = np.random.randint(-128, 127, size=(H, D_h, D_h), dtype=np.int8)
    W_v = np.random.randint(-128, 127, size=(H, D_h, D_h), dtype=np.int8)
    W_o = np.random.randint(-128, 127, size=(H, D_h, D_h), dtype=np.int8)
    X = np.random.randint(-128, 127, size=(H, L, D_h), dtype=np.int8)
    out = np.zeros((H, L, D_h), dtype=np.int8)
    scale = np.float32(8.0)
    s1.build(target="vitis_hls", mode="csyn", project="self_attention_base_dataflow_v3.prj")()
    