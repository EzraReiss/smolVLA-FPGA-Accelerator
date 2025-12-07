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
](
    X:   "T[H, L, D_h]",
    W_q: "T[H, D_h, D_h]",
    W_k: "T[H, D_h, D_h]",
    W_v: "T[H, D_h, D_h]",
    scale: "float32", #takes the value of 8
    out: "T[H, L, D_h]"
):
    # ===== QKV Projection Stage =====
    
    # # ===== QKV Projection (manual matmul-transpose) =====
    for h1 in allo.grid(H, name="head_loop"):
        Q: "T[L, D_h]" = 0
        K: "T[L, D_h]" = 0
        V: "T[L, D_h]" = 0

        for i_precalc in allo.grid(L, name="mm_i_loop"):
            for j_precalc in allo.grid(D_h, name="mm_j_loop"):
                for k_precalc in allo.reduction(D_h, name="prj_dot_product"):
                    Q[i_precalc, j_precalc] += X[h1, i_precalc, k_precalc] * W_q[h1, j_precalc, k_precalc]
                    K[i_precalc, j_precalc] += X[h1, i_precalc, k_precalc] * W_k[h1, j_precalc, k_precalc]
                    V[i_precalc, j_precalc] += X[h1, i_precalc, k_precalc] * W_v[h1, j_precalc, k_precalc]

        for i_out in allo.grid(L, name="row_loop"):
            attn_row: "int32[L]"
            max_val: "int32" = -2147483648
            
            for j_attn in allo.grid(L, name="attn_loop"):
                acc: "int32" = 0
                for k_attn in allo.reduction(D_h, name="dot_product"):
                    acc += Q[i_out, k_attn] * K[j_attn, k_attn]
                
                attn_row[j_attn] = acc
                
                if acc > max_val:
                    max_val = acc
                    
            softmax_rows: "T[L]"
            sum_exps: "T" = 0.0
            
            for j_exp in allo.grid(L, name="exp_loop"):
                exp_pow: "float32" = attn_row[j_exp] - max_val
                exp_val: "T" = allo.exp(exp_pow / scale)
                softmax_rows[j_exp] = exp_val
        
            softmax_rows_2: "T[L]"
            for j_exp_sum in allo.grid(L, name="sum_loop"):
                softmax_row = softmax_rows[j_exp_sum]
                softmax_rows_2[j_exp_sum] = softmax_row
                sum_exps += softmax_row
                    
            softmax_scaled: "int16[L]"
            for j_norm in allo.grid(L, name="norm_loop"):
                norm_val: "float32" = softmax_rows_2[j_norm] / sum_exps
                softmax_scaled[j_norm] = norm_val * 32768.0
            
            acc_out: "int32[D_h]" = 0
            for j_out in allo.grid(L, name="out_row_loop"):
                softmax_val: "int32" = softmax_scaled[j_out]
                
                for k_out in allo.reduction(D_h, name="out_loop"):   
                    v_val: "int32" = V[j_out, k_out]
                    acc_out[k_out] += softmax_val * v_val

            for k_final in allo.grid(D_h, name="final_loop"):
                out[h1, i_out, k_final] = acc_out[k_final] >> 15

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
    