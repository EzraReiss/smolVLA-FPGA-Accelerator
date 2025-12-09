import allo
import numpy as np
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4, int64, Index, bool
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
    L: int16, # Number of Tokens
    H: int16, # Number of Heads
    D: int16, # Embedding Length
    D_h: int16, # Head Embedding Length
    P: int16, # Parallelism factor - number of rows to process together
    P_s: int16, # Summation parallelism factor
](
    X:   "T[L, D]",
    W_q: "T[H, D_h, D]",
    W_k: "T[H, D_h, D]",
    W_v: "T[H, D_h, D]",
    W_o: "T[H, D_h, D]",
    scale: "float32", #takes the value of 8
    out: "T[L, D]"
):
    # ===== QKV Projection Stage =====
    
    # # ===== QKV Projection (manual matmul-transpose) =====
    for h1 in allo.grid(H, name="head_loop"):
        Q: "int32[L, D_h]"
        K: "int32[L, D_h]"
        V: "int32[L, D_h]" 

        for i_precalc in allo.grid(L//P, name="mm_i_loop"):
            for k_precalc in allo.reduction(D, name="prj_dot_product"):
                for j_precalc in allo.grid(D_h, name="mm_j_loop"):
                    for p_inner in allo.grid(P, name="p_inner_loop"):
                        i_precalc_actual: int32 = i_precalc * P + p_inner
                        X_int32: "int16" = X[i_precalc, k_precalc]
                        Q[i_precalc_actual, j_precalc] = (0 if k_precalc == 0 else Q[i_precalc_actual, j_precalc]) + X_int32 * W_q[h1, j_precalc, k_precalc]
                        K[i_precalc_actual, j_precalc] = (0 if k_precalc == 0 else K[i_precalc_actual, j_precalc]) + X_int32 * W_k[h1, j_precalc, k_precalc]
                        V[i_precalc_actual, j_precalc] = (0 if k_precalc == 0 else V[i_precalc_actual, j_precalc]) + X_int32 * W_v[h1, j_precalc, k_precalc]

        for i_out in allo.grid(L//P, name="row_loop"):
            attn_row: "int32[P, L]"
            max_val: "int32[P]" = -2147483648
            loop_base: int32 = i_out * P
            for j_attn in allo.grid(L, name="attn_loop"):
                for p_attn in allo.grid(P, name="p_attn_loop"):
                    acc: "int32" = 0
                    for k_attn in allo.reduction(D_h, name="dot_product"):
                        acc += Q[loop_base + p_attn, k_attn] * K[j_attn, k_attn]
                    
                    attn_row[p_attn, j_attn] = acc
                    
                    if acc > max_val[p_attn]:
                        max_val[p_attn] = acc
                    
            sum_exps_p: "float32[P, P_s]" = 0.0
            softmax_rows: "float32[P, L]" 
            
            for j_exp_P_s in allo.grid(L // P_s, name="exp_loop"):
                for j_exp_sum in allo.grid(P_s, name="exp_loop_2"):
                    j_exp = j_exp_P_s * P_s + j_exp_sum
                    for p_exp in allo.grid(P, name="p_exp_loop"):
                        exp_pow: "float32" = attn_row[p_exp, j_exp] - max_val[p_exp]
                        exp_val: "float32" = allo.exp(exp_pow / scale)
                        softmax_rows[p_exp, j_exp] = exp_val
                        sum_exps_p[p_exp, j_exp_sum] += exp_val

            sum_exps: "float32[P]" = 0.0

            for p_sum in allo.grid(P, name="p_sum_loop"):
                for sum_i in allo.grid(P_s, name="sum_loop"):
                    sum_exps[p_sum] += sum_exps_p[p_sum, sum_i]

    
            softmax_scaled: "float32[P, L]"
            for j_norm in allo.grid(L, name="norm_loop"):
                for p_norm in allo.grid(P, name="p_norm_loop"):
                    norm_val: "float32" = softmax_rows[p_norm, j_norm] / sum_exps[p_norm]
                    softmax_scaled[p_norm, j_norm] = norm_val * 32768.0
            
            acc_out: "int32[P, D_h]" = 0
            for j_out in allo.grid(L, name="out_row_loop"):
                for p_out in allo.grid(P, name="p_out_loop"):
                    softmax_val: "int32" = softmax_scaled[p_out, j_out]
                    
                    for k_out in allo.reduction(D_h, name="out_loop"):   
                        v_val: "int32" = V[j_out, k_out]
                        acc_out[p_out, k_out] += softmax_val * v_val

            for i_final in allo.grid(D, name="i_final_loop"):
                for k_final in allo.grid(D_h, name="final_loop"):
                    weight: "T" = W_o[h1, k_final, i_final]
                    for p_final in allo.grid(P, name="p_final_loop"):
                        val: "int32" = acc_out[p_final, k_final]
                        out[i_out*P + p_final, i_final] = (val >> 15) * weight
                        



def layer_norm[
    T: (int4, int8),
    L: int16,
    D: int16
](
    x: "T[L, D]",
    gamma: "T[D]",
    beta: "T[D]",
    x_out: "T[L, D]"
):
    total: "int32[L]" = 0
    total_sq: "int32[L]" = 0
    
    for i_sum in allo.grid(L, name="ln_inner_outer"):
        for j_sum in allo.reduction(D, name="ln_inner"):
            val: "int32" = x[i_sum, j_sum]
            total[i_sum] += val
            total_sq[i_sum] += val * val
            
    mean: "float32[L]"
    inv_std: "float32[L]"
            
    for i_stat in allo.grid(L, name="ln_stats_loop"):
        mean_i: "float32" = total[i_stat] / D
        mean[i_stat] = mean_i 
        variance: "float32" = (total_sq[i_stat] / D) - (mean_i * mean_i)
        inv_std[i_stat] = 1.0 / allo.sqrt(variance + 1e-8)
        
    for i_out in allo.grid(L, name="ln_out_outer"):
        mean_i: "float32" = mean[i_out]
        inv_std_i: "float32" = inv_std[i_out]
        
        for j_out in allo.grid(D, name="ln_out_inner"):
            norm_val: "float32" = (x[i_out, j_out] - mean_i) * inv_std_i
            scaled: "float32" = norm_val * gamma[j_out]
            shifted: "float32" = scaled + beta[j_out]
            x_out[i_out, j_out] = shifted
    
def self_attention_return[
    T: (bfloat16, float32, int4, int8),
    L: int16, # Number of Tokens
    H: int16, # Number of Heads
    D: int16, # Embedding Length
    D_h: int16, # Head Embedding Length
    P: int16, # Parallelism factor - number of rows to process together
    P_s: int16, # Summation parallelism factor
](
    X:   "T[L, D]",
    W_q: "T[H, D_h, D]",
    W_k: "T[H, D_h, D]",
    W_v: "T[H, D_h, D]",
    W_o: "T[H, D_h, D]",
    scale: "float32", #takes the value of 8
    gamma: "T[D]",
    beta: "T[D]",
    x_ln: "T[L, D]"
):
    out: "T[L, D]"
    self_attention[T, L, H, D, D_h, P, P_s, "sa1"](X, W_q, W_k, W_v, W_o, scale, out)
    layer_norm[T, L, D, "layer_norm2"](out, gamma,  beta, x_ln)
        
        

H: int16 = 12
P: int16 = 8
L: int16 = 1024
D_h: int16 = 64
if __name__ == "__main__":
    pass
    