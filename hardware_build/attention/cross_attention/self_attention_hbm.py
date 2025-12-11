import allo
import numpy as np
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4, int64, Index, bool


def compute_kv_to_hbm[
    T: (bfloat16, float32, int4, int8),
    L: int16,   # Number of Tokens
    H: int16,   # Number of Heads
    D: int16,   # Embedding Length
    D_h: int16, # Head Embedding Length
](
    X:     "T[L, D]",
    W_k:   "T[H, D_h, D]",
    W_v:   "T[H, D_h, D]",
    K_out: "int32[H, L, D_h]",  # Output to HBM
    V_out: "int32[H, L, D_h]"   # Output to HBM
):
    """
    Compute K and V projections for all heads and write to HBM.
    
    This kernel reads X once and computes K/V for all heads in parallel,
    writing the results to HBM for later use in attention computation.
    """
    # Compute K and V for all heads
    for i in allo.grid(L, name="row_loop"):
        for k in allo.reduction(D, name="reduction_loop"):
            X_val: int32 = X[i, k]
            for j in allo.grid(D_h, name="col_loop"):
                for h in allo.grid(H, name="head_loop"):
                    K_out[h, i, j] = (0 if k == 0 else K_out[h, i, j]) + X_val * W_k[h, j, k]
                    V_out[h, i, j] = (0 if k == 0 else V_out[h, i, j]) + X_val * W_v[h, j, k]


def compute_q_row[
    T: (bfloat16, float32, int4, int8),
    L: int16,
    H: int16,
    D: int16,
    D_h: int16,
](
    X_row:  "T[D]",          # Single row of X (streamed in)
    W_q:    "T[H, D_h, D]",
    Q_row:  "int32[H, D_h]"  # Output: Q for this row, all heads
):
    """
    Compute Q projection for a single row of X, all heads.
    This can be called iteratively to stream Q rows.
    """
    for k in allo.reduction(D, name="reduction_loop"):
        X_val: int32 = X_row[k]
        for j in allo.grid(D_h, name="col_loop"):
            for h in allo.grid(H, name="head_loop"):
                Q_row[h, j] = (0 if k == 0 else Q_row[h, j]) + X_val * W_q[h, j, k]


def attention_row_from_hbm[
    T: (bfloat16, float32, int4, int8),
    L: int16,
    H: int16,
    D_h: int16,
](
    Q_row:   "int32[H, D_h]",       # Q for current row, all heads (from compute_q_row)
    K_hbm:   "int32[H, L, D_h]",    # Full K matrix from HBM
    V_hbm:   "int32[H, L, D_h]",    # Full V matrix from HBM
    scale:   "float32",
    out_row: "int32[H, D_h]"        # Output: attention result for this row, all heads
):
    """
    Compute attention for a single output row, reading K/V from HBM.
    
    For each head:
      attn_scores[j] = Q_row[h] @ K[h, j, :] for j in [0, L)
      attn_weights = softmax(attn_scores / scale)
      out_row[h] = attn_weights @ V[h]
    """
    for h in allo.grid(H, name="head_loop"):
        # Compute attention scores: Q_row @ K^T
        attn_scores: "int32[L]"
        max_val: int32 = -2147483648
        
        for j in allo.grid(L, name="score_col_loop"):
            acc: int32 = 0
            for k in allo.reduction(D_h, name="score_reduction"):
                acc += Q_row[h, k] * K_hbm[h, j, k]
            attn_scores[j] = acc
            if acc > max_val:
                max_val = acc
        
        # Softmax with numerical stability
        sum_exp: float32 = 0.0
        attn_weights: "float32[L]"
        
        for j in allo.grid(L, name="exp_loop"):
            exp_val: float32 = allo.exp((attn_scores[j] - max_val) / scale)
            attn_weights[j] = exp_val
            sum_exp += exp_val
        
        for j in allo.grid(L, name="norm_loop"):
            attn_weights[j] = attn_weights[j] / sum_exp
        
        # Weighted sum with V: attn_weights @ V
        for d in allo.grid(D_h, name="output_col_loop"):
            acc_out: float32 = 0.0
            for j in allo.reduction(L, name="output_reduction"):
                acc_out += attn_weights[j] * V_hbm[h, j, d]
            out_row[h, d] = acc_out


def self_attention_hbm[
    T: (bfloat16, float32, int4, int8),
    L: int16,
    H: int16,
    D: int16,
    D_h: int16,
](
    X:       "T[L, D]",
    W_q:     "T[H, D_h, D]",
    W_k:     "T[H, D_h, D]",
    W_v:     "T[H, D_h, D]",
    W_o:     "T[H, D_h, D]",
    scale:   "float32",
    K_hbm:   "int32[H, L, D_h]",  # Intermediate storage in HBM
    V_hbm:   "int32[H, L, D_h]",  # Intermediate storage in HBM
    out:     "T[L, D]"
):
    """
    Full self-attention with HBM staging for K and V.
    
    Architecture:
    1. Compute K, V for all heads → write to HBM
    2. For each output row i:
       a. Compute Q[i] (single row, all heads)
       b. Read K, V from HBM
       c. Compute attention for row i
       d. Apply output projection
       e. Write output row
    """
    # Phase 1: Compute all K and V, store in HBM
    for i in allo.grid(L, name="kv_row_loop"):
        for k in allo.reduction(D, name="kv_reduction"):
            X_val: int32 = X[i, k]
            for j in allo.grid(D_h, name="kv_col_loop"):
                for h in allo.grid(H, name="kv_head_loop"):
                    K_hbm[h, i, j] = (0 if k == 0 else K_hbm[h, i, j]) + X_val * W_k[h, j, k]
                    V_hbm[h, i, j] = (0 if k == 0 else V_hbm[h, i, j]) + X_val * W_v[h, j, k]
    
    # Phase 2: Compute attention row by row
    for i_out in allo.grid(L, name="attn_row_loop"):
        # Compute Q for current row
        Q_row: "int32[H, D_h]"
        for k in allo.reduction(D, name="q_reduction"):
            X_val: int32 = X[i_out, k]
            for j in allo.grid(D_h, name="q_col_loop"):
                for h in allo.grid(H, name="q_head_loop"):
                    Q_row[h, j] = (0 if k == 0 else Q_row[h, j]) + X_val * W_q[h, j, k]
        
        # Compute attention for each head
        attn_out: "int32[H, D_h]"
        for h in allo.grid(H, name="attn_head_loop"):
            # Compute attention scores
            attn_scores: "int32[L]"
            max_val: int32 = -2147483648
            
            for j in allo.grid(L, name="score_loop"):
                acc: int32 = 0
                for d in allo.reduction(D_h, name="score_reduction"):
                    acc += Q_row[h, d] * K_hbm[h, j, d]
                attn_scores[j] = acc
                if acc > max_val:
                    max_val = acc
            
            # Softmax
            sum_exp: float32 = 0.0
            attn_weights: "float32[L]"
            for j in allo.grid(L, name="softmax_exp_loop"):
                exp_val: float32 = allo.exp((attn_scores[j] - max_val) / scale)
                attn_weights[j] = exp_val
                sum_exp += exp_val
            
            for j in allo.grid(L, name="softmax_norm_loop"):
                attn_weights[j] = attn_weights[j] / sum_exp
            
            # Weighted sum with V
            for d in allo.grid(D_h, name="output_loop"):
                acc_out: float32 = 0.0
                for j in allo.reduction(L, name="v_reduction"):
                    acc_out += attn_weights[j] * V_hbm[h, j, d]
                attn_out[h, d] = acc_out * 32768.0  # Scale back to int range
        
        # Output projection: concat heads and multiply by W_o
        for d_out in allo.grid(D, name="output_proj_outer"):
            acc_proj: int32 = 0
            for h in allo.grid(H, name="output_proj_h"):
                for d_h in allo.reduction(D_h, name="output_proj_inner"):
                    acc_proj += (attn_out[h, d_h] >> 15) * W_o[h, d_h, d_out]
            out[i_out, d_out] = acc_proj


# Test/build configuration
if __name__ == "__main__":
    L = 2048
    H = 12
    D = 768
    D_h = 64
    
    # Test compute_kv_to_hbm
    s = allo.customize(compute_kv_to_hbm, instantiate=[int8, L, H, D, D_h])
    
    # HBM mapping for the kernel
    hbm_mapping = {
        "X": 0,
        "W_k": 1,
        "W_v": 2,
        "K_out": 3,
        "V_out": 4,
    }
    
    print("Building compute_kv_to_hbm kernel...")
    s.build(
        target="vitis_hls",
        mode="csyn",
        project="compute_kv_hbm.prj",
        configs={"hbm_mapping": hbm_mapping},
    )()
