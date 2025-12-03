import allo
import numpy as np
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4, int64
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from matrix_multiplies import mm_transpose, mm1, mm_transpose_return, mm1_return
from attention.cross_attention.softmax import softmax_baseline, softmax_return


def numpy_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    

def sdpa_np(Q, K, V, d_h = 768 / 12):
    # Q: (L, D_h) queries
    # K: (L, D_h) keys
    # V: (L, D_h) values
    # compute scaled dot-product attention: softmax(Q @ K.T / sqrt(d_h)) @ V
    B = Q @ K.T / np.sqrt(d_h)
    softmaxed_output = numpy_softmax(B, axis=-1)
    output = softmaxed_output @ V
    return output

def sdpa[    
    T: (bfloat16, float32),
    L: int16,  
    D_h: int16     
](
    Q: "T[L, D_h]",
    K: "T[L, D_h]",
    V: "T[L, D_h]",
    scale: "T",            # scalar divisor (e.g. sqrt(d_h))
    out: "T[L, D_h]"
):
    # Temporary buffer for attention scores: (L, L)
    B: "T[L, L]" = 0.0

    # Compute raw scores: B = Q @ K^T
    # mm_transpose[T, P, Q, R]: A[P,Q] @ B[R,Q]^T = out[P,R]
    # Q[L, D_h] @ K[L, D_h]^T = B[L, L]
    mm_transpose[T, L, D_h, L](Q, K, B)

    # Scale by divisor (e.g. sqrt(d_h))
    for i0, j0 in allo.grid(L, L):
        B[i0, j0] = B[i0, j0] / scale

    # Apply row-wise softmax over keys (each row has length L)
    softmax_baseline[T, L, L](B)

    # Final weighted sum: out = softmax(B) @ V
    # mm1[T, P, Q, R]: A[P,Q] @ B[Q,R] = out[P,R]
    # B[L, L] @ V[L, D_h] = out[L, D_h]
    mm1[T, L, L, D_h](B, V, out)


def sdpa_with_return[
    T: (bfloat16, float32),
    L: int16,
    D_h: int16
](
    Q: "T[L, D_h]",
    K: "T[L, D_h]",
    V: "T[L, D_h]",
    scale: "T"
) -> "T[L, D_h]":
    """
    SDPA variant that returns the output instead of modifying in-place.
    Better for dataflow as it has clear producer/consumer relationship.
    This avoids the read-write conflict on output arrays in dataflow regions.
    """
    # Compute Q @ K^T
    B: "T[L, L]" = 0.0
    mm_transpose[T, L, D_h, L](Q, K, B)
    
    # Scale by divisor
    for i, j in allo.grid(L, L):
        B[i, j] = B[i, j] / scale
    
    # Apply row-wise softmax
    softmax_baseline[T, L, L](B)
    
    # Compute B @ V and return
    out: "T[L, D_h]" = 0.0
    mm1[T, L, L, D_h](B, V, out)
    
    return out


def sdpa_dataflow[
    T: (bfloat16, float32),
    L: int16,
    D_h: int16
](
    Q: "T[L, D_h]",
    K: "T[L, D_h]",
    V: "T[L, D_h]",
    scale: "T"
) -> "T[L, D_h]":
    """
    Fully dataflow-optimized SDPA using only return functions.
    All subfunctions return values creating a clear producer/consumer chain:
    Q,K -> mm_transpose_return -> B -> scale -> softmax_return -> B_softmax -> mm1_return -> out
    
    This enables true streaming dataflow where each stage can start as soon as
    the previous stage produces data.
    """
    # Stage 1: Compute Q @ K^T and return
    B: "T[L, L]" = mm_transpose_return[T, L, D_h, L](Q, K)
    
    # Stage 2: Scale (element-wise, can't avoid in-place)
    B_scaled: "T[L, L]" = 0.0
    for i, j in allo.grid(L, L, name="scale"):
        B_scaled[i, j] = B[i, j] / scale
    
    # Stage 3: Apply softmax and return
    B_softmax: "T[L, L]" = softmax_return[T, L, L](B_scaled)
    
    # Stage 4: Final matrix multiply and return
    out: "T[L, D_h]" = mm1_return[T, L, L, D_h](B_softmax, V)
    
    return out


def sdpa_streaming[
    T: (bfloat16, float32, int4, int8),
    L: int16,
    D_h: int16
](
    Q: "T[L, D_h]",
    K: "T[L, D_h]",
    V: "T[L, D_h]",
    scale: "float32",
    out: "T[L, D_h]"
):
    """
    Row-streaming SDPA that processes one output row at a time.
    
    Supports int4/int8 quantized inputs with mixed-precision compute:
    - Integer matmul with int32 accumulator for int4/int8 types
    - Floating-point softmax (required for exp/div operations)
    - Output quantized back to input type T
    
    Key insight: Softmax is row-independent. Once we compute a complete row
    of Q @ K^T, we can immediately apply softmax to that row and compute
    the corresponding output row, all while using only O(L) intermediate storage
    instead of O(L^2).
    
    Memory: 2 * L floats for row buffers vs L * L floats for full materialization
    For L=1024, float32: 8KB vs 4MB = 512x reduction!
    
    Pipeline structure (per row i):
      1. Compute row i of Q @ K^T (dot products with all K rows)
      2. Scale the row
      3. Apply softmax to the row (requires full row for max/sum)
      4. Compute row i of output = softmax_row @ V
    """
    # Row buffers - use float32 for softmax (exp/div require floating point)
    attn_row: "float32[L]"      # One row of attention scores (after Q @ K^T)
    softmax_row: "float32[L]"   # One row after softmax (float, before scaling)
    softmax_row_int: "int16[L]" # Scaled softmax row for integer accumulation
    max_val: "float32"         # Max value for numerical stability
    
    # Process one output row at a time
    for i in allo.grid(L, name="row_loop"):
        
        # ===== Stage 1: Compute row i of Q @ K^T =====
        # attn_row[j1] = sum_k1 Q[i,k1] * K[j1,k1] (K transposed)
        for j1 in allo.grid(L, name="mm_j"):
            # Use int32 accumulator for integer types to prevent overflow
            acc: "int32" = 0
            for k1 in allo.grid(D_h, name="mm_k"):
                q_val: "int32" = Q[i, k1]
                k_val: "int32" = K[j1, k1]
                acc += q_val * k_val
            # Convert to float32 and scale
            acc_float: "float32" = acc
            acc_float = acc_float / scale
            #Find max val here to save a loop later
            if j1 == 0:
                max_val = acc_float
            else:                
                if acc_float > max_val:
                    max_val = acc_float
            attn_row[j1] = acc_float
        
        sum_exp: "float32" = 0.0
        for j2 in allo.grid(L, name="exp_j"):
            exp_val: "float32" = allo.exp(attn_row[j2] - max_val)
            softmax_row[j2] = exp_val
            sum_exp += exp_val

        # ===== Stage 2: Normalize and scale softmax row =====
        softmax_scale: "float32" = 32768.0
        for j3 in allo.grid(L, name="norm_j"):
            norm_val: "float32" = softmax_row[j3] / sum_exp
            # Scale to fixed-point int16
            softmax_row_scaled: "int16" = norm_val * softmax_scale
            softmax_row_int[j3] = softmax_row_scaled
            
        # ===== Stage 2: Compute output row i with integer arithmetic =====
        # acc_out[d] = sum_j(softmax_row_int[j] * V[j,d]) 
        # Result is scaled by softmax_scale, will rescale at output
        acc_out: "int32[D_h]" = 0
        
        for j4 in allo.grid(L, name="out_j"):
            s_val: "int32" = softmax_row_int[j4]
            for d in allo.grid(D_h, name="out_d"):
                v_val: "int32" = V[j4, d]
                acc_out[d] += s_val * v_val
        
        # Write outputs - rescale from fixed-point back to int8
        # Divide by softmax_scale (32768) to get back to original scale
        for d2 in allo.grid(D_h, name="out_write"):
            # Shift right by 15 bits = divide by 32768
            rescaled: "int32" = acc_out[d2] >> 15
            out_val: T = rescaled
            out[i, d2] = out_val


def sdpa_streaming_8row[
    T: (bfloat16, float32, int4, int8),
    L: int16,
    D_h: int16,
    P: int16  # Parallelism factor (4)
](
    Q: "T[L, D_h]",
    K: "T[L, D_h]",
    V: "T[L, D_h]",
    scale: "float32",
    out: "T[L, D_h]"
):
    """
    P-row parallel streaming SDPA with P on the outside.
    
    By processing P rows simultaneously with P as the outer loop,
    we get P independent accumulator chains. With P=8 and fadd latency~7,
    each accumulator has enough distance between accesses for II=1.
    
    Structure:
    - Outer loop: L//P iterations (batch of P rows)
    - Middle loop: P (row index within batch) 
    - Inner loops: pipelined computation for each row
    
    This matches the structure of sdpa_streaming but processes P rows per batch.
    """

    
    # Process P rows at a time
    for i_outer in allo.grid(L // P, name="row_outer"):
        # Row buffers for P rows - each row has its own buffers
        attn_rows: "float32[P, L]"      # P rows of attention scores
        softmax_rows: "float32[P, L]"   # P rows after softmax
        softmax_rows_int: "int16[P, L]" # Scaled softmax for integer MAC
        max_vals: "float32[P]"          # Max value per row
        sum_exps: "float32[P]"          # Sum of exp per row
        acc_out: "int32[P, D_h]"        # Output accumulators for P rows
        # ===== Stage 1: Compute P rows of Q @ K^T =====
        # P is outer, j1 is inner (pipelined)
        for p in allo.grid(P, name="mm_p"):
            i: "int16" = i_outer * P + p
            for j1 in allo.grid(L, name="mm_j"):
                # Dot product Q[i,:] @ K[j1,:]
                acc: "int32" = 0
                for k1 in allo.grid(D_h, name="mm_k"):
                    q_val: "int32" = Q[i, k1]
                    k_val: "int32" = K[j1, k1]
                    acc += q_val * k_val
                # Convert to float32 and scale
                acc_float: "float32" = acc
                acc_float = acc_float / scale
                # Track max for this row
                if j1 == 0:
                    max_vals[p] = acc_float
                else:
                    if acc_float > max_vals[p]:
                        max_vals[p] = acc_float
                attn_rows[p, j1] = acc_float
        
        # ===== Stage 2: Compute exp and sum for P rows =====
        # Initialize sum_exps
        for p_init in allo.grid(P, name="init_sum"):
            sum_exps[p_init] = 0.0
        
        # P is outer, j2 is inner (pipelined)
        # Each row has its own sum_exp accumulator
        for j2 in allo.grid(L, name="exp_p"):
            for p2 in allo.grid(P, name="exp_j"):
                exp_val: "float32" = allo.exp(attn_rows[p2, j2] - max_vals[p2])
                softmax_rows[p2, j2] = exp_val
                sum_exps[p2] += exp_val

        # ===== Stage 3: Normalize and scale softmax rows =====
        # P is outer, j3 is inner (pipelined)
        softmax_scale: "float32" = 32768.0
        for p3 in allo.grid(P, name="norm_p"):
            for j3 in allo.grid(L, name="norm_j"):
                norm_val: "float32" = softmax_rows[p3, j3] / sum_exps[p3]
                softmax_scaled: "int16" = norm_val * softmax_scale
                softmax_rows_int[p3, j3] = softmax_scaled
                
        # ===== Stage 4: Initialize output accumulators =====
        for p_init in allo.grid(P, name="init_p"):
            for d_init in allo.grid(D_h, name="init_d"):
                acc_out[p_init, d_init] = 0
        
        # ===== Stage 5: Compute output rows with integer arithmetic =====
        # P is outer, j4 is inner (pipelined), d is innermost
        for p4 in allo.grid(P, name="out_p"):
            for j4 in allo.grid(L, name="out_j"):
                s_val: "int32" = softmax_rows_int[p4, j4]
                for d in allo.grid(D_h, name="out_d"):
                    v_val: "int32" = V[j4, d]
                    acc_out[p4, d] += s_val * v_val
        
        # ===== Stage 6: Write outputs - rescale from fixed-point =====
        for p5 in allo.grid(P, name="write_p"):
            i_out: "int16" = i_outer * P + p5
            for d2 in allo.grid(D_h, name="write_d"):
                rescaled: "int32" = acc_out[p5, d2] >> 15
                out_val: T = rescaled
                out[i_out, d2] = out_val




def self_attention[
    T: (bfloat16, float32, int4, int8),
    L: int16,
    D_h: int16,
    P: int16  # Parallelism factor (4)
](
    X: "T[L, D_h]",
    W_q: "T[D_h, D_h]",
    W_k: "T[D_h, D_h]",
    W_v: "T[D_h, D_h]",
    scale: "float32",
    out: "T[L, D_h]"
):
    """
    Self-attention with integrated QKV projection.
    
    Computes Q, K, V from input X using projection matrices W_q, W_k, W_v,
    then applies P-row parallel streaming SDPA.
    
    By processing P rows simultaneously with P as the outer loop,
    we get P independent accumulator chains. With P=8 and fadd latency~7,
    each accumulator has enough distance between accesses for II=1.
    
    Structure:
    - QKV Projection: Compute Q=X@W_q, K=X@W_k, V=X@W_v
    - Outer loop: L//P iterations (batch of P rows)
    - Middle loop: P (row index within batch) 
    - Inner loops: pipelined computation for each row
    """
    # ===== QKV Projection Stage =====
    Q: "T[L, D_h]" = 0.0
    K: "T[L, D_h]" = 0.0
    V: "T[L, D_h]" = 0.0
    
    # ===== QKV Projection (manual matmul-transpose) =====
    # Compute Q = X @ W_q  where W_q is stored as [D_q, D_h]
    # mm_transpose semantics: out[i, j] = sum_k X[i,k] * W_q[j,k]
    for i in allo.grid(L, name="q_i"):
        for j in allo.grid(D_h, name="q_j"):
            acc_q: "T" = 0.0
            for k in allo.grid(D_h, name="q_k"):
                acc_q += X[i, k] * W_q[j, k]
            Q[i, j] = acc_q

    # Compute K = X @ W_k
    for i in allo.grid(L, name="k_i"):
        for j in allo.grid(D_h, name="k_j"):
            acc_k: "T" = 0.0
            for k in allo.grid(D_h, name="k_k"):
                acc_k += X[i, k] * W_k[j, k]
            K[i, j] = acc_k

    # Compute V = X @ W_v
    for i in allo.grid(L, name="v_i"):
        for j in allo.grid(D_h, name="v_j"):
            acc_v: "T" = 0.0
            for k in allo.grid(D_h, name="v_k"):
                acc_v += X[i, k] * W_v[j, k]
            V[i, j] = acc_v

    # ===== SDPA Stage: Process P rows at a time =====
    for i_outer in allo.grid(L // P, name="row_outer"):
        # Row buffers for P rows - each row has its own buffers
        attn_rows: "float32[P, L]"      # P rows of attention scores
        softmax_rows: "float32[P, L]"   # P rows after softmax
        softmax_rows_int: "int16[P, L]" # Scaled softmax for integer MAC
        max_vals: "float32[P]"          # Max value per row
        sum_exps: "float32[P]"          # Sum of exp per row
        acc_out: "int32[P, D_h]"        # Output accumulators for P rows
        # ===== Stage 1: Compute P rows of Q @ K^T =====
        # P is outer, j1 is inner (pipelined)
        for p in allo.grid(P, name="mm_p"):
            i: "int16" = i_outer * P + p
            for j1 in allo.grid(L, name="mm_j"):
                # Dot product Q[i,:] @ K[j1,:]
                acc: "int32" = 0
                for k1 in allo.grid(D_h, name="mm_k"):
                    q_val: "int32" = Q[i, k1]
                    k_val: "int32" = K[j1, k1]
                    acc += q_val * k_val
                # Convert to float32 and scale
                acc_float: "float32" = acc
                acc_float = acc_float / scale
                # Track max for this row
                if j1 == 0:
                    max_vals[p] = acc_float
                else:
                    if acc_float > max_vals[p]:
                        max_vals[p] = acc_float
                attn_rows[p, j1] = acc_float
        
        # ===== Stage 2: Compute exp and sum for P rows =====
        # Initialize sum_exps
        for p_init in allo.grid(P, name="init_sum"):
            sum_exps[p_init] = 0.0
        
        # P is outer, j2 is inner (pipelined)
        # Each row has its own sum_exp accumulator
        for j2 in allo.grid(L, name="exp_p"):
            for p2 in allo.grid(P, name="exp_j"):
                exp_val: "float32" = allo.exp(attn_rows[p2, j2] - max_vals[p2])
                softmax_rows[p2, j2] = exp_val
                sum_exps[p2] += exp_val

        # ===== Stage 3: Normalize and scale softmax rows =====
        # P is outer, j3 is inner (pipelined)
        softmax_scale: "float32" = 32768.0
        for p3 in allo.grid(P, name="norm_p"):
            for j3 in allo.grid(L, name="norm_j"):
                norm_val: "float32" = softmax_rows[p3, j3] / sum_exps[p3]
                softmax_scaled: "int16" = norm_val * softmax_scale
                softmax_rows_int[p3, j3] = softmax_scaled
                
        # ===== Stage 4: Initialize output accumulators =====
        for p_init in allo.grid(P, name="init_p"):
            for d_init in allo.grid(D_h, name="init_d"):
                acc_out[p_init, d_init] = 0
        
        # ===== Stage 5: Compute output rows with integer arithmetic =====
        # P is outer, j4 is inner (pipelined), d is innermost
        for p4 in allo.grid(P, name="out_p"):
            for j4 in allo.grid(L, name="out_j"):
                s_val: "int32" = softmax_rows_int[p4, j4]
                for d in allo.grid(D_h, name="out_d"):
                    v_val: "int32" = V[j4, d]
                    acc_out[p4, d] += s_val * v_val
        
        # ===== Stage 6: Write outputs - rescale from fixed-point =====
        for p5 in allo.grid(P, name="write_p"):
            i_out: "int16" = i_outer * P + p5
            for d2 in allo.grid(D_h, name="write_d"):
                rescaled: "int32" = acc_out[p5, d2] >> 15
                out_val: T = rescaled
                out[i_out, d2] = out_val



def self_attention_and_mlp[
    T: (bfloat16, float32, int4, int8),
    L: int16,
    D_h: int16,
    D_mlp: int16,
    P: int16  # Parallelism factor (4)
](
    X: "T[L, D_h]",
    W_q: "T[D_h, D_h]",
    W_k: "T[D_h, D_h]",
    W_v: "T[D_h, D_h]",
    W_fc1: "T[D_mlp, D_h]",   # MLP fc1 weight [3072, 768]
    b_fc1: "T[D_mlp]",        # MLP fc1 bias [3072]
    W_fc2: "T[D_h, D_mlp]",   # MLP fc2 weight [768, 3072]
    b_fc2: "T[D_h]",          # MLP fc2 bias [768]
    scale: "float32",
    out: "T[L, D_h]"
):
    """
    Self-attention with integrated QKV projection and MLP.
    
    Computes Q, K, V from input X using projection matrices W_q, W_k, W_v,
    then applies P-row parallel streaming SDPA. The attention output (delta)
    is added to the input X (residual connection), then fed through the MLP.
    
    MLP structure: fc1 (768->3072) with ReLU, then fc2 (3072->768)
    
    Structure:
    - QKV Projection: Compute Q=X@W_q, K=X@W_k, V=X@W_v
    - SDPA: Scaled dot-product attention -> delta
    - Residual: X + delta
    - MLP: fc1(ReLU) -> fc2
    """
    # ===== QKV Projection Stage =====
    Q: "T[L, D_h]" = 0.0
    K: "T[L, D_h]" = 0.0
    V: "T[L, D_h]" = 0.0
    
    # ===== QKV Projection (manual matmul-transpose) =====
    # Compute Q = X @ W_q  where W_q is stored as [D_q, D_h]
    # mm_transpose semantics: out[i, j] = sum_k X[i,k] * W_q[j,k]
    for i in allo.grid(L, name="q_i"):
        for j in allo.grid(D_h, name="q_j"):
            acc_q: "T" = 0.0
            for k in allo.grid(D_h, name="q_k"):
                acc_q += X[i, k] * W_q[j, k]
            Q[i, j] = acc_q

    # Compute K = X @ W_k
    for i in allo.grid(L, name="k_i"):
        for j in allo.grid(D_h, name="k_j"):
            acc_k: "T" = 0.0
            for k in allo.grid(D_h, name="k_k"):
                acc_k += X[i, k] * W_k[j, k]
            K[i, j] = acc_k

    # Compute V = X @ W_v
    for i in allo.grid(L, name="v_i"):
        for j in allo.grid(D_h, name="v_j"):
            acc_v: "T" = 0.0
            for k in allo.grid(D_h, name="v_k"):
                acc_v += X[i, k] * W_v[j, k]
            V[i, j] = acc_v

    # ===== SDPA Stage: Process P rows at a time =====
    for i_outer in allo.grid(L // P, name="row_outer"):
        # Row buffers for P rows - each row has its own buffers
        attn_rows: "float32[P, L]"      # P rows of attention scores
        softmax_rows: "float32[P, L]"   # P rows after softmax
        softmax_rows_int: "int16[P, L]" # Scaled softmax for integer MAC
        max_vals: "float32[P]"          # Max value per row
        sum_exps: "float32[P]"          # Sum of exp per row
        acc_out: "int32[P, D_h]"        # Output accumulators for P rows
        # ===== Stage 1: Compute P rows of Q @ K^T =====
        # P is outer, j1 is inner (pipelined)
        for p in allo.grid(P, name="mm_p"):
            i: "int16" = i_outer * P + p
            for j1 in allo.grid(L, name="mm_j"):
                # Dot product Q[i,:] @ K[j1,:]
                acc: "int32" = 0
                for k1 in allo.grid(D_h, name="mm_k"):
                    q_val: "int32" = Q[i, k1]
                    k_val: "int32" = K[j1, k1]
                    acc += q_val * k_val
                # Convert to float32 and scale
                acc_float: "float32" = acc
                acc_float = acc_float / scale
                # Track max for this row
                if j1 == 0:
                    max_vals[p] = acc_float
                else:
                    if acc_float > max_vals[p]:
                        max_vals[p] = acc_float
                attn_rows[p, j1] = acc_float
        
        # ===== Stage 2: Compute exp and sum for P rows =====
        # Initialize sum_exps
        for p_init in allo.grid(P, name="init_sum"):
            sum_exps[p_init] = 0.0
        
        # P is outer, j2 is inner (pipelined)
        # Each row has its own sum_exp accumulator
        for j2 in allo.grid(L, name="exp_p"):
            for p2 in allo.grid(P, name="exp_j"):
                exp_val: "float32" = allo.exp(attn_rows[p2, j2] - max_vals[p2])
                softmax_rows[p2, j2] = exp_val
                sum_exps[p2] += exp_val

        # ===== Stage 3: Normalize and scale softmax rows =====
        # P is outer, j3 is inner (pipelined)
        softmax_scale: "float32" = 32768.0
        for p3 in allo.grid(P, name="norm_p"):
            for j3 in allo.grid(L, name="norm_j"):
                norm_val: "float32" = softmax_rows[p3, j3] / sum_exps[p3]
                softmax_scaled: "int16" = norm_val * softmax_scale
                softmax_rows_int[p3, j3] = softmax_scaled
                
        # ===== Stage 4: Initialize output accumulators =====
        for p_init in allo.grid(P, name="init_p"):
            for d_init in allo.grid(D_h, name="init_d"):
                acc_out[p_init, d_init] = 0
        
        # ===== Stage 5: Compute output rows with integer arithmetic =====
        # P is outer, j4 is inner (pipelined), d is innermost
        for p4 in allo.grid(P, name="out_p"):
            for j4 in allo.grid(L, name="out_j"):
                s_val: "int32" = softmax_rows_int[p4, j4]
                for d in allo.grid(D_h, name="out_d"):
                    v_val: "int32" = V[j4, d]
                    acc_out[p4, d] += s_val * v_val
        
        # ===== Stage 6: Write attention output to delta buffer =====
        delta: "T[P, D_h]"
        for p5 in allo.grid(P, name="write_p"):
            for d2 in allo.grid(D_h, name="write_d"):
                rescaled: "int32" = acc_out[p5, d2] >> 15
                delta[p5, d2] = rescaled
        
        # ===== Stage 7: Residual connection - add delta to input X =====
        residual: "T[P, D_h]"
        for p6 in allo.grid(P, name="res_p"):
            i_res: "int16" = i_outer * P + p6
            for d3 in allo.grid(D_h, name="res_d"):
                residual[p6, d3] = X[i_res, d3] + delta[p6, d3]
        
        # ===== Stage 8: MLP fc1 broken into three kernels =====
        # 1) Matmul: hidden = residual @ W_fc1^T  (we pass W_fc1 as transposed layout)
        # 2) Bias add: hidden += b_fc1
        # 3) Activation: ReLU(hidden)
        hidden: "T[P, D_mlp]"
        for p7, m in allo.grid(P, D_mlp, name="fc1_pm"):
            acc_fc1: "int32" = 0
            for d4 in allo.reduction(D_h, name="fc1_d"):
                acc_fc1 += residual[p7, d4] * W_fc1[m, d4]
            hidden[p7, m] = acc_fc1

        # --- Bias add: hidden += b_fc1
        for p7 in allo.grid(P, name="fc1_p"):
            for m in allo.grid(D_mlp, name="fc1_m"):
                hidden[p7, m] = hidden[p7, m] + b_fc1[m]

        # --- Activation: ReLU in-place
        for p7 in allo.grid(P, name="fc1_p"):
            for m in allo.grid(D_mlp, name="fc1_m"):
                val: "T" = hidden[p7, m]
                if val < 0:
                    hidden[p7, m] = 0

        # ===== Stage 9: MLP fc2 broken into two kernels + writeback =====
        # 1) Matmul: out_block = hidden @ W_fc2^T
        # 2) Bias add: out_block += b_fc2
        out_block: "T[P, D_h]" = 0
        # --- Matmul: out_block[p, d] = sum_{m} hidden[p, m] * W_fc2[d, m]
        for p8, d5 in allo.grid(P, D_h, name="fc2_pd"):
            acc_fc2: "int32" = 0
            for m2 in allo.reduction(D_mlp, name="fc2_m"):
                acc_fc2 += hidden[p8, m2] * W_fc2[d5, m2]
            out_block[p8, d5] = acc_fc2

        # --- Bias add: out_block += b_fc2
        for p8 in allo.grid(P, name="fc2_p"):
            for d5 in allo.grid(D_h, name="fc2_d"):
                out_block[p8, d5] = out_block[p8, d5] + b_fc2[d5]

        # Write final outputs
        for p8 in allo.grid(P, name="fc2_p"):
            i_out: "int16" = i_outer * P + p8
            for d5 in allo.grid(D_h, name="fc2_d"):
                out[i_out, d5] = out_block[p8, d5]
