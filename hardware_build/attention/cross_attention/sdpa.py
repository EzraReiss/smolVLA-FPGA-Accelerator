import allo
import numpy as np
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
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
        
        # Compute exp and sum
        # Store exp values first, then accumulate (helps with II)
        sum_exp: "float32" = 0.0
        for j2 in allo.grid(L, name="exp_j"):
            exp_val: "float32" = allo.exp(attn_row[j2] - max_val)
            sum_exp += exp_val
            softmax_row[j2] = exp_val
        
        # Normalize and scale to fixed-point for int32 accumulation
        # Scale factor: 2^15 = 32768 (fits in int16, allows int32 accumulation without overflow)
        # Max accumulator value: 128 * 32768 * 127 ≈ 533M (fits in int32)
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

