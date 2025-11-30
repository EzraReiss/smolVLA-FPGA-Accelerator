import allo
import numpy as np
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from matrix_multiplies import mm_transpose


def numpy_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    

def sdpa_np(Q, K, V, d_h = 768 / 12):
    # Q: (M, N) queries
    # K: (P, N) keys
    # V: (P, D) values
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
    # Temporary buffer for attention scores: (M, P)
    B: "T[M, P]" = 0.0

    # Compute raw scores: B = Q @ K^T
    mm_transpose[T, L, D_h, D_h](Q, K, B)

    # Scale by divisor (e.g. sqrt(d_h))
    for i0, j0 in allo.grid(M, P):
        B[i0, j0] = B[i0, j0] / scale

    # Apply row-wise softmax over keys (each row has length P)
    softmax_baseline[T, L, D_h](B)

    # Final weighted sum: out = softmax(B) @ V
    mm1[T, L, D_h, D_h](B, V, out)
