import allo
from allo.ir.types import float32


def gemm[M, K, N](A: float32[M, K], B: float32[K, N]) -> float32[M, N]:
    """Generic GEMM: C = A * B with reduction over K.

    Template parameters: M, K, N
    """
    C: float32[M, N] = 0
    for i, j in allo.grid(M, N):
        for k in allo.reduction(K):
            C[i, j] += A[i, k] * B[k, j]
    return C


def add_bias[M, H](X: float32[M, H], b: float32[H]) -> float32[M, H]:
    Y: float32[M, H] = 0
    for i, j in allo.grid(M, H):
        Y[i, j] = X[i, j] + b[j]
    return Y


def gelu_approx[M, H](X: float32[M, H]) -> float32[M, H]:
    """Approximate GELU using tanh formulation (template over M,H)."""
    Y: float32[M, H] = 0
    for i, j in allo.grid(M, H):
        x = X[i, j]
        # Direct inline computation to match MLIR generation
        y = 0.5 * x * (1.0 + allo.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
        Y[i, j] = y
    return Y


__all__ = ["gemm", "add_bias", "gelu_approx"]
