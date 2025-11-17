import allo, math
from allo.ir.types import int8, int16, float32, bfloat16, int32
import numpy as np
import torch

print("we did't fail our example!")


L_VLM = 241 #241 input tokens into action axpert if we have 3 64-dim and 48 token text encoder and 1 action token (I think)
N = 50 #number of action tokens
H = 12 #number of heads
Q_H = 3 #number of heads Q shares with KV
A_D = 720 #action expert dimension
V_D = 320 #VLM output dimension
Q_I_D = V_D*Q_H #input dimension for Q shared over Q_H heads
Q_I_D_H = Q_I_D//H #input dimension per head for Q
V_D_H = V_D//H #dimension per head for K and V


def mm1[
    T: (int32, float32), # type: ignore
    P: int16, # type: ignore
    Q: int16, # type: ignore
    R: int16 # type: ignore
](
    A: "T[P, Q]", 
    B: "T[Q, R]", 
    out_AB: "T[P, R]"
):
    """
    Matrix multiplication.
    Computes out_AB = A @ B
    """
    for i0, j0 in allo.grid(P, R, name="mm1"):
        for k0 in allo.reduction(Q):
            out_AB[i0, j0] += A[i0, k0] * B[k0, j0]

def mm_transpose[
    T: (bfloat16, float32), 
    P: int16, 
    Q: int16, 
    R: int16 
](
    A: "T[P, Q]", 
    B: "T[R, Q]", 
    out_AB: "T[P, R]"
):
    """
    Matrix multiplication where B is transposed.
    Computes out_AB = A @ B^T
    """
    for i0, j0 in allo.grid(P, R, name="mm_transpose"):
        for k0 in allo.reduction(Q):
            out_AB[i0, j0] += A[i0, k0] * B[j0, k0]


def qkv_projection[
    T: (bfloat16, float32), 
    L_A: int16, 
    D_A: int16, 
    L_V: int16, 
    D_V: int16, 
    D_Q: int16
](
    A: "T[L_A, D_A]", 
    X: "T[L_V, D_V]", 
    W_q: "T[D_Q, D_A]", 
    W_k: "T[D_V, D_V]", 
    W_V: "T[D_V, D_V]", 
)-> "T[L_A, D_Q]":
    Q_n: "T[L_A, D_Q]" = 0.0
    K_n: "T[L_V, D_V]" = 0.0
    V_n: "T[L_V, D_V]" = 0.0
    mm_transpose[T, L_A, D_A, D_Q](A, W_q, Q_n)
    mm_transpose[T, L_V, D_V, D_V](X, W_k, K_n)
    mm_transpose[T, L_V, D_V, D_V](X, W_V, V_n)
    return Q_n

def q_projection[
    T: (bfloat16, float32),
    L_A: int16,
    D_A: int16,
    D_Q: int16
](
    A: "T[L_A, D_A]",
    W_q: "T[D_Q, D_A]",
    out_Q: "T[L_A, D_Q]"
)-> "T[L_A, D_Q]":
    """
    Q projection.
    Computes out_Q = A @ W_q
    """
    out_Q: "T[L_A, D_Q]" = 0.0
    mm_transpose[T, L_A, D_A, D_Q](A, W_q, out_Q)
    return out_Q

def k_projection[
    T: (bfloat16, float32),
    L_V: int16,
    D_V: int16,
    D_K: int16
](
    X: "T[L_V, D_V]",
    W_k: "T[D_K, D_V]",
    out_K: "T[L_V, D_K]"
)-> "T[L_V, D_K]":
    """
    K projection.
    Computes out_K = X @ W_k
    """
    out_K: "T[L_V, D_K]" = 0.0
    mm_transpose[T, L_V, D_V, D_K](X, W_k, out_K)
    return out_K

def v_projection[
    T: (bfloat16, float32),
    L_V: int16,
    D_V: int16,
    D_V: int16
](
    X: "T[L_V, D_V]",
    W_v: "T[D_V, D_V]",
    out_V: "T[L_V, D_V]"
)-> "T[L_V, D_V]":
    """
    V projection.
    Computes out_V = X @ W_v
    """
    out_V: "T[L_V, D_V]" = 0.0
    mm_transpose[T, L_V, D_V, D_V](X, W_v, out_V)
    return out_V

    
A_n = np.random.randn(N, A_D).astype(np.float32)
A_m = torch.from_numpy(A_n).to(dtype=torch.bfloat16) #is just so it doesn't have any of the names of things above

X_n = np.random.randn(1, L_VLM, V_D).astype(np.float32)
X_m = torch.from_numpy(X_n).to(dtype=torch.bfloat16)

W_q_n = np.random.randn(1, Q_I_D, V_D).astype(np.float32)
W_q_m = torch.from_numpy(W_q_n).to(dtype=torch.bfloat16)

W_k_n = np.random.randn(1, V_D, V_D).astype(np.float32)
W_k_m = torch.from_numpy(W_k_n).to(dtype=torch.bfloat16)

W_V_n = np.random.randn(1, V_D, V_D).astype(np.float32)
W_V_m = torch.from_numpy(W_V_n).to(dtype=torch.bfloat16)

Q_m = torch.zeros((N, Q_I_D), dtype=torch.bfloat16)
Q_n = np.zeros((N, Q_I_D), dtype=np.float32)
K_m = torch.zeros((L_VLM, V_D), dtype=torch.bfloat16)
K_n = np.zeros((L_VLM, V_D), dtype=np.float32)
V_m = torch.zeros((L_VLM, V_D), dtype=torch.bfloat16)
V_n = np.zeros((L_VLM, V_D), dtype=np.float32)
# s1 = allo.customize("")

# s_mm1 = allo.customize(mm1, instantiate=[bfloat16, N, V_D, Q_I_D])
# mod_mm1 = s_mm1.build()
# mod_mm1(A_m, W_q_m, Q_m)
# print(Q_m)
s = allo.customize(qkv_projection, instantiate=[float32, L_VLM, V_D, N, V_D, Q_I_D])
s_compiled = s.build(target="vitis_hls", mode="csyn", project="baseline_qkv_proj")
s_llvm = s.build()
q_n = s_llvm(A_n , X_n, W_q_n, W_k_n, W_V_n)
print("q_n:", q_n)
s_compiled()

# print("Q_n:", Q_n)
# print("K_n:", K_n)
# print("V_n:", V_n)
