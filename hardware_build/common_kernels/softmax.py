import allo
from allo.ir.types import index, int4, int8, int16, int32, float32, bfloat16
import numpy as np, math

def np_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def softmax_baseline[
    T: (float32, bfloat16),
    L: int16,
    D: int16
](
    A: "T[L, D]",
):
    
    for i0 in allo.grid(L): #loop through each row
        max_val = A[i0, 0]   
        for j0 in allo.grid(D):
            if A[i0, j0] > max_val:
                max_val = A[i0, j0]
        for j1 in allo.grid(D):
            A[i0, j1] = A[i0, j1] - max_val

        sum_exp_A = 0.0
        for j2 in allo.grid(D):
            A[i0, j2] = allo.exp(A[i0, j2])
            sum_exp_A += A[i0, j2]
        for j3 in allo.grid(D):
            A[i0, j3] = A[i0, j3] / sum_exp_A

if __name__ == "__main__":
    A = np.random.rand(10, 10).astype(np.float32)
    A2 = A.copy()
    out_A = np.zeros_like(A)
    s = allo.customize(softmax, instantiate=[float32, 10, 10])
    # s.build(target="vitis_hls", mode="llvm", project="softmax.prj")
    s.build()(A)

    print(np.allclose(A, np_softmax(A2, axis=1)))