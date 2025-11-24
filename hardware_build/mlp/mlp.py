import allo
from allo.ir.types import float32
from allo import dsl
import allo.backend.hls as hls
from allo.library.nn import linear2d, schedule_linear2d

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_kernels.kernels import gemm, add_bias

M = 3            # batch * seq (e.g. 1 * 3)
D_in = 768       # input feature dim
H = 3072         # hidden dim
D_out = 768      # output dim


def mlp_top(A: float32[M, D_in], W1: float32[D_in, H], b1: float32[H], W2: float32[H, D_out], b2: float32[D_out]) -> float32[M, D_out]:
    # FC1: (M x D_in) * (D_in x H) -> (M x H)
    C1 = gemm[M, D_in, H](A, W1)
    C1b = add_bias[M, H](C1, b1)
    
    # Apply GELU activation using DSL function inline
    gelu_out: float32[3, 3072] = 0.0
    for i, j in allo.grid(M, H):
        x = C1b[i, j]
        # Use tanh-based GELU approximation from dsl
        gelu_out[i, j] = 0.5 * x * (1.0 + dsl.tanh(0.7978845608028654 * (x + 0.044715 * dsl.power(x, 3.0))))

    # FC2: (M x H) * (H x D_out) -> (M x D_out)
    C2 = gemm[M, H, D_out](gelu_out, W2)
    
    # Add output bias (using direct operator since dsl.add is for NumPy)
    Out: float32[3, 768] = 0.0
    for i, j in allo.grid(M, D_out):
        Out[i, j] = C2[i, j] + b2[j]
    
    return Out


if __name__ == "__main__":
    # Create schedule for top-level function
    s = allo.customize(mlp_top)
    
    # Uncomment to see the full MLIR module:
    # print(s.module)

    # Build with Vitis HLS target and run the full flow
    print("Building HLS project...")
    mod = s.build(target="vitis_hls", mode="csyn", project="mlp.prj")
    mod()
    print("✅ MLP HLS project created and synthesis completed!")

    


