import allo
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
import numpy as np
from datetime import datetime
from output_projection import matmul_output_projection 
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from attention.config import CrossAttentionConfig as CAC


def schedule_baseline_randomized_matmul_two(
    N_T: np.dtype, 
    A_T: allo.ir.types,
    mode: str = "llvm"
):
    """
    Schedule the baseline randomized matmul two operation.
    Input matrix Q is of shape (A_L, H_D) and K is of shape (V_L, H_D)
    """
    type_str = str(str(N_T).split(".")[-1])[:-2]
    A_L = CAC.LENGTH_OF_ACTION_CHUNK #Action length
    A_D = CAC.ACTION_HIDDEN_SIZE #Action output dimension
    V_D = CAC.Q_PROJ_OUT_DIM #This is the VLM output dimension
    Z = np.random.randn(A_L, V_D).astype(N_T) #input matrix Z is of shape (A_L, V_D) (output of matmul two and fused between all the heads)
    O = np.random.randn(V_D, A_D).astype(N_T) #input matrix O is of shape (V_D, A_D) (output projection matrix)
    Z_NEW = np.zeros((A_L, A_D), dtype=N_T) #output matrix is of shape (A_L, A_D) (output of output projection) 
    s = allo.customize(matmul_output_projection, instantiate=[A_T, A_L, V_D, A_D])
    name = f"output_projection_{A_L}_{V_D}_{A_D}_{type_str}_{mode}_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.prj"
    if mode == "llvm":
        s_llvm = s.build()
        s_llvm(Z, O, Z_NEW)
        return Z, s
    elif mode == "csyn":
        s_csyn = s.build(target="vitis_hls", mode="csyn", project=name)
        s_csyn()
    elif mode == "hw_emu":
        s_hw_emu = s.build(target="vitis_hls", mode="hw_emu", project=name)
    elif mode == "hw":
        s_hw = s.build(target="vitis_hls", mode="hw", project=name)
    elif mode == "sw_emu":
        s_sw_emu = s.build(target="vitis_hls", mode="sw_emu", project=name)
    return None, s

if __name__ == "__main__":
    A, s = schedule_baseline_randomized_matmul_two(N_T=np.float32, A_T=float32, mode="csyn")
    print(A)