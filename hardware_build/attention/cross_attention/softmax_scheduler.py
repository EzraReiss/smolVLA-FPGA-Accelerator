import allo
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
import numpy as np
import softmax 
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from attention.config import CrossAttentionConfig as CAC




def schedule_baseline_randomized_softmax(
    N_T: np.dtype, 
    A_T: allo.ir.types,
    mode: str = "llvm"
):
    """
    Schedule the baseline randomized softmax operation.
    Input matrix A is of shape (L, D)
    L --> length of action chunk
    D --> dimension of action chunk
    """
    type_str = str(str(N_T).split(".")[-1])[:-2]
    L_A = CAC.LENGTH_OF_ACTION_CHUNK
    L_V = CAC.DEFAULT_Tf
    A = np.random.randn(L_A, L_V).astype(N_T)
    s = allo.customize(softmax.softmax_baseline, instantiate=[A_T, L_A, L_V])

    name = f"softmax_{L_A}_{L_V}_{type_str}_{mode}_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.prj"
    if mode == "llvm":
        s_llvm = s.build()
        s_llvm(A)
        return A, s
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
    A, s = schedule_baseline_randomized_softmax(N_T=np.float32, A_T=float32, mode="csyn")
    print(A)