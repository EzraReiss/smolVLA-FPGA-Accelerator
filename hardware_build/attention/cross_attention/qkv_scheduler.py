import allo
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
import numpy as np
import qkv_projection as qkv 
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from attention.config import CrossAttentionConfig as CAC

#should not need these and just config files but leave for now
L_VLM = 241 #241 input tokens into action axpert if we have 3 64-dim and 48 token text encoder and 1 action token (I think)
N = 50 #number of action tokens
H = 12 #number of heads
Q_H = 3 #number of heads Q shares with KV
A_D = 720 #action expert dimension
V_D = 320 #VLM output dimension
Q_I_D = V_D*Q_H #input dimension for Q shared over Q_H heads
Q_I_D_H = Q_I_D//H #input dimension per head for Q
V_D_H = V_D//H #dimension per head for K and V



def schedule_randomized_baseline_q_projection(
    N_T: np.dtype, 
    A_T: allo.ir.types,
    mode: str = "llvm"
):
    """

    N_T: numpy dtype
        - float32
        - bfloat16 (not yet supported)
        - int32
        - int16
        - int8
        - int4
    
    A_T: Allo dtype
        - float32
        - bfloat16
        - int32
        - int16
        - int8
        - int4

    mode: The compilation mode to use
        - "llvm": LLVM backend
        - "csyn": hardware synthesis emulation using Vitis HLS (performance estimation)
        - "hw_emu": hardware emulation using Vitis HLS (performance estimation)
        - "hw": hardware implementation using Vitis HLS (performance estimation)
        - "sw_emu": software emulation and actually run tests

    Baseline schedule for Q projection.
    Computes out_Q = A @ W_q

    Dimensions:
    A: (L_A, D_A) - action chunk (length of action chunk, dimension of action chunk)
    W_q: (D_Q, D_A) - query projection matrix (dimension of query, dimension of action chunk)
    out_Q: (L_A, D_Q) - query output (length of action chunk, dimension of query)
    """

    type_str = str(str(N_T).split(".")[-1])[:-2]
    
    L_A = CAC.LENGTH_OF_ACTION_CHUNK
    D_A = CAC.ACTION_HIDDEN_SIZE
    D_Q = CAC.Q_PROJ_OUT_DIM

    A = np.random.randn(L_A, D_A).astype(N_T)
    W_q = np.random.randn(D_Q, D_A).astype(N_T)
    out_Q = np.zeros((L_A, D_Q), dtype=N_T)
    s = allo.customize(qkv.q_projection, instantiate=[A_T, L_A, D_A, D_Q])

    if mode == "llvm":
        s_llvm = s.build()
        s_llvm(A, W_q, out_Q)
        return (out_Q, s)
    elif mode == "csyn":
        s_csyn = s.build(target="vitis_hls", mode="csyn", project=f"projection_q_{L_A}_{D_A}_{D_Q}_{type_str}_{mode}_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.prj")
        s_csyn()
    elif mode == "hw_emu":
        s_hw_emu = s.build(target="vitis_hls", mode="hw_emu", project=f"projection_q_{L_A}_{D_A}_{D_Q}_{type_str}_{mode}_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.prj")
    elif mode == "hw":
        s_hw = s.build(target="vitis_hls", mode="hw", project=f"projection_q_{L_A}_{D_A}_{D_Q}_{type_str}_{mode}_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.prj")
    elif mode == "sw_emu":
        s_sw_emu = s.build(target="vitis_hls", mode="sw_emu", project=f"projection_q_{L_A}_{D_A}_{D_Q}_{type_str}_{mode}_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.prj")
    return None, s

def schedule_randomized_baseline_kv_projection(
    N_T: np.dtype, 
    A_T: allo.ir.types,
    mode: str = "llvm"
):
    """

    N_T: numpy dtype
        - float32
        - bfloat16 (not yet supported)
        - int32
        - int16
        - int8
        - int4
    
    A_T: Allo dtype
        - float32
        - bfloat16
        - int32
        - int16
        - int8
        - int4

    mode: The compilation mode to use
        - "llvm": LLVM backend
        - "csyn": hardware synthesis emulation using Vitis HLS (performance estimation)
        - "hw_emu": hardware emulation using Vitis HLS (performance estimation)
        - "hw": hardware implementation using Vitis HLS (performance estimation)
        - "sw_emu": software emulation and actually run tests
    """

    type_str = str(str(N_T).split(".")[-1])[:-2]
    
    L_V = CAC.DEFAULT_Tf
    H_D = CAC.ACTION_HIDDEN_SIZE
    D_K = CAC.KV_DIM
    D_V = CAC.KV_DIM

    X = np.random.randn(L_V, H_D).astype(N_T)
    W_k = np.random.randn(D_K, H_D).astype(N_T)
    W_v = np.random.randn(D_V, H_D).astype(N_T)
    out_K = np.zeros((L_V, D_K), dtype=N_T)
    out_V = np.zeros((L_V, D_V), dtype=N_T)

    s0 = allo.customize(qkv.k_projection, instantiate=[A_T, L_V, H_D, D_K])
    s1 = allo.customize(qkv.v_projection, instantiate=[A_T, L_V, H_D, D_V])

    if mode == "llvm":
        s0_llvm = s0.build()
        s1_llvm = s1.build()
        s0_llvm(X, W_k, out_K)
        s1_llvm(X, W_v, out_V)
        return (out_K, out_V, s0, s1)
    elif mode == "csyn":
        s0_csyn = s0.build(target="vitis_hls", mode="csyn", project=f"projection_k_{L_V}_{H_D}_{D_K}_{type_str}_{mode}_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.prj")()
        s1_csyn = s1.build(target="vitis_hls", mode="csyn", project=f"projection_v_{L_V}_{H_D}_{D_V}_{type_str}_{mode}_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.prj")()
    elif mode == "hw_emu":
        s0_hw_emu = s0.build(target="vitis_hls", mode="hw_emu", project=f"projection_k_{L_V}_{H_D}_{D_K}_{type_str}_{mode}_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.prj")
        s1_hw_emu = s1.build(target="vitis_hls", mode="hw_emu", project=f"projection_v_{L_V}_{H_D}_{D_V}_{type_str}_{mode}_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.prj")
    elif mode == "hw":
        s0_hw = s0.build(target="vitis_hls", mode="hw", project=f"projection_k_{L_V}_{H_D}_{D_K}_{type_str}_{mode}_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.prj")
        s1_hw = s1.build(target="vitis_hls", mode="hw", project=f"projection_v_{L_V}_{H_D}_{D_V}_{type_str}_{mode}_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.prj")
    elif mode == "sw_emu":
        s0_sw_emu = s0.build(target="vitis_hls", mode="sw_emu", project=f"projection_k_{L_V}_{H_D}_{D_K}_{type_str}_{mode}_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.prj")
        s1_sw_emu = s1.build(target="vitis_hls", mode="sw_emu", project=f"projection_v_{L_V}_{H_D}_{D_V}_{type_str}_{mode}_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.prj")
    return None, None, s0, s1



if __name__ == "__main__":
    out_Q, s_q = schedule_randomized_baseline_q_projection(N_T=np.float32, A_T=float32, mode="csyn")
    out_K, out_V, s_k, s_v = schedule_randomized_baseline_kv_projection(N_T=np.float32, A_T=float32, mode="csyn")
    print(f"out_Q: {out_Q}")
    print(f"out_K: {out_K}")
    print(f"out_V: {out_V}")
