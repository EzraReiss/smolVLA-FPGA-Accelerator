import allo
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
from allo.customize import Partition as partition
import numpy as np
from self_attention import self_attention as sa_2
import self_attention
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from attention.config import VLMAttentionConfig as VAC
import allo.library.systolic as sys

# Test configuration
L = 1024  # Sequence length
# L = 128 #Temp smaller L to make compile faster
D_h = 64  # Head dimension
H = 12  # Number of heads

    
def schedule_self_attention(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn"
):
    s = allo.customize(sa, instantiate=[A_T, L, H, D_h])
    loops = s.get_loops()
    outer_loop = loops["head_loop"]
    s.pipeline(outer_loop["j_precalc"])
    # s.split(outer_loop["i_out"], factor=2)
    loops = s.get_loops()
    outer_loop = loops["head_loop"]
    # print(outer_loop)    
    s.dataflow(outer_loop["i_out"])  # Dataflow over outer row batches
    # s.unfold("head_loop", [5])  # Unroll inner row batch loop
    # Pipeline the inner loops (same pattern as sdpa_streaming)
    # ===== Stage 1: Matmul Q @ K^T =====
    # Pipeline j1 (inner loop over L columns)
    loops = s.get_loops()
    outer_loop = loops["head_loop"]
    s.partition(s.W_q, partition.Cyclic, dim=3, factor=4)
    s.partition(s.V, partition.Cyclic, dim=3, factor=4)

    s.pipeline(outer_loop["j_attn"])  # Pipeline inner tiled loop
    s.pipeline(outer_loop["j_exp"])
    s.pipeline(outer_loop["j_exp_sum"])
    s.pipeline(outer_loop["j_norm"])
    s.pipeline(outer_loop["j_out"])
    s.pipeline(outer_loop["k_final"])
    dtype_str = {
        int4: "int4",
        int8: "int8",
        float32: "float32",
        bfloat16: "bfloat16"
    }[A_T]
    project_name = f"self_attention_sam_{dtype_str}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    s.build(target="vitis_hls", mode="csyn", project=project_name)()
    

def schedule_self_attention_row_parallelism(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn",
    should_return=False
):
    P = 4
    P_s = 4
    p_2 = 2

    s = allo.customize(sa_2, instantiate=[A_T, L, H, H*D_h, D_h, P, P_s])
    loops = s.get_loops()
    outer_loop = loops["head_loop"]
    # s.split(outer_loop["j_precalc"], factor=p_2)
    loops = s.get_loops()
    outer_loop = loops["head_loop"]
    s.pipeline(outer_loop["k_precalc"])
    loops = s.get_loops()
    outer_loop = loops["head_loop"]
    print(outer_loop)    
    s.dataflow(outer_loop["h1"])
    s.dataflow(outer_loop["i_out"])  # Dataflow over outer row batches
    # s.unfold("head_loop", [5])  # Unroll inner row batch loop
    # Pipeline the inner loops (same pattern as sdpa_streaming)
    # ===== Stage 1: Matmul Q @ K^T =====
    # Pipeline j1 (inner loop over L columns)
    loops = s.get_loops()
    outer_loop = loops["head_loop"]
    print(outer_loop)
    s.partition(s.W_q, partition.Cyclic, dim=3, factor=4)
    s.partition(s.V, partition.Cyclic, dim=3, factor=4)
    # s.partition(s.sum_exps_p, partition.Complete, dim=0)

    s.pipeline(outer_loop["j_attn"])  # Pipeline inner tiled loop
    s.pipeline(outer_loop["j_exp_P_s"])
    s.pipeline(outer_loop["j_norm"])
    s.pipeline(outer_loop["j_out"])
    s.pipeline(outer_loop["i_final"])
    if should_return:
        return s
    dtype_str = {
        int4: "int4",
        int8: "int8",
        float32: "float32",
        bfloat16: "bfloat16"
    }[A_T]
    project_name = f"self_attention_rp_{P}_{dtype_str}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"

    s.build(target="vitis_hls", mode="csyn", project=project_name)()

def schedule_layer_norm(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn"
):
    s = allo.customize(self_attention.layer_norm, instantiate=[A_T, L, D_h*H])
    loops = s.get_loops()
    # print(loops)
    s.pipeline(loops["ln_stats_loop"]["i_stat"])
    s.pipeline(loops["ln_out_outer"]["j_out"])
    s.pipeline(loops["ln_inner_outer"]["j_sum"])
    return s

def schedule_full_attention(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn"
):
    P = 2
    P_s = 4
    p_2 = 2
    s = allo.customize(self_attention.self_attention_return, instantiate=[A_T, L, H, H*D_h, D_h, P, P_s])
    s1 = schedule_layer_norm(N_T, A_T, mode)
    s2 = schedule_self_attention_row_parallelism(N_T, A_T, mode, should_return=True)
    # s.dataflow("self_attention_return")
    s.compose(s1, id="layer_norm2")
    s.compose(s2, id="sa1")
    dtype_str = {
        int4: "int4",
        int8: "int8",
        float32: "float32",
        bfloat16: "bfloat16"
    }[A_T]
    project_name = f"full_self_attention_{P}_{dtype_str}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    print(s.module)
    mod = s.build(target="vitis_hls", mode="csyn", project=project_name)()

if __name__ == "__main__":
    schedule_full_attention(np.int8, int8, mode="csyn")
    # s1 = schedule_self_attention_row_parallelism(np.int8, int8, mode="csyn", should_return=True)
