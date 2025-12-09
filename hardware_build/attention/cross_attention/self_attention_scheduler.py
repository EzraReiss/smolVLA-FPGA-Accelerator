import allo
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
from allo.customize import Partition as partition
import numpy as np
from self_attention import self_attention_2 as sa
from self_attention import self_attention_3 as sa_2
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
    print(outer_loop)    
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
    mode: str = "csyn"
):
    U = 2
    P = 8*U
    s = allo.customize(sa_2, instantiate=[A_T, L, H, D_h, P])
    loops = s.get_loops()
    outer_loop = loops["head_loop"]
    s.pipeline(outer_loop["j_precalc"])
    # s.split(outer_loop["i_out"], factor=2)
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
    project_name = f"self_attention_rp_{U}_{dtype_str}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    s.build(target="vitis_hls", mode="csyn", project=project_name)()

if __name__ == "__main__":
    # schedule_self_attention(np.int8, int8, mode="csyn")
    schedule_self_attention_row_parallelism(np.int8, int8, mode="csyn")
