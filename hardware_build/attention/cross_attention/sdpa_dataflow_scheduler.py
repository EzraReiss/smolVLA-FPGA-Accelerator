import allo
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4
from allo.customize import Partition as partition
import numpy as np
import sdpa
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from attention.config import VLMAttentionConfig as VAC

# Test configuration
L = 1024  # Sequence length
# L = 128 #Temp smaller L to make compile faster
D_h = 64  # Head dimension

# Test data
Q = np.random.rand(L, D_h).astype(np.float32)
K = np.random.rand(L, D_h).astype(np.float32)
V = np.random.rand(L, D_h).astype(np.float32)
scale = float(np.sqrt(D_h))


def schedule_sdpa_no_dataflow(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn"
):
    """
    True baseline single-head SDPA with NO dataflow optimization.
    Uses standard sdpa function for comparison against dataflow versions.
    """
    s = allo.customize(sdpa.sdpa, instantiate=[A_T, L, D_h])
    
    # No optimizations - pure baseline
    
    project_name = f"sdpa_single_head_no_dataflow_{mode}_{A_T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            out = np.zeros((L, D_h), dtype=N_T)
            s_llvm = s.build(project=project_name)
            s_llvm(Q, K, V, scale, out)
            return out, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


def schedule_sdpa_dataflow_baseline(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn"
):
    """
    Baseline single-head SDPA with dataflow optimization.
    Uses sdpa_dataflow which has all subfunctions return values
    for proper dataflow semantics.
    """
    s = allo.customize(sdpa.sdpa_dataflow, instantiate=[A_T, L, D_h])
    
    # Apply dataflow to the top-level function
    # This enables pipelining between:
    # mm_transpose_return -> scale -> softmax_return -> mm1_return
    s.dataflow("sdpa_dataflow")
    
    project_name = f"sdpa_single_head_dataflow_baseline_{mode}_{A_T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            out = np.zeros((L, D_h), dtype=N_T)
            s_llvm = s.build(project=project_name)
            out = s_llvm(Q, K, V, scale)
            return out, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


def schedule_sdpa_dataflow_inlined(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn"
):
    """
    Single-head SDPA with aggressive inlining inside dataflow.
    Inlines all subfunctions so HLS can see the full pipeline.
    """
    s = allo.customize(sdpa.sdpa_dataflow, instantiate=[A_T, L, D_h])
    
    # Inline all subfunctions for better optimization
    s.inline("mm_transpose_return")
    s.inline("softmax_return")
    s.inline("mm1_return")
    
    # Apply dataflow to enable streaming
    s.dataflow("sdpa_dataflow")
    
    project_name = f"sdpa_single_head_dataflow_inlined_{mode}_{A_T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            out = np.zeros((L, D_h), dtype=N_T)
            s_llvm = s.build(project=project_name)
            out = s_llvm(Q, K, V, scale)
            return out, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


def schedule_sdpa_dataflow_optimized(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn"
):
    """
    Fully optimized single-head SDPA with dataflow + array partitioning.
    Partitions arrays to enable parallel access within pipelined stages.
    """
    s = allo.customize(sdpa.sdpa_dataflow, instantiate=[A_T, L, D_h])
    
    # Inline all subfunctions
    s.inline("mm_transpose_return")
    s.inline("softmax_return")
    s.inline("mm1_return")
    
    # Partition input/output arrays for parallel access
    # Complete partition on dimension 2 (D_h=64) - small enough to fully partition
    s.partition(s.Q, partition.Complete, dim=2)
    s.partition(s.K, partition.Complete, dim=2)
    s.partition(s.V, partition.Complete, dim=2)
    
    # Apply dataflow
    s.dataflow("sdpa_dataflow")
    
    project_name = f"sdpa_single_head_dataflow_optimized_{mode}_{A_T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            out = np.zeros((L, D_h), dtype=N_T)
            s_llvm = s.build(project=project_name)
            out = s_llvm(Q, K, V, scale)
            return out, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


def schedule_sdpa_streaming_baseline(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn"
):
    """
    Row-streaming SDPA baseline.
    
    Processes one output row at a time, reducing intermediate storage from
    O(L^2) to O(L). For L=1024: 4MB -> 8KB = 512x BRAM reduction.
    
    No optimizations applied - pure baseline for streaming approach.
    """
    s = allo.customize(sdpa.sdpa_streaming, instantiate=[A_T, L, D_h])
    
    # No optimizations - baseline
    
    project_name = f"sdpa_streaming_baseline_{mode}_{A_T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            out = np.zeros((L, D_h), dtype=N_T)
            s_llvm = s.build(project=project_name)
            s_llvm(Q, K, V, scale, out)
            return out, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


def schedule_sdpa_streaming_pipelined(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn"
):
    """
    Row-streaming SDPA with inner loop pipelining.
    
    Pipelines the innermost loops for each stage:
    - mm_k: D_h iterations for dot product (II=1)
    - max_j, exp_j, norm_j: L iterations for softmax passes
    - out_j: L iterations for output dot product
    """
    s = allo.customize(sdpa.sdpa_streaming, instantiate=[A_T, L, D_h])
    
    # Pipeline the innermost loops
    print(s.get_loops()["row_loop"])
    s.pipeline(s.get_loops()["row_loop"]["j1"])      # Dot product accumulation
    s.pipeline(s.get_loops()["row_loop"]["j2"])     # Max finding
    s.pipeline(s.get_loops()["row_loop"]["j3"])     # Exp computation
    s.pipeline(s.get_loops()["row_loop"]["j4"])    # Normalization
    s.pipeline(s.get_loops()["row_loop"]["j5"])     # Output dot product
    
    project_name = f"sdpa_streaming_pipelined_{mode}_{A_T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            out = np.zeros((L, D_h), dtype=N_T)
            s_llvm = s.build(project=project_name)
            s_llvm(Q, K, V, scale, out)
            return out, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


def schedule_sdpa_streaming_optimized(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn"
):
    """
    Fully optimized row-streaming SDPA.
    
    Optimizations:
    1. Array partitioning on D_h dimension for parallel access
    2. Loop pipelining on inner loops
    3. Unroll inner dot product loops for maximum parallelism
    
    This should give significant speedup while maintaining low BRAM usage.
    """
    s = allo.customize(sdpa.sdpa_streaming, instantiate=[A_T, L, D_h])
    
    # Partition Q, K, V on D_h dimension for parallel access
    # Complete partition since D_h=64 is small
    s.partition(s.Q, partition.Complete, dim=2)
    s.partition(s.K, partition.Complete, dim=2)
    s.partition(s.V, partition.Complete, dim=2)
    
    # Partition row buffers for parallel access
    # s.partition(s.attn_row, partition.Block, factor=L//2, dim=1)
    # s.partition(s.softmax_row, partition.Block, factor=L//2, dim=1)
    
    # s.pipeline(s.get_loops()["row_loop"]["i"])  # Pipeline outer row loop
    # Pipeline the j loops (iterate over L dimension)
    # s.pipeline(s.get_loops()["row_loop"]["j1"])      # Outer loop of first matmul
    s.unroll(s.get_loops()["row_loop"]["j1"]) #unroll the entire row generation to produce single cycle row
    s.pipeline(s.get_loops()["row_loop"]["j2"])     # Max finding
    s.pipeline(s.get_loops()["row_loop"]["j3"])     # Exp computation  
    s.pipeline(s.get_loops()["row_loop"]["j4"])    # Normalization
    s.pipeline(s.get_loops()["row_loop"]["j5"])     # Output computation
    
    project_name = f"sdpa_streaming_optimized_{mode}_{A_T}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            out = np.zeros((L, D_h), dtype=N_T)
            s_llvm = s.build(project=project_name)
            s_llvm(Q, K, V, scale, out)
            return out, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


def schedule_sdpa_streaming_quantized_baseline(
    N_T: np.dtype,
    A_T: allo.ir.types,
    mode: str = "csyn"
):
    """
    Baseline quantized (int4/int8) row-streaming SDPA.
    
    No optimizations - pure baseline for quantized version.
    Supports int4 and int8 with mixed-precision compute:
    - Integer matmul with int32 accumulator
    - Float32 softmax
    - Quantized output
    """
    s = allo.customize(sdpa.sdpa_streaming, instantiate=[A_T, L, D_h])
    
    # No optimizations - baseline
    
    dtype_str = "int4" if A_T == int4 else "int8"
    project_name = f"sdpa_streaming_quantized_{dtype_str}_baseline_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            out = np.zeros((L, D_h), dtype=N_T)
            s_llvm = s.build(project=project_name)
            # Create quantized test data
            Q_quant = np.random.randint(-8, 8, (L, D_h)).astype(N_T)
            K_quant = np.random.randint(-8, 8, (L, D_h)).astype(N_T)
            V_quant = np.random.randint(-8, 8, (L, D_h)).astype(N_T)
            s_llvm(Q_quant, K_quant, V_quant, scale, out)
            return out, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


def schedule_sdpa_streaming_quantized_tiled(
    N_T: np.dtype,
    A_T: allo.ir.types,
    tile_factor: int = 16,
    mode: str = "csyn"
):
    """
    Optimized quantized (int4/int8) row-streaming SDPA with loop tiling.
    
    Key optimizations:
    1. Split outer matmul loop (mm_j) by tile_factor (default 16)
       - Processes tile_factor columns of the attention matrix at once
       - Goal: II=1 for the inner tiled loop
       - Throughput: Complete row in L/tile_factor cycles
    
    2. Array partitioning to support parallel access
       - Partition D_h dimension completely (64 elements)
       - Cyclic partition on tiled dimension for parallel column access
    
    3. Pipeline inner loops for maximum throughput
    
    For L=1024, tile_factor=16:
    - Each row takes ~64 cycles (1024/16) instead of 1024
    - 16× speedup on matmul stage
    """
    s = allo.customize(sdpa.sdpa_streaming, instantiate=[A_T, L, D_h])
    
    # Get loop hierarchy - loops are accessed by iterator variable names
    loops = s.get_loops()
    row_loop = loops["row_loop"]
    
    # ===== Stage 1: Tile the matmul (j1 loop) =====
    # j1 is the outer loop that iterates over columns of attention matrix
    # Split j1 into outer and inner loops
    # Outer: L/tile_factor iterations
    # Inner: tile_factor iterations (this gets pipelined to II=1)
    s.split(row_loop["j1"], factor=tile_factor)

    # After split, need to re-fetch loops to get the new split loops
    loops = s.get_loops()
    row_loop = loops["row_loop"]
    
    # The split creates j1.outer and j1.inner
    # Pipeline the inner tiled loop for II=1
    # This means we process tile_factor attention scores per cycle
    s.pipeline(row_loop["j1.inner"])
    
    # Pipeline the innermost dot product loop (k1)
    s.pipeline(row_loop["k1"])
    
    # ===== Stage 2: Array Partitioning =====
    # s.partition(s.Q, partition.Complete, dim=2)
    # s.partition(s.K, partition.Complete, dim=2)
    # s.partition(s.V, partition.Complete, dim=0)
    # s.partition(s.out, partition.Block, factor=64, dim=2)
    # s.partition(s.softmax_row, partition.Complete, dim=1)  #this partition breaks the norm loop 
    s.partition(s.acc_out, partition.Complete, dim=1)  

    
    # Cyclic partition attn_row and softmax_row by tile_factor
    # This allows parallel writes to tile_factor elements in the inner loop
    # s.partition(s.attn_row, partition.Cyclic, dim=1, factor=tile_factor)
    # s.partition(s.softmax_row, partition.Cyclic, dim=1, factor=tile_factor)
    
    
    # ===== Stage 3: Pipeline softmax passes =====
    # exp_j loop: computes exp, stores to softmax_row, and accumulates sum_exp
    # The accumulation has II=7 due to fadd latency, but this is acceptable:
    # - Total cycles: 128 * 7 = 896 cycles per row (vs 128*64=8192 for matmul)
    # - The exp() computation can overlap with the fadd pipeline
    s.pipeline(row_loop["j2"])       # exp_j loop. Not good with pipeline try unroll
    s.unroll(row_loop["j2"], factor=L//2)      # exp_j loop
    s.pipeline(row_loop["j3"])       # norm_j loop - normalize softmax
    
    # ===== Stage 4: Output computation =====
    # Kernel now has j4 outer (L=128), d inner (D_h=64)
    # Pipeline j4: each iteration does 64 parallel MACs (d gets unrolled)
    # Expected: II=7 (accumulator dependency), Depth=~7 (just fadd latency)
    # Much better than previous Depth=909 with 128-element reduction tree
    s.pipeline(row_loop["j4"])
    
    # Also pipeline the output write loop
    s.pipeline(row_loop["d2"])
    
    dtype_str = "int4" if A_T == int4 else "int8"
    project_name = f"sdpa_streaming_quantized_{dtype_str}_tiled_{tile_factor}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            out = np.zeros((L, D_h), dtype=N_T)
            s_llvm = s.build(project=project_name)
            # Create quantized test data
            Q_quant = np.random.randint(-8, 8, (L, D_h)).astype(N_T)
            K_quant = np.random.randint(-8, 8, (L, D_h)).astype(N_T)
            V_quant = np.random.randint(-8, 8, (L, D_h)).astype(N_T)
            s_llvm(Q_quant, K_quant, V_quant, scale, out)
            return out, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


def schedule_sdpa_streaming_4row_parallel(
    N_T: np.dtype,
    A_T: allo.ir.types,
    P: int = 8,  # Row parallelism factor (8 to hide fadd latency ~7)
    mode: str = "csyn"
):
    """
    P-row parallel streaming SDPA with P on the outside.
    
    Key insight: With P=8 rows processed per batch, we have 8 independent
    accumulator chains. The exp/sum loop runs sequentially per row, so each
    row's fadd chain is independent - no cross-iteration dependency within
    the pipelined inner loop.
    
    Loop structure:
    - row_outer: L//P iterations (e.g., 16 for L=128, P=8)
    - P loop: outer, processes each row in the batch
    - Inner loops (j1, j2, j3, j4): pipelined, same as sdpa_streaming
    
    This matches the structure of sdpa_streaming but processes P rows per batch.
    """
    s = allo.customize(sdpa.sdpa_streaming_8row, instantiate=[A_T, L, D_h, P])
    
    loops = s.get_loops()
    outer_loop = loops["row_outer"]    
    s.dataflow(outer_loop["i_outer"])  # Dataflow over outer row batches
    # Pipeline the inner loops (same pattern as sdpa_streaming)
    # ===== Stage 1: Matmul Q @ K^T =====
    # Pipeline j1 (inner loop over L columns)
    loops = s.get_loops()
    outer_loop = loops["row_outer"]

    s.pipeline(outer_loop["j1"])  # Pipeline inner tiled loop
    s.partition(s.acc_out, partition.Complete, dim=1)  
    s.partition(s.max_vals, partition.Complete, dim=1)
    
    # ===== Stage 2: Exp and sum =====
    # Each row has its own sum_exp accumulator, so no cross-row dependency
    s.pipeline(outer_loop["p2"])
    
    # ===== Stage 3: Normalize =====
    # Pipeline j3 (inner loop over L elements)
    s.pipeline(outer_loop["j3"])
    
    # ===== Stage 4: Output matmul =====
    # Pipeline j4 (inner loop over L softmax positions)
    s.pipeline(outer_loop["j4"])
    
    # ===== Stage 5: Write outputs =====
    # Pipeline d2 (inner loop over D_h)
    s.pipeline(outer_loop["d2"])
    dtype_str = "int4" if A_T == int4 else "int8"
    project_name = f"sdpa_streaming_{P}row_parallel_{dtype_str}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            out = np.zeros((L, D_h), dtype=N_T)
            s_llvm = s.build(project=project_name)
            Q_quant = np.random.randint(-8, 8, (L, D_h)).astype(N_T)
            K_quant = np.random.randint(-8, 8, (L, D_h)).astype(N_T)
            V_quant = np.random.randint(-8, 8, (L, D_h)).astype(N_T)
            s_llvm(Q_quant, K_quant, V_quant, scale, out)
            return out, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()
            return s, s


def schedule_self_attention_4row_parallel(
    N_T: np.dtype,
    A_T: allo.ir.types,
    P: int16 = 8,   # Row parallelism factor (8 to hide fadd latency ~7)
    mode: str = "csyn"
):
    """
    4-row parallel schedule for the `self_attention` kernel.

    This is a copy of the SDPA 4-row parallel scheduler adapted to
    the `self_attention` implementation. It applies the same partitioning,
    pipelining and dataflow pragmas but customizes `sdpa.self_attention`.
    """
    print(f"{A_T=}, {N_T=}, {D_h=}, {P=}, {mode=}")
    s = allo.customize(sdpa.schedule_self_attention_4row_parallel, instantiate=[A_T, L, D_h, P])
    
    loops = s.get_loops()
    outer_loop = loops["row_outer"]    
    s.dataflow(outer_loop["i_outer"])  # Dataflow over outer row batches
    s.unroll(outer_loop["i_outer"], factor=2)
    # Pipeline the inner loops (same pattern as sdpa_streaming)
    # ===== Stage 0: Pipeline & Partition QKV projection =====
    # Apply targeted pipelining to the inner k-loops of the manual
    # Q/K/V projection (q_k, k_k, v_k). Also partition the computed
    # Q/K/V and the weight matrices on the D_h dimension for parallel
    # access inside the projection and downstream matmuls.
    
    # Pipeline the innermost k-loops of Q/K/V projections
    # loops.q_i contains nested loops: i, j, k (the innermost is 'k')
    s.pipeline(loops.q_i.k)
    s.pipeline(loops.k_i.k)
    s.pipeline(loops.v_i.k)

       
    # ===== Stage 1: Matmul Q @ K^T =====
    # Pipeline j1 (inner loop over L columns)
    loops = s.get_loops()
    outer_loop = loops["row_outer"]

    s.pipeline(outer_loop["j1"])  # Pipeline inner tiled loop
    s.partition(s.acc_out, partition.Complete, dim=1)  
    # Partition the internally computed Q/K/V for parallel access
    s.partition(s.Q, partition.Complete, dim=2)
    s.partition(s.max_vals, partition.Complete, dim=1)
    
    # ===== Stage 2: Exp and sum =====
    # Each row has its own sum_exp accumulator, so no cross-row dependency
    s.pipeline(outer_loop["p2"])
    
    # ===== Stage 3: Normalize =====
    # Pipeline j3 (inner loop over L elements)
    s.pipeline(outer_loop["j3"])
    
    # ===== Stage 4: Output matmul =====
    # Pipeline j4 (inner loop over L softmax positions)
    s.pipeline(outer_loop["j4"])
    s.unroll(outer_loop["j4"])
    
    # ===== Stage 5: Write outputs =====
    # Pipeline d2 (inner loop over D_h)
    s.pipeline(outer_loop["d2"])
    
    dtype_str = "int4" if A_T == int4 else "int8"
    project_name = f"self_attention_{P}row_parallel_{dtype_str}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            out = np.zeros((L, D_h), dtype=N_T)
            s_llvm = s.build(project=project_name)
            Q_quant = np.random.randint(-8, 8, (L, D_h)).astype(N_T)
            K_quant = np.random.randint(-8, 8, (L, D_h)).astype(N_T)
            V_quant = np.random.randint(-8, 8, (L, D_h)).astype(N_T)
            # self_attention expects X, W_q, W_k, W_v, scale, out
            # Use random quantized inputs for X and random small weights
            W_q = np.random.randint(-8, 8, (D_h, D_h)).astype(N_T)
            W_k = np.random.randint(-8, 8, (D_h, D_h)).astype(N_T)
            W_v = np.random.randint(-8, 8, (D_h, D_h)).astype(N_T)
            s_llvm(Q_quant, W_q, W_k, W_v, scale, out)
            return out, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


def schedule_self_attention_mlp_4row_parallel(
    N_T: np.dtype,
    A_T: allo.ir.types,
    P: int = 8,  # Row parallelism factor (8 to hide fadd latency ~7)
    D_mlp: int = 3072,  # MLP hidden dimension
    mode: str = "csyn"
):
    """
    4-row parallel schedule for the `self_attention_and_mlp` kernel.

    Customizes `sdpa.self_attention_and_mlp` which includes:
    - QKV projection
    - SDPA computation
    - Residual connection (attention output + input)
    - MLP layers (fc1 with ReLU, fc2)
    """
    s = allo.customize(sdpa.self_attention_and_mlp, instantiate=[A_T, L, D_h, D_mlp, P])
    
    loops = s.get_loops()
    print("All loops:", loops)
    outer_loop = loops["row_outer"]
    print("Outer loop keys:", outer_loop)
    
    s.dataflow(outer_loop["i_outer"])  # Dataflow over outer row batches
    s.unroll(outer_loop["i_outer"], factor=2)
    
    # ===== Stage 0: Pipeline & Partition QKV projection =====
    # Pipeline the innermost k-loops of Q/K/V projections
    s.pipeline(loops.q_i.k)
    s.pipeline(loops.k_i.k)
    s.pipeline(loops.v_i.k)

    # ===== Stage 1: Matmul Q @ K^T =====
    loops = s.get_loops()
    outer_loop = loops["row_outer"]

    s.pipeline(outer_loop["j1"])  # Pipeline inner loop over L columns
    s.partition(s.acc_out, partition.Complete, dim=1)  
    s.partition(s.Q, partition.Complete, dim=2)
    s.partition(s.max_vals, partition.Complete, dim=1)
    
    # ===== Stage 2: Exp and sum =====
    s.pipeline(outer_loop["p2"])
    
    # ===== Stage 3: Normalize =====
    s.pipeline(outer_loop["j3"])
    
    # ===== Stage 4: Output matmul =====
    s.pipeline(outer_loop["j4"])
    
    # ===== Stage 5: Write attention outputs to delta =====
    s.pipeline(outer_loop["d2"])
    
    # ===== Stage 6-7: Residual & MLP fc1 =====
    # fc1_pm is the outer grid loop, fc1_d is the reduction loop
    s.pipeline(outer_loop["d4"])  # Pipeline the fc1 reduction loop
    
    # ===== Stage 8-9: MLP fc2 =====
    # fc2_pd is the outer grid loop, fc2_m is the reduction loop  
    # Partition hidden and W_fc2 for parallel access on reduction dimension
    s.partition(s.hidden, partition.Cyclic, dim=2, factor=64)
    s.partition(s.W_fc2, partition.Cyclic, dim=2, factor=64)
    s.pipeline(outer_loop["m2"])  # Pipeline the fc2 reduction loop
    s.unroll(outer_loop["m2"], factor=64)  # Unroll for parallelism


    
    dtype_str = "int4" if A_T == int4 else "int8"
    project_name = f"self_attention_mlp_{P}row_parallel_{dtype_str}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prj"
    match mode:
        case "llvm":
            out = np.zeros((L, D_h), dtype=N_T)
            s_llvm = s.build(project=project_name)
            X = np.random.randint(-8, 8, (L, D_h)).astype(N_T)
            W_q = np.random.randint(-8, 8, (D_h, D_h)).astype(N_T)
            W_k = np.random.randint(-8, 8, (D_h, D_h)).astype(N_T)
            W_v = np.random.randint(-8, 8, (D_h, D_h)).astype(N_T)
            W_fc1 = np.random.randint(-8, 8, (D_mlp, D_h)).astype(N_T)
            b_fc1 = np.random.randint(-8, 8, D_mlp).astype(N_T)
            W_fc2 = np.random.randint(-8, 8, (D_h, D_mlp)).astype(N_T)
            b_fc2 = np.random.randint(-8, 8, D_h).astype(N_T)
            scale = float(np.sqrt(D_h))
            s_llvm(X, W_q, W_k, W_v, W_fc1, b_fc1, W_fc2, b_fc2, scale, out)
            return out, s
        case "csyn":
            s_csyn = s.build(target="vitis_hls", mode="csyn", project=project_name)
            s_csyn()


if __name__ == "__main__":
    # Test baseline dataflow
    # print("=== Testing Single-Head SDPA Dataflow Baseline ===")
    # schedule_sdpa_no_dataflow(np.float32, float32, mode="csyn")
    
    # Uncomment to test other versions:
    # print("\n=== Testing Single-Head SDPA Dataflow Inlined ===")
    # schedule_sdpa_dataflow_inlined(np.float32, float32, mode="csyn")
    
    # print("\n=== Testing Single-Head SDPA Dataflow Optimized ===")
    # schedule_sdpa_dataflow_optimized(np.float32, float32, mode="csyn")
    
    # Streaming versions (recommended - much lower BRAM usage)
    # print("\n=== Testing Row-Streaming SDPA Baseline ===")
    # schedule_sdpa_streaming_baseline(np.float32, float32, mode="csyn")
    
    # print("\n=== Testing Row-Streaming SDPA Pipelined ===")
    # schedule_sdpa_streaming_pipelined(np.float32, float32, mode="csyn")
    
    # print("\n=== Testing Row-Streaming SDPA Optimized ===")
    # schedule_sdpa_streaming_optimized(np.float32, float32, mode="csyn")
    
    # Quantized versions (int4/int8 support)
    # print("\n=== Testing Quantized SDPA int8 Baseline ===")
    # schedule_sdpa_streaming_quantized_baseline(np.int8, int8, mode="csyn")
    
    # print("\n=== Testing Quantized SDPA int8 Tiled (16x) ===")
    # schedule_sdpa_streaming_quantized_tiled(np.int8, int8, tile_factor=16, mode="csyn")
    
    # 4-row parallel version - achieves II=1 on accumulator loops
    # print("\n=== Testing 4-Row Parallel SDPA int8 ===")
    #schedule_sdpa_streaming_4row_parallel(np.int8, int8, P=8, mode="csyn")
    
    # Uncomment to test int4
    # print("\n=== Testing Quantized SDPA int4 Baseline ===")
    # schedule_sdpa_streaming_quantized_baseline(np.int8, int4, mode="csyn")  # Note: int4 stored as int8 in numpy
    
    # print("\n=== Testing Quantized SDPA int4 Tiled (16x) ===")
    # schedule_sdpa_streaming_quantized_tiled(np.int8, int4, tile_factor=16, mode="csyn")
    
    # 4-row parallel version for self-attention
    # print("\n=== Testing 4-Row Parallel Self-Attention int8 ===")
    # schedule_self_attention_4row_parallel(np.int8, int8, P=8, mode="csyn")
    
    # 4-row parallel version for self-attention with MLP
    print("\n=== Testing 4-Row Parallel Self-Attention+MLP int8 with FC2 Unroll ===")
    schedule_self_attention_mlp_4row_parallel(np.int8, int8, P=8, D_mlp=3072, mode="csyn")
    
