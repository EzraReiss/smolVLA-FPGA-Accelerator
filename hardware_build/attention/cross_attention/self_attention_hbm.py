"""
HBM-backed K/V Projection Kernel for Self-Attention

This kernel computes K and V projections with minimal BRAM usage by:
1. Processing X in tiles (T_i rows at a time)
2. Buffering W columns locally for each output column
3. Streaming results directly to HBM

Uses wrap_io=False to avoid full BRAM buffering of outputs.
"""
import allo
import numpy as np
from allo.ir.types import float32, bfloat16, int32, int16, int8, int4, int64, Index, bool


def compute_kv_to_hbm_fast[
    T: (bfloat16, float32, int4, int8),
    L: int16,     # Total rows (must be divisible by T_i)
    H: int16,     # Number of Heads
    D: int16,     # Embedding Length
    D_h: int16,   # Head Embedding Length
    T_i: int16,   # Tile size for rows
](
    X:     "T[L, D]",
    W_k:   "T[H, D_h, D]",
    W_v:   "T[H, D_h, D]",
    K_out: "int32[H, L, D_h]",  # Output to HBM
    V_out: "int32[H, L, D_h]"   # Output to HBM
):
    """
    Fast tiled kernel for K/V projection with HBM streaming.
    
    Architecture:
    - Tile loop processes T_i rows at a time
    - For each output column j, loads W column for all heads
    - Reduction loop achieves II=1 with W buffered locally
    - Results written to HBM after each tile
    
    BRAM usage (T_i=64, H=12, D=768, D_h=64):
      X_local:   192 KB
      K_local:   192 KB
      V_local:   192 KB
      W_k/v_col:  72 KB
      Total:    ~650 KB (~3% of U280 BRAM)
    """
    # Tile loop - process T_i rows at a time
    for i_tile in allo.grid(L // T_i, name="tile_loop"):
        # Local buffer for X tile
        X_local: "int32[T_i, D]"
        
        # Local accumulators for K/V
        K_local: "int32[H, T_i, D_h]"
        V_local: "int32[H, T_i, D_h]"
        
        # Phase 1: Load X tile from HBM to BRAM
        for ii in allo.grid(T_i, name="load_x_row"):
            for kk in allo.grid(D, name="load_x_col"):
                X_local[ii, kk] = X[i_tile * T_i + ii, kk]
        
        # Phase 2: Compute K/V for each output column
        for j in allo.grid(D_h, name="compute_col"):
            # Load W column for all heads (H * D values)
            W_k_col: "int32[H, D]"
            W_v_col: "int32[H, D]"
            
            for h in allo.grid(H, name="load_w_head"):
                for k in allo.grid(D, name="load_w_k"):
                    W_k_col[h, k] = W_k[h, j, k]
                    W_v_col[h, k] = W_v[h, j, k]
            
            # Compute all rows for this column
            for i_compute in allo.grid(T_i, name="compute_row"):
                # Initialize accumulators
                K_acc: "int32[H]"
                V_acc: "int32[H]"
                for h in allo.grid(H, name="init_acc"):
                    K_acc[h] = 0
                    V_acc[h] = 0
                
                # Reduction loop - achieves II=1 with local W buffers
                for k1 in allo.reduction(D, name="reduction"):
                    X_val: int32 = X_local[i_compute, k1]
                    for h in allo.grid(H, name="parallel_head"):
                        K_acc[h] = K_acc[h] + X_val * W_k_col[h, k1]
                        V_acc[h] = V_acc[h] + X_val * W_v_col[h, k1]
                
                # Write accumulators to local KV buffers
                for h in allo.grid(H, name="store_acc"):
                    K_local[h, i_compute, j] = K_acc[h]
                    V_local[h, i_compute, j] = V_acc[h]
        
        # Phase 3: Write K/V tile to HBM
        for h in allo.grid(H, name="store_head"):
            for i_store in allo.grid(T_i, name="store_row"):
                for j_store in allo.grid(D_h, name="store_col"):
                    K_out[h, i_tile * T_i + i_store, j_store] = K_local[h, i_store, j_store]
                    V_out[h, i_tile * T_i + i_store, j_store] = V_local[h, i_store, j_store]


# =============================================================================
# Build Configuration
# =============================================================================
if __name__ == "__main__":
    # Production dimensions
    L = 2048
    H = 12
    D = 768
    D_h = 64
    T_i = 64  # Tile size
    
    print("=" * 60)
    print("Building compute_kv_to_hbm_fast Kernel")
    print("=" * 60)
    print(f"Dimensions: L={L}, H={H}, D={D}, D_h={D_h}")
    print(f"Tile size: T_i={T_i}")
    
    # Customize kernel
    print("\nCustomizing kernel...")
    s = allo.customize(
        compute_kv_to_hbm_fast,
        instantiate=[int8, L, H, D, D_h, T_i]
    )
    
    # HBM mapping
    hbm_mapping = {
        "X": 0,
        "W_k": 1,
        "W_v": 2,
        "K_out": 3,
        "V_out": 4,
    }
    
    print("\nBuilding for csyn...")
    try:
        # s.build(
        #     target="vitis_hls",
        #     mode="csyn",
        #     project="compute_kv_fast_other.prj",
        #     configs={"hbm_mapping": hbm_mapping},
        #     wrap_io=False
        # )()
        s.unroll(s.get_loops()["tile_loop"]["k"], factor=32)
        s.unroll(s.get_loops()["tile_loop"]["kk"], factor=32)

        s.build(
            target="vitis_hls",
            mode="csyn",
            project="compute_kv_fast_unroll_no_io.prj",
            configs={"hbm_mapping": hbm_mapping},
            wrap_io=False
        )()
        
        print("\n" + "=" * 60)
        print("✓ SYNTHESIS COMPLETED!")
        print("  Check compute_kv_fast_unroll.prj/out.prj/solution1/syn/report/")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Synthesis failed: {e}")
