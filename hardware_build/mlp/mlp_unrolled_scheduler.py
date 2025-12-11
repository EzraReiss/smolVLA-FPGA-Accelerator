# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int8, int16, int32, float32
from allo.customize import Partition as partition
import allo.backend.hls as hls
import numpy as np
from datetime import datetime

# Import the kernel top-level region and dimensions
from mlp_unrolled_cascaded_relu import top, M, D_in, H, D_out, DEFAULT_P


def schedule_mlp_cascade(A_T=int8, P=8, mode="csyn"):
    """
    Schedule the MLP cascade kernel with pipeline, dataflow, and partition directives.
    Follows the self-attention scheduler pattern.
    
    Args:
        A_T: Data type for activations and weights (int8 or float32)
        P: P-row parallelism factor
        mode: "csyn" for C synthesis, "sw_emu" for simulation
    """
    
    print(f"Scheduling MLP cascade with P={P}, dtype={A_T}")
    
    # Customize the top dataflow region so we can schedule inner-stage loops
    # The `top` region in the kernel is a concrete region (contains a concrete
    # wrapper `mlp_cascade_inst`), so no template instantiation list is needed.
    s = allo.customize(top)

    loops = s.get_loops()
    outer = loops["row_outer"]
    # Pipeline inner reduction loops in the stage kernels
    # Access nested loops by name under the outer band. Loop names can
    # vary depending on how kernels were inlined/instantiated, so try
    # flexible matching and print available names for debugging.
    import re
    outer_repr = str(outer)
    names = re.findall(r"'([^']+)'", outer_repr)
    print(f"Nested loop names under row_outer: {names}")

    def find_name(sub):
        for n in names:
            if sub in n:
                return n
        return None

    # Try to pipeline the reduction loops for FC1 and FC2 using flexible names
    for target in ("k_fc1", "fc1_k", "k_fc"):
        n = find_name(target)
        if n:
            try:
                s.pipeline(outer[n])
                print(f"Pipelined loop: {n}")
            except Exception as e:
                print(f"Failed to pipeline {n}: {e}")
            break

    for target in ("k_fc2", "fc2_k", "k_fc"):
        n = find_name(target)
        if n:
            try:
                s.pipeline(outer[n])
                print(f"Pipelined loop: {n}")
            except Exception as e:
                print(f"Failed to pipeline {n}: {e}")
            break

    # Conservative partitioning of weight arrays on their reduction dim
    s.partition(s.W1, partition.Cyclic, dim=1, factor=P)
    s.partition(s.W2, partition.Cyclic, dim=1, factor=P)

    return s


def test_mlp_scheduler():
    """Test the scheduled MLP cascade with HLS synthesis."""
    
    print("=" * 80)
    print("MLP Cascade with P-Row Parallelism (Self-Attention Pattern)")
    print("=" * 80)
    print(f"\nMLP Architecture:")
    print(f"  FC1: ({M}, {D_in}) x ({D_in}, {H}) -> ({M}, {H})")
    print(f"  ReLU activation")
    print(f"  FC2: ({M}, {H}) x ({H}, {D_out}) -> ({M}, {D_out})")
    print(f"\nScheduling:")
    print(f"  P-row parallelism: {DEFAULT_P}")
    print(f"  Pipeline: k_fc1, k_fc2, j_fc1, j_relu, j_fc2, p_fc1, p_relu, p_fc2")
    print(f"  Dataflow: row_outer")
    print(f"  Partition: W1, W2 (cyclic), b1, b2 (complete)")
    
    if not hls.is_available("vitis_hls"):
        print("\n[SKIPPED] Vitis HLS not available")
        return
    
    print("\n[Running HLS C Synthesis]")
    
    # Create schedule
    s = schedule_mlp_cascade(A_T=int8, P=DEFAULT_P, mode="csyn")
    
    # Build HLS project
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    proj_name = f"mlp_cascaded_relu_dataflow_P{DEFAULT_P}_M{M}_Din{D_in}_H{H}_{timestamp}.prj"
    
    print(f"Building HLS project: {proj_name}")
    
    mod = s.build(
        target="vitis_hls",
        mode="csyn",
        project=proj_name
    )
    
    # Execute synthesis
    mod()
    
    print(f"\n✅ HLS C Synthesis completed!")
    print(f"  Project: {proj_name}")
    print(f"  Report: {proj_name}/out.prj/solution1/syn/report/mlp_cascade_csynth.rpt")
    print(f"  Top report: {proj_name}/out.prj/solution1/syn/report/top_csynth.rpt")


if __name__ == "__main__":
    test_mlp_scheduler()
