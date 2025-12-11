import allo
from allo.ir.types import float32, int8, int32, Stream
from allo import dsl
import allo.backend.hls as hls
import allo.dataflow as df
from allo.library.systolic import systolic_tile
import numpy as np

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_kernels.kernels import add_bias


def test_mlp_feedforward():
    """Test MLP feedforward network with int8 systolic arrays, fp32 GELU, streaming dataflow"""
    
    MODE = "csyn"
    # MLP dimensions
    M = 1024           # batch size
    D_in = 768       # input feature dim
    H = 3072         # hidden dim
    D_out = 768      # output dim
    
    # Systolic array tile sizes
    M0 = 8         # Tile size for M dimension
    M1 = 8         # Tile size for N/K dimension
    
    @df.region()
    def top():
        # Streaming FIFOs between stages (depth=4 for buffering)
        fc1_stream: Stream[int32, 16]
        bias1_stream: Stream[float32, 16]
        gelu_stream: Stream[float32, 16]
        quant_stream: Stream[int8, 16]
        fc2_stream: Stream[int32, 16]
        
        @df.kernel(mapping=[1])
        def fc1_matmul(A: int8[M, D_in], W1: int8[D_in, H]):
            # FC1: Matrix multiply streaming output
            for i in range(M):
                for j in range(H):
                    acc: int32 = 0
                    for k in range(D_in):
                        a_val: int8 = A[i, k]
                        w_val: int8 = W1[k, j]
                        acc += a_val * w_val
                    fc1_stream.put(acc)
        
        @df.kernel(mapping=[1])
        def add_bias1(b1: float32[H]):
            # Add bias and convert to fp32
            for i in range(M):
                for j in range(H):
                    int_val: int32 = fc1_stream.get()
                    fp_val: float32 = int_val
                    bias1_stream.put(fp_val + b1[j])
        
        @df.kernel(mapping=[1])
        def apply_gelu():
            # GELU activation
            for i in range(M):
                for j in range(H):
                    x: float32 = bias1_stream.get()
                    x_cubed: float32 = x * x * x
                    inner: float32 = 0.7978845608028654 * (x + 0.044715 * x_cubed)
                    tanh_val: float32 = dsl.tanh(inner)
                    result: float32 = 0.5 * x * (1.0 + tanh_val)
                    gelu_stream.put(result)
        
        @df.kernel(mapping=[1])
        def quantize():
            # Quantize fp32 to int8
            for i in range(M):
                for j in range(H):
                    fp_val: float32 = gelu_stream.get()
                    quant_val: int8 = 0
                    if fp_val > 127.0:
                        quant_val = 127
                    elif fp_val < -128.0:
                        quant_val = -128
                    else:
                        quant_val = fp_val
                    quant_stream.put(quant_val)
        
        @df.kernel(mapping=[1])
        def fc2_matmul(W2: int8[H, D_out]):
            # Buffer quantized values
            gelu_buf: int8[M, H]
            for i in range(M):
                for j in range(H):
                    gelu_buf[i, j] = quant_stream.get()
            
            # FC2: Matrix multiply
            for i in range(M):
                for j in range(D_out):
                    acc: int32 = 0
                    for k in range(H):
                        a_val: int8 = gelu_buf[i, k]
                        w_val: int8 = W2[k, j]
                        acc += a_val * w_val
                    fc2_stream.put(acc)
        
        @df.kernel(mapping=[1])
        def add_bias2_output(b2: float32[D_out], Out: float32[M, D_out]):
            # Add output bias
            for i in range(M):
                for j in range(D_out):
                    int_val: int32 = fc2_stream.get()
                    fp_val: float32 = int_val
                    Out[i, j] = fp_val + b2[j]
    
    # HLS synthesis using dataflow streaming
    if hls.is_available("vitis_hls"):
        proj_name = f"mlp_stream_{M}x{D_in}_H{H}.prj"
        print(f"Building HLS dataflow project: {proj_name}")
        print(f"  Dimensions: M={M}, D_in={D_in}, H={H}, D_out={D_out}")
        print(f"  Architecture: Pure streaming with FIFOs between stages")
        print(f"  Log file: {proj_name}/out.prj/solution1/solution1.log")
        print("  (Monitor with: tail -f <log_file>)")
        print("\n🔄 Starting Vitis HLS synthesis (this may take several minutes)...")
        
        hls_mod = df.build(
            top,
            target="vitis_hls",
            mode=MODE,
            project=proj_name,
            wrap_io=True,
        )
        hls_mod()
        print("\n✅ MLP dataflow synthesis completed!")
        print(f"  Report: {proj_name}/out.prj/solution1/syn/report/top_csynth.rpt")


if __name__ == "__main__":
    test_mlp_feedforward()

    


