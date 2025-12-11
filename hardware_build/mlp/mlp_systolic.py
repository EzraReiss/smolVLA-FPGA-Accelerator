# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo
from allo.ir.types import int8, Stream, UInt
from allo.utils import get_np_struct_type
import allo.dataflow as df
import allo.backend.hls as hls
import allo.dsl as dsl
import numpy as np

# M, N, K = 64, 64, 64
# M, N, K = 128, 128, 128
# M, N, K = 256, 256, 256
# M, N, K = 512, 512, 512
M, N, K = 1024, 768, 3072
# Rt, Ct = 16, 16
# Rt, Ct = 8, 8
# Rt, Ct = 4, 4
Rt, Ct = 2, 4
P0, P1 = Rt + 2, Ct + 2

# Toggle flags
RUN_SIMULATION = False  # Set to True to run simulation
RUN_HLS_CSYN = True     # Set to True to run HLS C synthesis


@df.region()
def top():
    L3_A: Stream[UInt(Rt * 8), 4]
    L3_B: Stream[UInt(Ct * 8), 4]
    L3_C: Stream[UInt(Rt * 8), 4]

    L2_A: Stream[UInt(Rt * 8), 4][P0 - 1]
    L2_B: Stream[UInt(Ct * 8), 4][P1 - 1]

    L1_C: Stream[UInt(Rt * 8), 4][Rt, Ct]
    L2_C: Stream[UInt(Rt * 8), 4][Ct]

    fifo_A: Stream[int8, 4][Rt, Ct]
    fifo_B: Stream[int8, 4][Rt, Ct]

    @df.kernel(mapping=[1])
    def offchip_loadA(A_Packed: UInt(Rt * 8)[M * K // Rt]):
        for mt, nt in dsl.grid(M // Rt, N // Ct):
            for k in range(K):
                L3_A.put(A_Packed[mt * K + k])

    @df.kernel(mapping=[1])
    def offchip_loadB(B_Packed: UInt(Ct * 8)[K * N // Ct]):
        for mt, nt in dsl.grid(M // Rt, N // Ct):
            for k in range(K):
                L3_B.put(B_Packed[nt * K + k])

    @df.kernel(mapping=[P0, P1])
    def gemm():
        i, j = df.get_pid()
        # peripheral kernels
        with allo.meta_if(i == 0 and j == 0):
            for mt, nt in dsl.grid(M // Rt, N // Ct):
                for k in range(K):
                    L2_A[1].put(L3_A.get())
                    L2_B[1].put(L3_B.get())

        with allo.meta_elif(i == P0 - 1 and j == P1 - 1):
            for mt, nt in dsl.grid(M // Rt, N // Ct):
                for n in range(Ct):
                    L3_C.put(L2_C[Ct - 1].get())

        with allo.meta_elif(i in {0, P0 - 1} and j in {0, P1 - 1}):
            pass

        with allo.meta_elif(j == 0):
            # i > 0, the first column
            for mt, nt in dsl.grid(M // Rt, N // Ct):
                for k in range(K):
                    a = L2_A[i].get()
                    # unpack data
                    fifo_A[i - 1, 0].put(a[8 * (i - 1) : 8 * i])
                    with allo.meta_if(i < Rt):
                        L2_A[i + 1].put(a)

        with allo.meta_elif(i == 0):
            # j > 0, the first row
            for mt, nt in dsl.grid(M // Rt, N // Ct):
                for k in range(K):
                    b = L2_B[j].get()
                    fifo_B[0, j - 1].put(b[8 * (j - 1) : 8 * j])
                    with allo.meta_if(j < Ct):
                        L2_B[j + 1].put(b)

        with allo.meta_elif(i == P0 - 1):
            for mt, nt in dsl.grid(M // Rt, N // Ct):
                c_C = L1_C[i - 2, Ct - j].get()
                L2_C[j - 1].put(c_C)
                with allo.meta_if(j != 1):
                    for ind in range(j - 1):
                        L2_C[j - 1].put(L2_C[j - 2].get())

        with allo.meta_elif(j == P1 - 1):
            pass

        # main body
        with allo.meta_else():
            for mt, nt in dsl.grid(M // Rt, N // Ct):
                c: int8 = 0
                for k in range(K):
                    a: int8 = fifo_A[i - 1, j - 1].get()
                    b: int8 = fifo_B[i - 1, j - 1].get()
                    c += a * b
                    with allo.meta_if(j < Ct):
                        fifo_A[i - 1, j].put(a)
                    with allo.meta_if(i < Rt):
                        fifo_B[i, j - 1].put(b)

                packed_tmp: UInt(Rt * 8)
                with allo.meta_if(i == 1):
                    packed_tmp = 0
                with allo.meta_else():
                    packed_tmp = L1_C[i - 2, j - 1].get()

                packed_c: UInt(Rt * 8) = 0
                for m in range(Rt):
                    if m == i - 1:
                        packed_c[m * 8 : (m + 1) * 8] = c
                    else:
                        packed_c[m * 8 : (m + 1) * 8] = packed_tmp[m * 8 : (m + 1) * 8]
                L1_C[i - 1, j - 1].put(packed_c)

    @df.kernel(mapping=[1])
    def offchip_store(C_Packed: UInt(Rt * 8)[M * N // Rt]):
        for mt, nt in dsl.grid(M // Rt, N // Ct):
            for n in range(Ct):
                C_Packed[mt * N + nt * Ct + n] = L3_C.get()


@pytest.mark.skip(
    reason="Hang when using large sizes. Raise error when using small sizes (seems like something wrong with data types)."
)
def test_large_scale_gemm():
    print("Starting test_large_scale_gemm...")
    print(f"Matrix dimensions: M={M}, N={N}, K={K}")
    print(f"Tile dimensions: Rt={Rt}, Ct={Ct}")
    
    def serialize_A(matrix_A):
        A_ser = np.zeros((M * K), dtype=np.int8)
        for mt in range(M // Rt):
            for k in range(K):
                for m in range(Rt):
                    A_ser[mt * (K * Rt) + k * Rt + m] = matrix_A[mt * Rt + m, k]
        return A_ser

    def serialize_B(matrix_B):
        B_ser = np.zeros((K * N), dtype=np.int8)
        for nt in range(N // Ct):
            for k in range(K):
                for n in range(Ct):
                    B_ser[nt * (K * Ct) + k * Ct + n] = matrix_B[k, nt * Ct + n]
        return B_ser

    def deserialize_C(C_ser):
        matrix_C = np.zeros((M, N), dtype=np.int8)
        for mt in range(M // Rt):
            for n in range(N):
                for m in range(Rt):
                    matrix_C[mt * Rt + m, n] = C_ser[mt * (N * Rt) + n * Rt + m]
        return matrix_C

    print("\nGenerating random matrices...")
    # # TODO: Fix the packing-related issue!
    # Use native numpy integer types instead of structured types
    bits_A = Rt * 8
    bits_B = Ct * 8
    bits_C = Rt * 8
    print(f"Bit widths: A={bits_A}, B={bits_B}, C={bits_C}")
    
    if bits_A <= 8:
        np_type_A = np.uint8
    elif bits_A <= 16:
        np_type_A = np.uint16
    elif bits_A <= 32:
        np_type_A = np.uint32
    else:
        np_type_A = np.uint64
    
    if bits_B <= 8:
        np_type_B = np.uint8
    elif bits_B <= 16:
        np_type_B = np.uint16
    elif bits_B <= 32:
        np_type_B = np.uint32
    else:
        np_type_B = np.uint64
    
    if bits_C <= 8:
        np_type_C = np.uint8
    elif bits_C <= 16:
        np_type_C = np.uint16
    elif bits_C <= 32:
        np_type_C = np.uint32
    else:
        np_type_C = np.uint64

    A = np.random.randint(-2, 2, (M, K), dtype=np.int8)
    B = np.random.randint(-2, 2, (K, N), dtype=np.int8)
    C = np.zeros((M, N), dtype=np.int8)
    print(f"Created matrices: A.shape={A.shape}, B.shape={B.shape}, C.shape={C.shape}")
    print(f"Numpy types: np_type_A={np_type_A}, np_type_B={np_type_B}, np_type_C={np_type_C}")

    print("\nSerializing matrices...")
    A_ser = serialize_A(A)
    print(f"A serialized: shape={A_ser.shape}, dtype={A_ser.dtype}")
    A_packed = A_ser.view(np_type_A)
    print(f"A packed: shape={A_packed.shape}, dtype={A_packed.dtype}")
    
    B_ser = serialize_B(B)
    print(f"B serialized: shape={B_ser.shape}, dtype={B_ser.dtype}")
    B_packed = B_ser.view(np_type_B)
    print(f"B packed: shape={B_packed.shape}, dtype={B_packed.dtype}")
    
    C_packed = np.zeros((M * N // Rt), dtype=np_type_C)
    print(f"C packed: shape={C_packed.shape}, dtype={C_packed.dtype}")

    if RUN_SIMULATION:
        print("\nBuilding dataflow simulator...")
        sim_mod = df.build(top, target="simulator")
        print("Simulator built successfully.")
        
        print("\nRunning simulation...")
        sim_mod(A_packed, B_packed, C_packed)
        print("Simulation completed.")
        
        print("\nDeserializing output...")
        C = deserialize_C(C_packed.view(np.int8))
        print("\nVerifying results...")
        C_golden = np.dot(A, B)
        print(f"C shape: {C.shape}, C_golden shape: {C_golden.shape}")
        np.testing.assert_allclose(C, C_golden, atol=1e-5)
        print("Dataflow Simulator Passed!")
    else:
        print("\n[SKIPPED] Simulation (RUN_SIMULATION=False)")

    if RUN_HLS_CSYN and hls.is_available("vitis_hls"):
        print("\n[Running HLS C Synthesis]")
        modc = df.build(
            top,
            target="vitis_hls",
            mode="csyn",
            project=f"mlp_systolic_M{M}N{N}K{K}_Systolic_{Rt}x{Ct}.prj",
            wrap_io=False,
        )
        modc()
        print("HLS C Synthesis completed!")
    else:
        print("\n[SKIPPED] HLS C Synthesis (RUN_HLS_CSYN=False or Vitis HLS not available)")

        '''
        C_packed = np.zeros((M * N // Rt), dtype=np_type_C)
        modhw = df.build(
            top,
            target="vitis_hls",
            mode="hw",
            project=f"hwmulticache-M{M}N{N}K{K}-R{Rt}xC{Ct}.prj",
            wrap_io=False,
        )
        modhw(A_packed, B_packed, C_packed)
        '''

        # # Enable the hw_emu test with data types that OpenCL supports
        # C_packed = np.zeros((M * N // Rt), dtype=np_type_C)
        # mod = df.build(
        #     top,
        #     target="vitis_hls",
        #     mode="hw_emu",
        #     project=f"df-packed-{Rt}x{Ct}.prj",
        #     wrap_io=False,
        # )
        # mod(A_packed, B_packed, C_packed)
        # C = deserialize_C(C_packed.view(np.int8))
        # C_golden = np.dot(A, B)
        # np.testing.assert_allclose(C, C_golden, atol=1e-5)
        # print("Passed hw_emu Test!")


if __name__ == "__main__":
    test_large_scale_gemm()