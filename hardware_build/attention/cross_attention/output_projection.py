import allo
from allo.ir.types import int8, int16, float32, bfloat16, int32
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from matrix_multiplies import mm1


def matmul_output_projection[
    T: (bfloat16, float32),
    A_L: int16, #action length
    V_D: int16, #VLM embedded dimension
    A_D: int16 #Action output dimension
](
    Z: "T[A_L, V_D]",
    O: "T[V_D, A_D]",
    Z_NEW: "T[A_L, A_D]"
):
    mm1[T, A_L, V_D, A_D](Z, O, Z_NEW)