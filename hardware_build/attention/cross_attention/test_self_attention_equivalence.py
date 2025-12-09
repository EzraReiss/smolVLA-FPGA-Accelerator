"""Test functional equivalence between self_attention_2 and self_attention"""
import allo
import numpy as np
from allo.ir.types import float32, int8, int16, int32

# Use smaller dimensions for quick testing
H_val = 12    # Number of heads
L_val = 1024   # Sequence length (must be divisible by P)
D_h_val = 64  # Head dimension
P_val = 4    # Parallelism factor

def self_attention_2[
    T: (float32, int8),
    L: int16,
    H: int16,
    D_h: int16,
](
    X:   "T[H, L, D_h]",
    W_q: "T[H, D_h, D_h]",
    W_k: "T[H, D_h, D_h]",
    W_v: "T[H, D_h, D_h]",
    scale: "float32",
    out: "T[H, L, D_h]"
):
    for h1 in allo.grid(H, name="head_loop"):
        Q: "T[L, D_h]" = 0
        K: "T[L, D_h]" = 0
        V: "T[L, D_h]" = 0

        for i_precalc in allo.grid(L, name="mm_i_loop"):
            for j_precalc in allo.grid(D_h, name="mm_j_loop"):
                for k_precalc in allo.reduction(D_h, name="prj_dot_product"):
                    Q[i_precalc, j_precalc] += X[h1, i_precalc, k_precalc] * W_q[h1, j_precalc, k_precalc]
                    K[i_precalc, j_precalc] += X[h1, i_precalc, k_precalc] * W_k[h1, j_precalc, k_precalc]
                    V[i_precalc, j_precalc] += X[h1, i_precalc, k_precalc] * W_v[h1, j_precalc, k_precalc]

        for i_out in allo.grid(L, name="row_loop"):
            attn_row: "int32[L]"
            max_val: "int32" = -2147483648
            
            for j_attn in allo.grid(L, name="attn_loop"):
                acc: "int32" = 0
                for k_attn in allo.reduction(D_h, name="dot_product"):
                    acc += Q[i_out, k_attn] * K[j_attn, k_attn]
                attn_row[j_attn] = acc
                if acc > max_val:
                    max_val = acc
                    
            softmax_rows: "T[L]"
            sum_exps: "T" = 0.0
            
            for j_exp in allo.grid(L, name="exp_loop"):
                exp_pow: "float32" = attn_row[j_exp] - max_val
                exp_val: "T" = allo.exp(exp_pow / scale)
                softmax_rows[j_exp] = exp_val
        
            softmax_rows_2: "T[L]"
            for j_exp_sum in allo.grid(L, name="sum_loop"):
                softmax_row = softmax_rows[j_exp_sum]
                softmax_rows_2[j_exp_sum] = softmax_row
                sum_exps += softmax_row
                    
            softmax_scaled: "int16[L]"
            for j_norm in allo.grid(L, name="norm_loop"):
                norm_val: "float32" = softmax_rows_2[j_norm] / sum_exps
                softmax_scaled[j_norm] = norm_val * 32768.0
            
            acc_out: "int32[D_h]" = 0
            for j_out in allo.grid(L, name="out_row_loop"):
                softmax_val: "int32" = softmax_scaled[j_out]
                for k_out in allo.reduction(D_h, name="out_loop"):   
                    v_val: "int32" = V[j_out, k_out]
                    acc_out[k_out] += softmax_val * v_val

            for k_final in allo.grid(D_h, name="final_loop"):
                out[h1, i_out, k_final] = acc_out[k_final] >> 15


def self_attention[
    T: (float32, int8),
    L: int16,
    H: int16,
    D_h: int16,
    P: int16,
](
    X:   "T[H, L, D_h]",
    W_q: "T[H, D_h, D_h]",
    W_k: "T[H, D_h, D_h]",
    W_v: "T[H, D_h, D_h]",
    scale: "float32",
    out: "T[H, L, D_h]"
):
    for h1 in allo.grid(H, name="head_loop"):
        Q: "T[L, D_h]" = 0
        K: "T[L, D_h]" = 0
        V: "T[L, D_h]" = 0

        # FIXED: Write to correct Q/K/V indices
        for i_precalc in allo.grid(L//P, name="mm_i_loop"):
            for p_precalc in allo.grid(P, name="p_precalc_loop"):
                for j_precalc in allo.grid(D_h, name="mm_j_loop"):
                    for k_precalc in allo.reduction(D_h, name="prj_dot_product"):
                        row_idx: int16 = i_precalc * P + p_precalc
                        Q[row_idx, j_precalc] += X[h1, row_idx, k_precalc] * W_q[h1, j_precalc, k_precalc]
                        K[row_idx, j_precalc] += X[h1, row_idx, k_precalc] * W_k[h1, j_precalc, k_precalc]
                        V[row_idx, j_precalc] += X[h1, row_idx, k_precalc] * W_v[h1, j_precalc, k_precalc]

        for i_out in allo.grid(L//P, name="row_loop"):
            attn_row: "int32[P, L]"
            max_val: "int32[P]" = -2147483648
            
            for j_attn in allo.grid(L, name="attn_loop"):
                for p_attn in allo.grid(P, name="p_attn_loop"):
                    acc: "int32" = 0
                    row_idx: int16 = i_out * P + p_attn
                    for k_attn in allo.reduction(D_h, name="dot_product"):
                        acc += Q[row_idx, k_attn] * K[j_attn, k_attn]
                    attn_row[p_attn, j_attn] = acc
                    if acc > max_val[p_attn]:
                        max_val[p_attn] = acc
                    
            softmax_rows: "T[P, L]"
            sum_exps: "T[P]" = 0.0
            
            for j_exp in allo.grid(L, name="exp_loop"):
                for p_exp in allo.grid(P, name="p_exp_loop"):
                    exp_pow: "float32" = attn_row[p_exp, j_exp] - max_val[p_exp]
                    exp_val: "T" = allo.exp(exp_pow / scale)
                    softmax_rows[p_exp, j_exp] = exp_val
        
            softmax_rows_2: "T[P, L]"
            for j_exp_sum in allo.grid(L, name="sum_loop"):
                for p_sum in allo.grid(P, name="p_sum_loop"):
                    softmax_row = softmax_rows[p_sum, j_exp_sum]
                    softmax_rows_2[p_sum, j_exp_sum] = softmax_row
                    sum_exps[p_sum] += softmax_row
                    
            softmax_scaled: "int16[P, L]"
            for j_norm in allo.grid(L, name="norm_loop"):
                for p_norm in allo.grid(P, name="p_norm_loop"):
                    norm_val: "float32" = softmax_rows_2[p_norm, j_norm] / sum_exps[p_norm]
                    softmax_scaled[p_norm, j_norm] = norm_val * 32768.0
            
            acc_out: "int32[P, D_h]" = 0
            for j_out in allo.grid(L, name="out_row_loop"):
                for p_out in allo.grid(P, name="p_out_loop"):
                    softmax_val: "int32" = softmax_scaled[p_out, j_out]
                    for k_out in allo.reduction(D_h, name="out_loop"):   
                        v_val: "int32" = V[j_out, k_out]
                        acc_out[p_out, k_out] += softmax_val * v_val

            for k_final in allo.grid(D_h, name="final_loop"):
                for p_final in allo.grid(P, name="p_final_loop"):
                    # FIXED: Use i_out * P + p_final instead of i_out + p_final
                    out[h1, i_out * P + p_final, k_final] = acc_out[p_final, k_final] >> 15


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create test inputs
    X = np.random.randint(-10, 10, size=(H_val, L_val, D_h_val)).astype(np.int8)
    W_q = np.random.randint(-5, 5, size=(H_val, D_h_val, D_h_val)).astype(np.int8)
    W_k = np.random.randint(-5, 5, size=(H_val, D_h_val, D_h_val)).astype(np.int8)
    W_v = np.random.randint(-5, 5, size=(H_val, D_h_val, D_h_val)).astype(np.int8)
    scale: float32 = 8.0  # Must be plain Python float, not numpy type
    
    out_2 = np.zeros((H_val, L_val, D_h_val), dtype=np.int8)
    out_3 = np.zeros((H_val, L_val, D_h_val), dtype=np.int8)
    
    print("Building self_attention_2...")
    s2 = allo.customize(self_attention_2, instantiate=[int8, L_val, H_val, D_h_val])
    mod2 = s2.build(target="llvm")
    
    print("Building self_attention...")
    s3 = allo.customize(self_attention, instantiate=[int8, L_val, H_val, D_h_val, P_val])
    mod3 = s3.build(target="llvm")
    
    print("Running self_attention_2...")
    mod2(X, W_q, W_k, W_v, scale, out_2)
    
    print("Running self_attention...")
    mod3(X, W_q, W_k, W_v, scale, out_3)
    
    print("\n===== Results =====")
    print(f"out_2 shape: {out_2.shape}")
    print(f"out_3 shape: {out_3.shape}")
    
    # Check if outputs match
    if np.array_equal(out_2, out_3):
        print("\n✅ SUCCESS: self_attention_2 and self_attention are FUNCTIONALLY EQUIVALENT!")
    else:
        print("\n❌ FAILURE: Outputs differ!")
        diff = np.abs(out_2.astype(np.int32) - out_3.astype(np.int32))
        print(f"Max absolute difference: {np.max(diff)}")
        print(f"Number of differing elements: {np.sum(out_2 != out_3)} / {out_2.size}")
        
        # Show first few differences
        diff_indices = np.argwhere(out_2 != out_3)
        print("\nFirst 5 differences:")
        for idx in diff_indices[:5]:
            h, l, d = idx
            print(f"  [{h},{l},{d}]: out_2={out_2[h,l,d]}, out_3={out_3[h,l,d]}")
