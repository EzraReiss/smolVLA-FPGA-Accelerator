"""
Test vector for self_attention - INT8 Self-Attention Kernel Verification

Compares Allo self_attention against PyTorch's scaled_dot_product_attention
as the ground truth reference.
"""

import allo
import numpy as np
import torch
import torch.nn.functional as F
from allo.ir.types import int8, int16, int32, float32
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from attention.cross_attention.self_attention import self_attention


def pytorch_int8_attention(
    Q: np.ndarray,      # [H, L, D_h] int8
    K: np.ndarray,      # [H, L, D_h] int8
    V: np.ndarray,      # [H, L, D_h] int8
    scale: float,
) -> np.ndarray:
    """
    PyTorch reference for INT8 self-attention.
    
    Matches the numerical flow of self_attention:
    1. Q @ K^T in int32
    2. Softmax with temperature scaling  
    3. Scale to int16 range (32768)
    4. softmax @ V in int32
    5. Right shift by 15
    """
    # Convert to torch tensors (int32 for matmul precision)
    Q_t = torch.from_numpy(Q.astype(np.int32))
    K_t = torch.from_numpy(K.astype(np.int32))
    V_t = torch.from_numpy(V.astype(np.int32))
    
    # Attention scores: Q @ K^T -> [H, L, L]
    attn_scores = torch.bmm(Q_t, K_t.transpose(1, 2))
    
    # Softmax with temperature scaling
    attn_softmax = F.softmax(attn_scores.float() / scale, dim=-1)
    
    # Scale to int16 range (matching self_attention's 32768 scaling)
    attn_scaled = (attn_softmax * 32768.0).to(torch.int32)
    
    # Output: attn @ V -> [H, L, D_h]
    out_int32 = torch.bmm(attn_scaled, V_t)
    
    # Right shift by 15
    out_int8 = (out_int32 >> 15).to(torch.int8)
    
    return out_int8.numpy()


def pytorch_full_self_attention(
    X: np.ndarray,      # [H, L, D_h] int8
    W_q: np.ndarray,    # [H, D_h, D_h] int8
    W_k: np.ndarray,    # [H, D_h, D_h] int8
    W_v: np.ndarray,    # [H, D_h, D_h] int8
    scale: float,
) -> tuple[np.ndarray, dict]:
    """
    Full self-attention including QKV projection using PyTorch.
    """
    # Use int32 to avoid overflow in projection
    X_t = torch.from_numpy(X.astype(np.int32))
    W_q_t = torch.from_numpy(W_q.astype(np.int32))
    W_k_t = torch.from_numpy(W_k.astype(np.int32))
    W_v_t = torch.from_numpy(W_v.astype(np.int32))
    
    # QKV Projection: Q = X @ W^T (matching self_attention's access pattern)
    Q = torch.bmm(X_t, W_q_t.transpose(1, 2))
    K = torch.bmm(X_t, W_k_t.transpose(1, 2))
    V = torch.bmm(X_t, W_v_t.transpose(1, 2))
    
    # Clip to int8 range
    Q_int8 = Q.clamp(-128, 127).to(torch.int8)
    K_int8 = K.clamp(-128, 127).to(torch.int8)
    V_int8 = V.clamp(-128, 127).to(torch.int8)
    
    out = pytorch_int8_attention(Q_int8.numpy(), K_int8.numpy(), V_int8.numpy(), scale)
    
    return out, {'Q': Q.numpy(), 'K': K.numpy(), 'V': V.numpy(),
                 'Q_int8': Q_int8.numpy(), 'K_int8': K_int8.numpy(), 'V_int8': V_int8.numpy()}


def test_full_self_attention_vs_pytorch():
    """Test self_attention against PyTorch reference."""
    print("=" * 60)
    print("Full INT8 Self-Attention: Allo vs PyTorch")
    print("=" * 60)
    
    H, L, D_h, P, P_s, scale = 2, 16, 8, 4, 4, 8.0
    
    np.random.seed(42)
    X = np.random.randint(-8, 8, size=(H, L, D_h), dtype=np.int8)
    W_q = np.random.randint(-4, 4, size=(H, D_h, D_h), dtype=np.int8)
    W_k = np.random.randint(-4, 4, size=(H, D_h, D_h), dtype=np.int8)
    W_v = np.random.randint(-4, 4, size=(H, D_h, D_h), dtype=np.int8)
    
    print(f"Config: H={H}, L={L}, D_h={D_h}, P={P}, P_s={P_s}, scale={scale}")
    
    # PyTorch reference
    print("\nPyTorch reference...")
    pytorch_out, _ = pytorch_full_self_attention(X, W_q, W_k, W_v, scale)
    print(f"  Output range: [{pytorch_out.min()}, {pytorch_out.max()}]")
    
    # Allo implementation
    print("\nAllo self_attention...")
    s = allo.customize(self_attention, instantiate=[int8, L, H, D_h, P, P_s])
    mod = s.build()
    allo_out = np.zeros((H, L, D_h), dtype=np.int8)
    mod(X, W_q, W_k, W_v, float(scale), allo_out)
    print(f"  Output range: [{allo_out.min()}, {allo_out.max()}]")
    
    # Compare
    diff = np.abs(pytorch_out.astype(np.int32) - allo_out.astype(np.int32))
    print(f"\nMax diff: {np.max(diff)}, Mean diff: {np.mean(diff):.2f}")
    print(f"Exact matches: {np.sum(pytorch_out == allo_out)}/{pytorch_out.size}")
    
    print("\nSample (head 0, rows 0-3, cols 0-3):")
    print("  PyTorch:", pytorch_out[0, :4, :4].tolist())
    print("  Allo:   ", allo_out[0, :4, :4].tolist())
    
    return np.max(diff) <= 2

def pytorch_layer_norm(
    X: np.ndarray,      # [H, L, D_h] float32
    epsilon: float = 1e-5,
) -> np.ndarray:
    """
    PyTorch reference for Layer Normalization.
    """
    X_t = torch.from_numpy(X)
    mean = X_t.mean(dim=-1, keepdim=True)
    variance = X_t.var(dim=-1, unbiased=False, keepdim=True)
    X_norm = (X_t - mean) / torch.sqrt(variance + epsilon)
    return X_norm.numpy()

def test_layer_norm_vs_pytorch():
    """Test layer normalization against PyTorch reference."""
    print("=" * 60)
    print("Layer Normalization: Allo vs PyTorch")
    print("=" * 60) 
    
    H, L, D_h = 2, 16, 8 
    
    np.random.seed(42)
    X = np.random.randn(H, L, D_h).astype(np.float32) * 0.1
    
    print(f"Config: H={H}, L={L}, D_h={D_h}")
    # PyTorch reference
    print("\nPyTorch reference...")
    pytorch_out = pytorch_layer_norm(X)
    print(f"  Output range: [{pytorch_out.min()}, {pytorch_out.max()}]")
    
    


if __name__ == "__main__":
    success = test_full_self_attention_vs_pytorch()
    print("\n✓ PASSED" if success else "\n✗ FAILED")
    sys.exit(0 if success else 1)
