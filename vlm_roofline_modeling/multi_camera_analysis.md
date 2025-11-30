# Multi-Camera Input in smolVLA

## Answer: YES - Multiple Cameras Supported

Based on the research and model architecture, **smolVLA is designed to accept input from multiple cameras**.

---

## How It Works

### Standard Operation (Robotics)
smolVLA takes:
1. **Multiple RGB images** from different camera viewpoints (e.g., top-down, wrist-mounted, side view)
2. **Robot state** (joint positions, gripper status)
3. **Natural language instruction** (e.g., "pick up the cup")

### Example Camera Setup
```
Camera 1 (OBS_IMAGE_1): Top-down view     → 512×512 RGB → 1024 patches
Camera 2 (OBS_IMAGE_2): Wrist-mounted     → 512×512 RGB → 1024 patches
Camera 3 (OBS_IMAGE_3): Side view         → 512×512 RGB → 1024 patches
                                                    ↓
                            Total: 3072 patches (3 × 1024)
```

---

## Impact on Your Roofline Analysis

### Current Analysis Assumption
- **Single camera**: 1024 patches
- Vision Encoder processes: `(1024, 768) × (768, 768)` GEMMs

### Multi-Camera Scenario
If using **N cameras**:
- **Total patches**: N × 1024
- Vision Encoder processes: `(N×1024, 768) × (768, 768)` GEMMs

#### Example: 3 Cameras
```python
# Single Camera (Current)
M = 1024
GEMM: (1024, 768) × (768, 768)

# Three Cameras
M = 3 × 1024 = 3072  
GEMM: (3072, 768) × (768, 768)
```

### Roofline Impact

**Good News**: More cameras = **Higher Operational Intensity**!

```
OI (1 camera)  = 2×1024×768×768 / [(1024×768 + 768×768 + 1024×768) × bytes]
OI (3 cameras) = 2×3072×768×768 / [(3072×768 + 768×768 + 3072×768) × bytes]
```

The weight matrix (768×768) is **amortized over 3× more activations**, moving kernels further right on the roofline (less memory-bound).

---

## How Images Are Combined

### Two Possible Architectures:

#### Option 1: Sequential Processing (Most Likely)
```
Camera 1 → Vision Encoder → 1024 tokens
Camera 2 → Vision Encoder → 1024 tokens  } → Concatenate → 3072 tokens
Camera 3 → Vision Encoder → 1024 tokens
                                    ↓
                    Vision Encoder Processes 3072-token sequence
```

- Vision Encoder sees **3072 patches total**
- Attention operates over all camera views simultaneously
- Cross-camera correlations captured

#### Option 2: Parallel Processing + Fusion
```
Camera 1 → Vision Encoder → Features 1 ─┐
Camera 2 → Vision Encoder → Features 2 ─┼→ Fusion Module → Combined Features
Camera 3 → Vision Encoder → Features 3 ─┘
```

- Each camera processed separately
- Features fused before text decoder
- Less common for VLA models

---

## Why This Matters for Your Analysis

### If Processing N Cameras:
1. **Vision Encoder Sequence Length** = N × 1024
2. **GEMM M dimension** = N × 1024
3. **Higher OI** = Better utilization of FPGA compute
4. **Still Memory-Bound** at low N, but improves as N increases

### Recommendation:
When finalizing your roofline analysis, **ask the user**:
- How many cameras will be used in their deployment?
- Single-view (1024 patches) or multi-view (N×1024 patches)?

Then update the notebooks accordingly!

---

## Architecture Note from Research

According to the smolVLA paper and HuggingFace documentation:
- **Training**: Used datasets with standardized camera mappings (OBS_IMAGE_1, OBS_IMAGE_2, etc.)
- **SmolVLM-2 backbone**: Optimized for multi-image and video inputs
- **Token reduction**: Token-shuffling reduces sequence length for efficiency (this might affect the exact number of patches)

The 1024 position embeddings we found could be:
- **Per camera** (total = N×1024 for N cameras), OR
- **Total budget** (distributed across cameras)

**Check with the user** to confirm their specific configuration!
