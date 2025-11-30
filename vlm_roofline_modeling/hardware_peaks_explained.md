# Derivation of Peak Performance Numbers for Alveo U280

The peak performance numbers used in the roofline analysis (`vision_roofline.ipynb` and `text_roofline.ipynb`) are **Theoretical Datasheet Peaks** for the Xilinx Alveo U280 Data Center Accelerator Card.

## 1. The Numbers Used

* **FP32 Peak**: 5.41 TFLOPs
* **INT8 Peak**: 18.6 TOPS
* **INT4 Peak**: 37.2 TOPS

## 2. Source & Derivation

### A. FP32 (Single Precision) - 5.41 TFLOPs

* **Source**: Xilinx Alveo U280 Datasheet / Vitis Library Benchmarks.
* **Derivation**:
  * The U280 contains **9,024 DSP48E2 slices**.
  * For FP32, DSPs are combined to perform floating-point multiply-accumulate (MAC) operations.
  * This peak assumes the hardware is fully utilized, often running at the maximum DSP frequency (typically **~600 MHz** or higher for DSP-only logic), not necessarily the 300 MHz system clock.
  * *Note*: Achieving this in a custom HLS design running at 300 MHz is difficult; you would typically get closer to ~2.7 TFLOPs (half) if constrained to 300 MHz.

### B. INT8 (8-bit Integer) - 18.6 TOPS

* **Source**: Xilinx Vitis AI DPU (Deep Learning Processor Unit) Benchmarks.
* **Derivation**:
  * INT8 operations are highly efficient on FPGAs because:
        1. **DSP Packing**: The DSP48E2 slice can perform two INT8 MACs per cycle using its pre-adder and multiplier features.
        2. **Logic (LUT) Compute**: FPGAs can implement massive amounts of INT8 math using standard logic cells (LUTs) in parallel with DSPs.
  * The **18.6 TOPS** figure is a standard "AI Inference" peak cited for the U280 when using the Xilinx DPU architecture (specifically the DPUv3E or similar configurations optimized for INT8).
  * It represents the aggregate compute of DSPs + LUTs running at high frequency.

### C. INT4 (4-bit Integer) - 37.2 TOPS

* **Source**: Estimation / Scaling.
* **Derivation**:
  * **2x Scaling Rule**: Since INT4 requires half the bit-width of INT8, we can theoretically pack **2x as many operations** into the same hardware resources (both DSPs and LUTs).
  * $18.6 \text{ TOPS (INT8)} \times 2 = 37.2 \text{ TOPS (INT4)}$
  * This assumes the architecture supports native INT4 packing (which modern Xilinx tools and DPUs increasingly support).

## 3. "Realistic" vs. "Peak" for Your Design

The roofline model plots these **Absolute Ceilings**. Your actual HLS design running at **300 MHz** will have a lower *practical* ceiling unless you highly optimize for parallelism.

* **System Clock**: 300 MHz
* **Theoretical Max at 300 MHz** (assuming perfect 2 ops/DSP/cycle for INT8):
  * $9024 \text{ DSPs} \times 2 \text{ ops} \times 300 \text{ MHz} \approx 5.4 \text{ TOPS}$ (DSP only)
  * To reach 18.6 TOPS at 300 MHz, you need massive LUT-based parallelism (using the FPGA fabric) in addition to DSPs.

**Summary**: The numbers represent the **hardware's maximum potential**, not necessarily what a basic HLS kernel achieves out-of-the-box.
