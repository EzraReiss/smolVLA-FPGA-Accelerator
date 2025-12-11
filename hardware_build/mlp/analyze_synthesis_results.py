#!/usr/bin/env python3
"""
Analyze MLP systolic cascaded synthesis results.
Extracts BRAM, LUT, FF, DSP usage and latency from csynth.rpt files.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
import json

SCRIPT_DIR = Path(__file__).parent

# Device specs for xcu280-fsvh2892-2L-e (VU280)
DEVICE_SPECS = {
    'bram': 4032,      # Total BRAM_18K blocks
    'dsp': 9024,       # Total DSP blocks
    'ff': 2607360,     # Total Flip-Flops
    'lut': 1303680,    # Total LUTs
}


def extract_resources_from_report(report_path: Path) -> Optional[Dict]:
    """Extract resource usage from csynth.rpt file."""
    
    if not report_path.exists():
        return None
    
    try:
        with open(report_path, 'r') as f:
            content = f.read()
        
        resources = {
            'path': str(report_path),
            'bram': None,
            'lut': None,
            'ff': None,
            'dsp': None,
            'latency_cycles_min': None,
            'latency_cycles_max': None,
            'latency_ns_min': None,
            'latency_ns_max': None,
        }
        
        # Extract latency info from "Latency (cycles)" table
        # Table format can use seconds or milliseconds for absolute times.
        # Example row (cycles|min|max|abs_min|abs_max|...):
        # |  20299415|  20299417|  67.597 ms|  67.597 ms|  19365890|  19365890|  dataflow|
        latency_match = re.search(
            r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*(ms|sec)\s*\|\s*([\d.]+)\s*(ms|sec)",
            content
        )
        if latency_match:
            resources['latency_cycles_min'] = int(latency_match.group(1))
            resources['latency_cycles_max'] = int(latency_match.group(2))
            abs_min_val = float(latency_match.group(3))
            abs_min_unit = latency_match.group(4)
            abs_max_val = float(latency_match.group(5))
            abs_max_unit = latency_match.group(6)

            def to_ns(val, unit):
                if unit == 'sec':
                    return val * 1e9
                # assume 'ms'
                return val * 1e6

            resources['latency_ns_min'] = to_ns(abs_min_val, abs_min_unit)
            resources['latency_ns_max'] = to_ns(abs_max_val, abs_max_unit)
        
        # Extract resource usage from "Utilization Estimates" summary table
        # Look for the Total row in the table with BRAM_18K, DSP, FF, LUT, URAM
        utilization_match = re.search(
            r'\|Total\s+\|\s+(\d+)\s*\|\s+(\d+)\s*\|\s+(\d+)\s*\|\s+(\d+)\s*\|\s+(\d+)\s*\|',
            content
        )
        if utilization_match:
            resources['bram'] = int(utilization_match.group(1))
            resources['dsp'] = int(utilization_match.group(2))
            resources['ff'] = int(utilization_match.group(3))
            resources['lut'] = int(utilization_match.group(4))
            # URAM is group(5), not used currently
        
        return resources
    
    except Exception as e:
        print(f"Error reading {report_path}: {e}")
        return None


def parse_config_from_folder(folder_name: str) -> Optional[Dict]:
    """Parse systolic array config from folder name."""
    # Format: mlp_cascaded_relu_systolic_M1024_Din768_H3072_FC1_2x4_FC2_2x4
    # or: mlp_cascaded_systolic_M1024_Din768_H3072_FC1_2x4_FC2_2x4
    
    try:
        # Extract FC1 and FC2 tile configs
        fc1_match = re.search(r'FC1_(\d+)x(\d+)', folder_name)
        fc2_match = re.search(r'FC2_(\d+)x(\d+)', folder_name)
        
        if fc1_match and fc2_match:
            return {
                'fc1_rt': int(fc1_match.group(1)),
                'fc1_ct': int(fc1_match.group(2)),
                'fc2_rt': int(fc2_match.group(1)),
                'fc2_ct': int(fc2_match.group(2)),
                'is_relu': 'relu' in folder_name.lower(),
            }
    except Exception as e:
        print(f"Error parsing config from {folder_name}: {e}")
    
    return None


def find_and_analyze_projects():
    """Find all project folders and analyze their synthesis results."""
    
    # Find all .prj folders
    prj_folders = sorted(SCRIPT_DIR.glob("**/mlp_cascaded*systolic*.prj"))
    
    if not prj_folders:
        print("No project folders found matching pattern '**/mlp_cascaded*systolic*.prj'")
        print(f"Search directory: {SCRIPT_DIR}")
        return
    
    print(f"\n{'='*100}")
    print(f"MLP Systolic Cascaded - Synthesis Results Analysis")
    print(f"{'='*100}")
    print(f"Found {len(prj_folders)} project folders\n")
    
    results = []
    
    for prj_folder in prj_folders:
        # Find the csynth.rpt file
        report_path = prj_folder / "out.prj" / "solution1" / "syn" / "report" / "top_csynth.rpt"
        
        if not report_path.exists():
            # Try alternate path
            report_path = prj_folder / "out.prj" / "solution1" / "syn" / "report" / "csynth.rpt"
        
        if not report_path.exists():
            print(f"⚠️  Report not found: {prj_folder.name}")
            continue
        
        # Extract resources
        resources = extract_resources_from_report(report_path)
        if not resources:
            print(f"⚠️  Could not parse report: {prj_folder.name}")
            continue

        # Read report content for per-instance parsing
        try:
            with open(report_path, 'r') as f:
                rpt_text = f.read()
        except Exception:
            rpt_text = ''

        # Extract per-module instance latencies (absolute times) and LUTs
        module_stats = {
            'fc1': {'count': 0, 'lut': 0, 'cycles_min': [], 'cycles_max': [], 'abs_ms': []},
            'fc2': {'count': 0, 'lut': 0, 'cycles_min': [], 'cycles_max': [], 'abs_ms': []},
            'relu': {'count': 0, 'lut': 0, 'cycles_min': [], 'cycles_max': [], 'abs_ms': []},
        }

        # Latency instance regex (instance | module | cycles_min | cycles_max | abs_min unit | abs_max unit)
        lat_re = re.compile(r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*(ms|sec)")
        for m in lat_re.finditer(rpt_text):
            inst = m.group(1).strip()
            mod = m.group(2).strip()
            cmin = int(m.group(3))
            cmax = int(m.group(4))
            abs_val = float(m.group(5))
            unit = m.group(6)
            abs_ms = abs_val if unit == 'ms' else abs_val * 1000.0

            if 'fc1_gemm' in mod or mod.startswith('fc1_') and 'fc1_load' not in mod and 'fc1_store' not in mod:
                cat = 'fc1'
            elif 'fc2_gemm' in mod or mod.startswith('fc2_') and 'fc2_load' not in mod and 'fc2_store' not in mod:
                cat = 'fc2'
            elif 'apply_relu' in mod or mod.startswith('apply_relu'):
                cat = 'relu'
            else:
                # also consider store/load latencies belonging to layers
                if mod.startswith('fc1_'):
                    cat = 'fc1'
                elif mod.startswith('fc2_'):
                    cat = 'fc2'
                else:
                    cat = None

            if cat:
                module_stats[cat]['count'] += 1
                module_stats[cat]['cycles_min'].append(cmin)
                module_stats[cat]['cycles_max'].append(cmax)
                module_stats[cat]['abs_ms'].append(abs_ms)

        # Utilization instance regex to find LUT per instance
        util_re = re.compile(r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*(?:\d+|\s*-?)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(?:\d+)\s*\|")
        # groups: instance, module, BRAM, FF, LUT
        for m in util_re.finditer(rpt_text):
            inst = m.group(1).strip()
            mod = m.group(2).strip()
            try:
                bram_i = int(m.group(3))
            except:
                bram_i = 0
            try:
                ff_i = int(m.group(4))
            except:
                ff_i = 0
            try:
                lut_i = int(m.group(5))
            except:
                lut_i = 0

            if mod.startswith('fc1_'):
                module_stats['fc1']['lut'] += lut_i
            elif mod.startswith('fc2_'):
                module_stats['fc2']['lut'] += lut_i
            elif 'apply_relu' in mod:
                module_stats['relu']['lut'] += lut_i

        
        # Parse config
        config = parse_config_from_folder(prj_folder.name)
        if not config:
            print(f"⚠️  Could not parse config: {prj_folder.name}")
            continue
        
        # Combine results
        result = {
            'folder': prj_folder.name,
            **config,
            **resources,
            'module_stats': module_stats,
        }
        results.append(result)
    
    # Sort by FC1 tile size
    results.sort(key=lambda x: (x['fc1_rt'] * x['fc1_ct'], x['is_relu']))
    
    # Print header
    print(f"{'Folder':<70} | {'FC1':<8} | {'FC2':<8} | Type  | {'BRAM':>15} | {'LUT':>15} | {'FF':>15} | {'DSP':>15} | {'Latency (ms)':>12}")
    print(f"{'-'*175}")
    
    # Print results
    for result in results:
        folder_short = result['folder'][:65] + "..." if len(result['folder']) > 68 else result['folder']
        fc1_config = f"{result['fc1_rt']}x{result['fc1_ct']}"
        fc2_config = f"{result['fc2_rt']}x{result['fc2_ct']}"
        activation_type = "ReLU" if result['is_relu'] else "GELU"
        
        # Format resources as "number (xx%)"
        bram_str = "N/A"
        if result['bram'] is not None:
            bram_pct = result['bram'] * 100 / DEVICE_SPECS['bram']
            bram_str = f"{result['bram']:,} ({bram_pct:5.2f}%)"
        
        lut_str = "N/A"
        if result['lut'] is not None:
            lut_pct = result['lut'] * 100 / DEVICE_SPECS['lut']
            lut_str = f"{result['lut']:,} ({lut_pct:5.2f}%)"
        
        ff_str = "N/A"
        if result['ff'] is not None:
            ff_pct = result['ff'] * 100 / DEVICE_SPECS['ff']
            ff_str = f"{result['ff']:,} ({ff_pct:5.2f}%)"
        
        dsp_str = "N/A"
        if result['dsp'] is not None:
            dsp_pct = result['dsp'] * 100 / DEVICE_SPECS['dsp']
            dsp_str = f"{result['dsp']:,} ({dsp_pct:5.2f}%)"
        
        latency_ms = "N/A"
        if result['latency_ns_min'] is not None:
            latency_ms = f"{result['latency_ns_min']/1e6:.2f}"
        
        print(f"{folder_short:<70} | {fc1_config:<8} | {fc2_config:<8} | {activation_type:<5} | {bram_str:>15} | {lut_str:>15} | {ff_str:>15} | {dsp_str:>15} | {latency_ms:>12}")
        # Print per-module breakdown (cycles and LUTs)
        stats = result.get('module_stats', {})
        if stats:
            def fmt_cat(cat):
                s = stats.get(cat, {})
                if not s or s.get('count', 0) == 0:
                    return f"{cat.upper()}: N/A"
                cmin = min(s['cycles_min'])
                cmax = max(s['cycles_max'])
                avg_ms = sum(s['abs_ms'])/len(s['abs_ms']) if s['abs_ms'] else 0.0
                return f"{cat.upper()}: count={s['count']} LUTs={s['lut']:,} cycles_min={cmin:,} cycles_max={cmax:,} avg_abs_ms={avg_ms:.2f}"

            print(f"    -> {fmt_cat('fc1')}")
            print(f"    -> {fmt_cat('fc2')}")
            print(f"    -> {fmt_cat('relu')}")
    
    # Print summary
    print(f"\n{'='*175}")
    print(f"Summary Statistics")
    print(f"{'='*175}")
    
    if results:
        # Get min/max/avg for each metric
        luts = [r['lut'] for r in results if r['lut'] is not None]
        brams = [r['bram'] for r in results if r['bram'] is not None]
        ffs = [r['ff'] for r in results if r['ff'] is not None]
        dsps = [r['dsp'] for r in results if r['dsp'] is not None]
        
        print(f"\nLUT Usage:")
        if luts:
            min_lut = min(luts)
            max_lut = max(luts)
            avg_lut = sum(luts) // len(luts)
            print(f"  Min:     {min_lut:>8,}  ({min_lut*100/DEVICE_SPECS['lut']:>5.2f}%)")
            print(f"  Max:     {max_lut:>8,}  ({max_lut*100/DEVICE_SPECS['lut']:>5.2f}%)")
            print(f"  Average: {avg_lut:>8,}  ({avg_lut*100/DEVICE_SPECS['lut']:>5.2f}%)")
        
        print(f"\nBRAM Usage:")
        if brams:
            min_bram = min(brams)
            max_bram = max(brams)
            avg_bram = sum(brams) // len(brams)
            print(f"  Min:     {min_bram:>8,}  ({min_bram*100/DEVICE_SPECS['bram']:>5.2f}%)")
            print(f"  Max:     {max_bram:>8,}  ({max_bram*100/DEVICE_SPECS['bram']:>5.2f}%)")
            print(f"  Average: {avg_bram:>8,}  ({avg_bram*100/DEVICE_SPECS['bram']:>5.2f}%)")
        
        print(f"\nFF Usage:")
        if ffs:
            min_ff = min(ffs)
            max_ff = max(ffs)
            avg_ff = sum(ffs) // len(ffs)
            print(f"  Min:     {min_ff:>8,}  ({min_ff*100/DEVICE_SPECS['ff']:>5.2f}%)")
            print(f"  Max:     {max_ff:>8,}  ({max_ff*100/DEVICE_SPECS['ff']:>5.2f}%)")
            print(f"  Average: {avg_ff:>8,}  ({avg_ff*100/DEVICE_SPECS['ff']:>5.2f}%)")
        
        print(f"\nDSP Usage:")
        if dsps:
            min_dsp = min(dsps)
            max_dsp = max(dsps)
            avg_dsp = sum(dsps) // len(dsps)
            print(f"  Min:     {min_dsp:>8,}  ({min_dsp*100/DEVICE_SPECS['dsp']:>5.2f}%)")
            print(f"  Max:     {max_dsp:>8,}  ({max_dsp*100/DEVICE_SPECS['dsp']:>5.2f}%)")
            print(f"  Average: {avg_dsp:>8,}  ({avg_dsp*100/DEVICE_SPECS['dsp']:>5.2f}%)")
    
    print(f"\n{'='*175}\n")
    
    # Export to JSON for further analysis
    json_path = SCRIPT_DIR / "synthesis_results.json"
    try:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results exported to: {json_path}")
    except Exception as e:
        print(f"⚠️  Could not export JSON: {e}")


if __name__ == "__main__":
    find_and_analyze_projects()
