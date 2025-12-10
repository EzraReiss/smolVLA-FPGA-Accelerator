#!/usr/bin/env python3
"""
Run MLP ReLU systolic cascaded synthesis with multiple tile configurations in parallel.
"""

import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool
import time
import os

# Configurations to test.
# Each entry may be either:
#  - (Rt, Ct)            -> apply same tile to both FC1 and FC2
#  - (Rt1, Ct1, Rt2, Ct2)-> specify FC1 and FC2 separately
CONFIGS = [
    (1, 1, 1, 1),            # 1x1 systolic tiles (both layers)
    (1, 3, 4, 3),     
    (2, 6, 8, 6),            
       
]

SCRIPT_DIR = Path(__file__).parent
SCRIPT_NAME = "mlp_systolic_cascaded_relu.py"
SCRIPT_PATH = SCRIPT_DIR / SCRIPT_NAME

process_num = 8


def run_config(config):
    """Run a single configuration.

    Accepts either a 2-tuple `(rt, ct)` meaning use same tile for FC1 and FC2,
    or a 4-tuple `(rt1, ct1, rt2, ct2)` to specify different tiles per layer.
    """
    if len(config) == 2:
        rt, ct = config
        rt1, ct1, rt2, ct2 = rt, ct, rt, ct
        config_name = f"{rt}x{ct}_both"
    elif len(config) == 4:
        rt1, ct1, rt2, ct2 = config
        config_name = f"FC1_{rt1}x{ct1}_FC2_{rt2}x{ct2}"
    else:
        raise ValueError("Config must be a 2-tuple or 4-tuple")
    
    print(f"\n{'='*60}")
    print(f"[PID {os.getpid()}] Starting synthesis: {config_name} (Rt={rt}, Ct={ct})")
    print(f"{'='*60}")
    
    # Create a temporary Python script with this config
    temp_script = SCRIPT_DIR / f"mlp_relu_temp_{config_name}.py"
    
    try:
        # Read the base script
        with open(SCRIPT_PATH, 'r') as f:
            script_content = f.read()
        
        # Modify the Rt/Ct values depending on config shape
        if len(config) == 2:
            modified_content = script_content.replace(
                "Rt1, Ct1 = 1, 1",
                f"Rt1, Ct1 = {rt1}, {ct1}"
            ).replace(
                "Rt2, Ct2 = 1, 1",
                f"Rt2, Ct2 = {rt2}, {ct2}"
            )
        else:
            modified_content = script_content.replace(
                "Rt1, Ct1 = 1, 1",
                f"Rt1, Ct1 = {rt1}, {ct1}"
            ).replace(
                "Rt2, Ct2 = 1, 1",
                f"Rt2, Ct2 = {rt2}, {ct2}"
            )
        
        # Write the temporary script
        with open(temp_script, 'w') as f:
            f.write(modified_content)
        
        # Run the synthesis
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            cwd=SCRIPT_DIR,
            capture_output=True,  # Capture output to avoid interleaving
            text=True,
            timeout=3600  # 1 hour timeout per config
        )
        elapsed_time = time.time() - start_time
        
        # Clean up temp script but keep project folders
        if temp_script.exists():
            temp_script.unlink()
        
        if result.returncode == 0:
            print(f"✅ {config_name}: SUCCESS (took {elapsed_time:.1f}s)")
            print(f"   Process ID: {os.getpid()}")
            return {
                'config': config_name,
                'status': 'SUCCESS',
                'time': elapsed_time,
                'rt1': rt1,
                'ct1': ct1,
                'rt2': rt2,
                'ct2': ct2,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            print(f"❌ {config_name}: FAILED (took {elapsed_time:.1f}s)")
            return {
                'config': config_name,
                'status': 'FAILED',
                'time': elapsed_time,
                'rt1': rt1,
                'ct1': ct1,
                'rt2': rt2,
                'ct2': ct2,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
    
    except subprocess.TimeoutExpired:
        print(f"⏱️  {config_name}: TIMEOUT (>1 hour)")
        if temp_script.exists():
            temp_script.unlink()
        return {
            'config': config_name,
            'status': 'TIMEOUT',
            'time': 3600,
            'rt': rt,
            'ct': ct
        }
    
    except Exception as e:
        print(f"💥 {config_name}: ERROR - {str(e)}")
        if temp_script.exists():
            temp_script.unlink()
        return {
            'config': config_name,
            'status': 'ERROR',
            'error': str(e),
            'rt': rt,
            'ct': ct
        }


def main():
    print(f"\n{'='*60}")
    print(f"MLP ReLU Systolic Cascaded - Parallel Synthesis Runner")
    print(f"{'='*60}")
    print(f"Script: {SCRIPT_PATH}")
    print(f"Configurations: {len(CONFIGS)}")
    print(f"Max parallel processes: {os.cpu_count()}")
    
    if not SCRIPT_PATH.exists():
        print(f"Error: Script not found: {SCRIPT_PATH}")
        return 1
    
    print(f"\nConfigurations to run:")
    for i, config in enumerate(CONFIGS, 1):
        if len(config) == 2:
            rt, ct = config
            print(f"  {i}. {rt}x{ct} (Rt={rt}, Ct={ct}) - both layers")
        else:
            rt1, ct1, rt2, ct2 = config
            print(f"  {i}. FC1={rt1}x{ct1} | FC2={rt2}x{ct2}")
    
    print(f"\n{'='*60}")
    print("Starting parallel synthesis runs...")
    print(f"Running {process_num} processes in parallel")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Run configs in parallel
    with Pool(processes=process_num) as pool:
        results = pool.map(run_config, CONFIGS)
    
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    success_count = 0
    failed_count = 0
    timeout_count = 0
    error_count = 0
    
    for result in results:
        config = result['config']
        status = result['status']
        time_taken = result.get('time', 0)
        rt1 = result.get('rt1', '?')
        ct1 = result.get('ct1', '?')
        rt2 = result.get('rt2', '?')
        ct2 = result.get('ct2', '?')
        
        # Find project folder
        proj_pattern = f"mlp_cascaded_relu_systolic_M1024_Din768_H3072"
        proj_folders = list(SCRIPT_DIR.glob(f"{proj_pattern}*"))
        proj_folder = f"{proj_pattern}.prj" if proj_folders else "N/A"
        
        if status == 'SUCCESS':
            print(f"✅ {config:30s} : SUCCESS ({time_taken:.1f}s)")
            print(f"   FC1: {rt1}x{ct1} | FC2: {rt2}x{ct2}")
            print(f"   Folder: {proj_folder}")
            success_count += 1
        elif status == 'FAILED':
            print(f"❌ {config:30s} : FAILED ({time_taken:.1f}s)")
            failed_count += 1
        elif status == 'TIMEOUT':
            print(f"⏱️  {config:30s} : TIMEOUT")
            timeout_count += 1
        else:
            print(f"💥 {config:30s} : ERROR")
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Total: {len(CONFIGS)} | Success: {success_count} | Failed: {failed_count} | Timeout: {timeout_count} | Error: {error_count}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Project folders: {SCRIPT_DIR}")
    print(f"{'='*60}\n")
    
    return 0 if failed_count == 0 and timeout_count == 0 and error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
