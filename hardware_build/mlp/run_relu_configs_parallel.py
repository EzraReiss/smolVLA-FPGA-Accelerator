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

# Configurations to test: (Rt1, Ct1, Rt2, Ct2)
CONFIGS = [
    (1, 1),   # 1x1 systolic tiles
    (2, 4),   # 2x1 systolic tiles
    (4, 8),   # 4x2 systolic tiles
]

SCRIPT_DIR = Path(__file__).parent
SCRIPT_NAME = "mlp_systolic_cascaded_relu.py"
SCRIPT_PATH = SCRIPT_DIR / SCRIPT_NAME


def run_config(config):
    """Run a single configuration."""
    rt, ct = config
    config_name = f"{rt}x{ct}"
    
    print(f"\n{'='*60}")
    print(f"Starting synthesis: {config_name} (Rt={rt}, Ct={ct})")
    print(f"{'='*60}")
    
    # Create a temporary Python script with this config
    temp_script = SCRIPT_DIR / f"mlp_relu_temp_{rt}x{ct}.py"
    
    try:
        # Read the base script
        with open(SCRIPT_PATH, 'r') as f:
            script_content = f.read()
        
        # Modify the Rt1, Ct1, Rt2, Ct2 values
        modified_content = script_content.replace(
            "Rt1, Ct1 = 1, 1",
            f"Rt1, Ct1 = {rt}, {ct}"
        ).replace(
            "Rt2, Ct2 = 1, 1",
            f"Rt2, Ct2 = {rt}, {ct}"
        )
        
        # Write the temporary script
        with open(temp_script, 'w') as f:
            f.write(modified_content)
        
        # Run the synthesis
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            cwd=SCRIPT_DIR,
            capture_output=False,  # Show output in real-time
            text=True,
            timeout=3600  # 1 hour timeout per config
        )
        elapsed_time = time.time() - start_time
        
        # Clean up temp script but keep project folders
        if temp_script.exists():
            temp_script.unlink()
        
        if result.returncode == 0:
            print(f"✅ {config_name}: SUCCESS (took {elapsed_time:.1f}s)")
            return {
                'config': config_name,
                'status': 'SUCCESS',
                'time': elapsed_time,
                'rt': rt,
                'ct': ct
            }
        else:
            print(f"❌ {config_name}: FAILED (took {elapsed_time:.1f}s)")
            return {
                'config': config_name,
                'status': 'FAILED',
                'time': elapsed_time,
                'rt': rt,
                'ct': ct
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
        rt, ct = config
        print(f"  {i}. {rt}x{ct} (Rt={rt}, Ct={ct})")
    
    print(f"\n{'='*60}")
    print("Starting parallel synthesis runs...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Run configs in parallel
    with Pool(processes=min(4, os.cpu_count())) as pool:
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
        rt = result.get('rt', '?')
        ct = result.get('ct', '?')
        
        # Find project folder
        proj_pattern = f"mlp_cascaded_relu_systolic_M1024_Din768_H3072"
        proj_folders = list(SCRIPT_DIR.glob(f"{proj_pattern}*"))
        proj_folder = f"{proj_pattern}.prj" if proj_folders else "N/A"
        
        if status == 'SUCCESS':
            print(f"✅ {config:8s} : SUCCESS ({time_taken:.1f}s)")
            print(f"   Folder: {proj_folder}")
            success_count += 1
        elif status == 'FAILED':
            print(f"❌ {config:8s} : FAILED ({time_taken:.1f}s)")
            failed_count += 1
        elif status == 'TIMEOUT':
            print(f"⏱️  {config:8s} : TIMEOUT")
            timeout_count += 1
        else:
            print(f"💥 {config:8s} : ERROR")
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Total: {len(CONFIGS)} | Success: {success_count} | Failed: {failed_count} | Timeout: {timeout_count} | Error: {error_count}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Project folders: {SCRIPT_DIR}")
    print(f"{'='*60}\n")
    
    return 0 if failed_count == 0 and timeout_count == 0 and error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
