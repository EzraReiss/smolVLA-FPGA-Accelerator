#!/usr/bin/env python3
"""
Run MLP ReLU systolic cascaded synthesis with multiple tile configurations in parallel.
"""
import sys
from pathlib import Path
from multiprocessing import Pool
import time
import os
import argparse
from shutil import copyfile

# Configurations to test.
# Each entry may be either:
#  - (Rt, Ct)            -> apply same tile to both FC1 and FC2
#  - (Rt1, Ct1, Rt2, Ct2)-> specify FC1 and FC2 separately
CONFIGS = [
    (1, 1, 1, 1),            # 1x1 systolic tiles (both layers)
    (1, 3, 4, 3),     
    (2, 6, 8, 6),        
    (3, 9, 8, 6),            
    (4, 12, 8, 6),     
    (4, 12, 12, 9),            
    (4, 12, 16, 12),            
    (4, 12, 20, 15),            
    (5, 15, 8, 6),
    (5, 15, 12, 9),
    (6, 20, 15, 12),
    (8, 24, 16, 12),
    (8, 24, 20, 15),
    (10, 30, 20, 15),
    (10, 30, 40, 30),
    (12, 30, 40, 30),
]


SCRIPT_DIR = Path(__file__).parent
SCRIPT_NAME = "mlp_tiled_cascaded_relu.py"
SCRIPT_PATH = SCRIPT_DIR / SCRIPT_NAME

# Alternate script (row-streaming / unrolled dataflow style)
ALTERNATE_SCRIPT = "mlp_unrolled_cascaded_relu.py"
process_num = 8
# Per-config timeout in seconds. Set to `None` to disable timeouts.
# Default: 12 hours
TIMEOUT = 12 * 3600  # 43200 seconds


def run_config(config, target_script_path=None, cmd=None, keep_temp=False):
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
    
    # Ensure required modules are available in worker processes (multiprocessing spawn)
    import subprocess

    print(f"\n{'='*60}")
    if len(config) == 2:
        print(f"[PID {os.getpid()}] Starting synthesis: {config_name} (Rt={rt1}, Ct={ct1})")
    else:
        print(f"[PID {os.getpid()}] Starting synthesis: {config_name} (FC1={rt1}x{ct1}, FC2={rt2}x{ct2})")
    print(f"{'='*60}")
    
    # Resolve target script and temp script
    if target_script_path is None:
        target_script_path = SCRIPT_PATH

    temp_script = None
    if cmd is None:
        temp_script = SCRIPT_DIR / f"mlp_relu_temp_{config_name}.py"
    
    try:
        # If user provided a shell command, run it directly.
        if cmd is not None:
            start_time = time.time()
            # run as shell string
            result = subprocess.run(
                cmd,
                cwd=SCRIPT_DIR,
                shell=True,
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
            )
            elapsed_time = time.time() - start_time

        else:
            # Read the target script and substitute tile sizes if the patterns exist.
            with open(target_script_path, 'r') as f:
                script_content = f.read()

            # Decide whether the target script needs Rt/Ct substitution.
            # Some scripts (e.g. the new unrolled/dataflow version) do not
            # expose Rt1/Ct1 variables and should be run unchanged.
            if "Rt1, Ct1" in script_content or "Rt2, Ct2" in script_content:
                # Modify the Rt/Ct values depending on config shape (if present)
                modified_content = script_content.replace(
                    "Rt1, Ct1 = 1, 1",
                    f"Rt1, Ct1 = {rt1}, {ct1}"
                ).replace(
                    "Rt2, Ct2 = 1, 1",
                    f"Rt2, Ct2 = {rt2}, {ct2}"
                )
            else:
                # Script doesn't expose Rt/Ct variables (likely unrolled/dataflow).
                # Keep content unchanged — we'll still write a temp copy so each
                # run has an isolated script file.
                modified_content = script_content

            # Write the temporary script
            with open(temp_script, 'w') as f:
                f.write(modified_content)

            # Run the synthesized script with the same Python interpreter
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, str(temp_script)],
                cwd=SCRIPT_DIR,
                capture_output=True,  # Capture output to avoid interleaving
                text=True,
                timeout=TIMEOUT,
            )
            elapsed_time = time.time() - start_time

            # Clean up temp script unless user requested to keep it
            if temp_script.exists() and not keep_temp:
                temp_script.unlink()

        # Evaluate result
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
        print(f"⏱️  {config_name}: TIMEOUT (>{TIMEOUT} seconds)")
        if temp_script is not None and temp_script.exists():
            temp_script.unlink()
        return {
            'config': config_name,
            'status': 'TIMEOUT',
            'time': TIMEOUT if TIMEOUT is not None else 0,
            'rt1': rt1,
            'ct1': ct1,
            'rt2': rt2,
            'ct2': ct2,
        }
    
    except Exception as e:
        print(f"💥 {config_name}: ERROR - {str(e)}")
        if temp_script is not None and temp_script.exists():
            temp_script.unlink()
        return {
            'config': config_name,
            'status': 'ERROR',
            'error': str(e),
            'rt1': rt1,
            'ct1': ct1,
            'rt2': rt2,
            'ct2': ct2,
        }


def worker_helper(task):
    """Top-level worker helper so multiprocessing can pickle the callable.

    Expects a tuple: (config, target_str_or_None, cmd_or_None, keep_temp_bool)
    """
    cfg, target_str, cmd, keep_temp = task
    target = Path(target_str) if target_str is not None else None
    return run_config(cfg, target_script_path=target, cmd=cmd, keep_temp=keep_temp)


def main():
    parser = argparse.ArgumentParser(description="Parallel runner for MLP/HLS scripts")
    parser.add_argument("--target", type=str, default=None,
                        help="Path to target Python script to run (overrides default mlp script)")
    parser.add_argument("--cmd", type=str, default=None,
                        help="If provided, run this shell command (string) instead of editing/running a Python script")
    parser.add_argument("--processes", type=int, default=None,
                        help="Number of parallel worker processes to use")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Per-config timeout in seconds (use 0 for no timeout)")
    parser.add_argument("--keep-temp", action='store_true',
                        help="Keep temporary generated scripts for debugging")
    args = parser.parse_args()

    target = Path(args.target) if args.target else None
    global process_num, TIMEOUT
    if args.processes:
        process_num = args.processes
    if args.timeout is not None:
        TIMEOUT = None if args.timeout == 0 else args.timeout

    print(f"\n{'='*60}")
    print(f"MLP ReLU Systolic Cascaded - Parallel Synthesis Runner")
    print(f"{'='*60}")
    print(f"Script: {target if target is not None else SCRIPT_PATH}")
    print(f"Configurations: {len(CONFIGS)}")
    print(f"Max parallel processes: {os.cpu_count()}")

    if target is not None and args.cmd is None and not Path(target).exists():
        print(f"Error: Target script not found: {target}")
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
        tasks = []
        target_str = str(target) if target is not None else None
        for cfg in CONFIGS:
            tasks.append((cfg, target_str, args.cmd, args.keep_temp))

        results = pool.map(worker_helper, tasks)
    
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

        # Find project folder — support systolic and dataflow/unrolled naming
        proj_prefixes = [
            "mlp_cascaded_relu_systolic",
            "mlp_cascaded_relu_dataflow",
            "mlp_cascaded_relu_unrolled",
        ]
        proj_folder = "N/A"
        for pref in proj_prefixes:
            prog = list(SCRIPT_DIR.glob(f"{pref}*"))
            if prog:
                # Use the actual folder name found (first match)
                proj_folder = prog[0].name
                break

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
