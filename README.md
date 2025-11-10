# smolVLA-FPGA-Accelerator

## Help

If when running `pip install -r requirements.txt` you get an error that looks like:

```bash
        File "/home/.../miniconda3/envs/smolVLA/lib/python3.12/subprocess.py", line 571, in run
          raise CalledProcessError(retcode, process.args,
      subprocess.CalledProcessError: Command '['cmake', '-G Ninja', '/home/.../smolVLA-FPGA-Accelerator/submodules/allo/mlir', '-DMLIR_DIR=/work/shared/common/llvm-project-19.x/build/lib/cmake/mlir', '-DPython3_EXECUTABLE=/home/.../miniconda3/envs/smolVLA/bin/python3.12']' returned non-zero exit status 1.
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for allo
  Running setup.py clean for allo
Successfully built lerobot
Failed to build allo
error: failed-wheel-build-for-install

× Failed to build installable wheels for some pyproject.toml based projects
╰─> allo
```

To solve this error try downloading a different version of cmake and ninja using:
```bash 
  conda install -c conda-forge cmake=3.27.9 ninja=1.11.1
```
