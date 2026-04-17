#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os
import platform
import shutil
import subprocess
import sys
from typing import List, Tuple


def _run(cmd: List[str]) -> Tuple[int, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        return int(proc.returncode), str(proc.stdout or "").strip()
    except Exception as exc:
        return 999, f"{type(exc).__name__}: {exc}"


def _print_title(title: str) -> None:
    print(f"\n=== {title} ===")


def _check_dynlib(name: str) -> bool:
    try:
        ctypes.CDLL(name)
        print(f"[OK] load {name}")
        return True
    except Exception as exc:
        print(f"[FAIL] load {name}: {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose CUDA/GPU runtime issues for Qwen/PyTorch.")
    ap.add_argument("--require-gpu", type=int, default=1, help="Exit non-zero if GPU is unavailable (1/0).")
    args = ap.parse_args()
    require_gpu = bool(int(args.require_gpu))

    ok = True

    _print_title("System")
    print(f"python={sys.version.split()[0]} executable={sys.executable}")
    print(f"platform={platform.platform()}")
    print(f"cwd={os.getcwd()}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}")

    _print_title("Driver / nvidia-smi")
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        print("[FAIL] nvidia-smi not found in PATH")
        ok = False
    else:
        rc, out = _run([nvidia_smi, "--query-gpu=driver_version,name", "--format=csv,noheader"])
        if rc == 0:
            print("[OK] nvidia-smi query")
            print(out or "(empty)")
        else:
            print("[FAIL] nvidia-smi query")
            print(out)
            ok = False

    _print_title("Device files")
    devs = [p for p in ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia0") if os.path.exists(p)]
    if devs:
        print("[OK] found", ", ".join(devs))
    else:
        print("[FAIL] no /dev/nvidia* device files found")
        ok = False

    _print_title("CUDA dynamic libs")
    libcuda_ok = _check_dynlib("libcuda.so.1")
    _check_dynlib("libcudart.so")
    if not libcuda_ok:
        ok = False

    _print_title("PyTorch CUDA")
    try:
        import torch  # type: ignore

        print(f"torch={torch.__version__}")
        print(f"torch.version.cuda={torch.version.cuda}")
        cuda_avail = bool(torch.cuda.is_available())
        dev_count = int(torch.cuda.device_count()) if cuda_avail else 0
        print(f"torch.cuda.is_available={cuda_avail}")
        print(f"torch.cuda.device_count={dev_count}")
        if cuda_avail and dev_count > 0:
            for i in range(dev_count):
                try:
                    print(f"gpu[{i}]={torch.cuda.get_device_name(i)}")
                except Exception as exc:
                    print(f"[WARN] cannot read gpu[{i}] name: {exc}")
            try:
                x = torch.randn(8, 8, device="cuda")
                y = x @ x
                _ = float(y.mean().item())
                print("[OK] cuda tensor op succeeded")
            except Exception as exc:
                print(f"[FAIL] cuda tensor op failed: {exc}")
                ok = False
        else:
            ok = False
    except Exception as exc:
        print(f"[FAIL] import/use torch failed: {exc}")
        ok = False

    _print_title("nvcc")
    nvcc = shutil.which("nvcc")
    if nvcc:
        rc, out = _run([nvcc, "--version"])
        if rc == 0:
            print("[OK] nvcc --version")
            print(out)
        else:
            print("[WARN] nvcc exists but failed")
            print(out)
    else:
        print("[INFO] nvcc not found (this can be normal for runtime-only setup)")

    _print_title("Conclusion")
    if ok:
        print("[OK] GPU stack looks healthy.")
        return 0
    if require_gpu:
        print("[FAIL] GPU stack is not healthy for CUDA inference.")
        return 2
    print("[WARN] GPU checks failed, but --require-gpu=0 so returning success.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

