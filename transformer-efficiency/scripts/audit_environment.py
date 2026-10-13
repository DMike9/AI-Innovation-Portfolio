"""Record local hardware/dependencies without claiming inaccessible GPU results."""
import ctypes
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

import torch


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in
        ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")]


memory = MemoryStatus()
memory.length = ctypes.sizeof(memory)
physical_memory = None
if os.name == "nt" and ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
    physical_memory = memory.total_phys

def cim_query(query):
    if os.name != "nt":
        return {"available": False, "reason": "CIM is Windows-specific"}
    run = subprocess.run(["powershell", "-NoProfile", "-Command", query],
                         capture_output=True, text=True, timeout=30)
    if run.returncode != 0:
        return {"available": False, "reason": run.stderr.strip()}
    return {"available": True, "data": json.loads(run.stdout)}

result = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
          "python": platform.python_version(), "platform": platform.platform(),
          "processor": platform.processor(), "logical_cpus": os.cpu_count(),
          "physical_memory_bytes": physical_memory, "disk_free_bytes": shutil.disk_usage(".").free,
          "cuda_available": torch.cuda.is_available(), "torch": torch.__version__,
          "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
          "git_status": subprocess.check_output(["git", "status", "--short", "--branch"], text=True),
          "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "execution_device": "cpu",
          "cpu_from_CIM": cim_query("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json"),
          "gpu_from_CIM": cim_query("Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json"),
          "hardware_notes": "CPU execution; GPU metadata only if CIM access succeeds. No ROCm/DirectML claim."}
path = Path("transformer-efficiency/results/phase0/environment.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k not in ("dependencies", "git_status")}, indent=2))
