"""Read-only prior-evidence/model/hardware audit, saved in a new phase directory."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import urllib.request

import torch

ROOT=Path.cwd();output=ROOT/'transformer-efficiency/results/phase5b/start-audit.json'
if output.exists():raise FileExistsError('Never overwrite audit')
ledger=json.loads((ROOT/'CLAIM_EVIDENCE.json').read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
checks=[dict(path=a['path'],passed=sha(ROOT/a['path'])==a['sha256']) for a in ledger['artifacts']]
assert all(a['passed'] for a in checks)
checkpoint=json.loads((ROOT/'.model-cache/smollm2-135m/htis-checkpoint.json').read_text())
weights=[dict(path=name,passed=sha(ROOT/'.model-cache/smollm2-135m'/name)==digest) for name,digest in checkpoint['files_sha256'].items()]
assert all(a['passed'] for a in weights)
try:
    response=urllib.request.urlopen('https://huggingface.co/api/datasets/THUDM/LongBench',timeout=20)
    acquired=dict(available=True,metadata=json.load(response))
except Exception as e:
    acquired=dict(available=False,error=type(e).__name__+': '+str(e),
        consequence='Official LongBench acquisition unavailable in local execution environment; adapted public-data study only.')
try:
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        processor=winreg.QueryValueEx(key,'ProcessorNameString')[0]
except Exception as e:processor=platform.processor();registry_error=str(e)
record=dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    prior_artifact_count=len(checks),prior_artifacts=checks,model_files=weights,
    baseline_ledger_sha256=sha(ROOT/'CLAIM_EVIDENCE.json'),
    baseline_ledger=ledger,baseline_source_sha256={p.as_posix():sha(p) for p in (ROOT/'transformer-efficiency/src/htis_context').glob('*.py') if 'phase5b' not in p.name},
    platform=platform.platform(),processor=processor,logical_cpus=os.cpu_count(),torch=torch.__version__,
    cuda_available=torch.cuda.is_available(),cuda_build=torch.version.cuda,cuda_device_count=torch.cuda.device_count(),
    nvidia_smi_path=shutil.which('nvidia-smi'),
    prior_hardware_record='transformer-efficiency/results/phase5/hardware-v1.json',
    hardware_query_limit='Native CIM processor/GPU/RAM queries returned Access denied under current sandbox; registry CPU, Torch CUDA and prior same-host hardware record used.',
    official_benchmark_acquisition=acquired,phase6='NOT STARTED',passed=True)
output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:record[k] for k in ['baseline_commit','prior_artifact_count','processor','torch','cuda_available','cuda_device_count','official_benchmark_acquisition']},indent=2))
