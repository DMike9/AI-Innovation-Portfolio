"""Temporary process-owned Windows idle-sleep request for controlled CPU runs."""
import ctypes
from datetime import datetime,timezone
import json
from pathlib import Path
import time

root=Path('transformer-efficiency/results/phase5b')
record=root/'keep-awake-record-v1.json'
stop=root/'release-keep-awake.flag'
if record.exists():raise FileExistsError('Never overwrite power-request record')
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint32]
kernel.SetThreadExecutionState.restype=ctypes.c_uint32
previous=kernel.SetThreadExecutionState(0x80000001)  # continuous + system required
if not previous:raise ctypes.WinError(ctypes.get_last_error())
state=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),active=True,
    scope='Temporary process/thread-owned idle-sleep request; no display request or persistent power-setting changes; cleared on exit',
    reason='Avoid host idle suspension during authorized controlled CPU experiments')
record.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
print('Temporary benchmark sleep-prevention request active.',flush=True)
try:
    while not stop.exists():time.sleep(10)
finally:
    kernel.SetThreadExecutionState(0x80000000)
    state.update(active=False,released_at_utc=datetime.now(timezone.utc).isoformat())
    record.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
