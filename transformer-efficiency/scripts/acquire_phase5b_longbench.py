"""Acquire pinned public LongBench release; no model inference or labels tuning."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT=Path.cwd();P=ROOT/'transformer-efficiency/results/phase5b'
metadata=json.loads((P/'official-acquisition-retry.json').read_text())['metadata']
revision=metadata['sha']
cache=ROOT/'.model-cache/phase5b-longbench';cache.mkdir(exist_ok=False)
url=f'https://huggingface.co/datasets/THUDM/LongBench/resolve/{revision}/data.zip'
with urllib.request.urlopen(url,timeout=60) as stream:
    with (cache/'data.zip').open('wb') as destination:
        while block:=stream.read(1024**2):destination.write(block)
repo=json.load(urllib.request.urlopen('https://api.github.com/repos/THUDM/LongBench/commits/main',timeout=30))['sha']
for name in ['dataset2prompt.json','dataset2maxlen.json']:
    u=f'https://raw.githubusercontent.com/THUDM/LongBench/{repo}/LongBench/config/{name}'
    (cache/name).write_bytes(urllib.request.urlopen(u,timeout=30).read())
with zipfile.ZipFile(cache/'data.zip') as archive:
    names=archive.namelist()
    for dataset in ['hotpotqa','2wikimqa']:
        name=next(n for n in names if n.endswith('/'+dataset+'.jsonl') or n==dataset+'.jsonl')
        # Controlled filenames only; never extract arbitrary archive paths.
        (cache/(dataset+'.jsonl')).write_bytes(archive.read(name))
record=dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),dataset_revision=revision,repository_revision=repo,
    source_url=url,files={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in cache.iterdir() if p.is_file()},
    datasets=['hotpotqa','2wikimqa'],available=True,scope='Pinned complete public official-source data; unselected release remains in ignored cache')
(P/'official-acquisition-success.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
