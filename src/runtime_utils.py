"""Keep inference offline and serialize accelerator jobs on the experiment host."""
from pathlib import Path
import json,fcntl
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parents[1]
def model_snapshot(repo,revision):
 path=ROOT/'.cache/huggingface/hub'/('models--'+repo.replace('/','--'))/'snapshots'/revision
 if not (path/'config.json').exists():raise RuntimeError(f'Download model first: {repo}')
 index=path/'model.safetensors.index.json'
 expected=set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
 missing=[s for s in expected if not (path/s).exists()]
 if missing:raise RuntimeError(f'Incomplete model download {repo}: {missing}')
 return str(path)
@contextmanager
def accelerator_lock():
 with (ROOT/'logs/accelerator.lock').open('a') as file:
  fcntl.flock(file,fcntl.LOCK_EX)
  try:yield
  finally:fcntl.flock(file,fcntl.LOCK_UN)
