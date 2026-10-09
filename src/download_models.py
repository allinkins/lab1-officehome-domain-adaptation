from pathlib import Path
import os,json,argparse,time
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'.cache'/'huggingface')
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','60')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT','30')
from huggingface_hub import snapshot_download,HfApi
p=argparse.ArgumentParser();p.add_argument('--model',default='smol256');a=p.parse_args()
cfg=json.loads((ROOT/'configs/experiment.json').read_text())
for s in cfg['models']:
 if a.model not in ('all',s['key']):continue
 path=ROOT/'configs/model_revisions.json';revs=json.loads(path.read_text()) if path.exists() else {}
 revision=revs.get(s['repo']) or HfApi().model_info(s['repo']).sha
 revs[s['repo']]=revision;path.write_text(json.dumps(revs,indent=2))
 try:
  from runtime_utils import model_snapshot
  local=Path(model_snapshot(s['repo'],revision))
  if all((local/name).exists() for name in ['tokenizer.json','tokenizer_config.json','preprocessor_config.json']):
   print('Reusing complete local snapshot',s['key'],revision,flush=True);continue
 except RuntimeError:pass
 print('Downloading',s['repo'],revision,flush=True)
 for attempt in range(5):
  try:
   snapshot_download(s['repo'],revision=revision,allow_patterns=['*.json','*.txt','*.model','*.jinja','*.safetensors'],ignore_patterns=['onnx/*','coreml/*'],max_workers=2)
   break
  except Exception as exc:
   print('DOWNLOAD RETRY',attempt+1,type(exc).__name__,str(exc)[-220:],flush=True)
   if attempt==4:raise
   time.sleep(10*(attempt+1))
 print('Downloaded',s['key'],flush=True)
