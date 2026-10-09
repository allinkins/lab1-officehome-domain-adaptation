"""Sequential, resumable coordinator; never overlaps model inference jobs."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
def complete(key,tag='benchmark',n=416):
 p=ROOT/f'results/raw/{key}_{tag}.jsonl'
 if not p.exists():return False
 try:return len({(r['id'],r['mode'],r['variant']) for r in map(json.loads,p.read_text().splitlines())})==n
 except ValueError:return False
def call(script,*args):
 print('RUN',script,*args,flush=True)
 result=subprocess.run([sys.executable,str(ROOT/'src'/script),*args],cwd=ROOT)
 if result.returncode:raise RuntimeError(f'{script} failed with {result.returncode}')
if __name__=='__main__':
 # smol500 was started interactively. Wait for its completed result file.
 while not complete('smol500'):time.sleep(10)
 time.sleep(3)
 failures=[]
 for key in ['smol2_500','smol2_2200','llava500','qwen2_2000','qwen25_3000']:
  if complete(key):continue
  try:
   call('download_models.py','--model',key)
   call('benchmark.py','--model',key)
  except Exception as exc:
   failures.append({'model':key,'error':str(exc)});print('FAILED',key,str(exc),flush=True)
  (ROOT/'logs/coordinator_failures.json').write_text(json.dumps(failures,indent=2))
 if failures:raise SystemExit('Model failures require inspection before aggregation')
 call('summarize.py')
 selection=json.loads((ROOT/'results/selection.json').read_text())
 if not complete(selection['model'],'application',455):call('benchmark.py','--model',selection['model'],'--application')
 call('adaptation.py')
 if not (ROOT/'results/peft/protocol.json').exists() or json.loads((ROOT/'results/peft/protocol.json').read_text()).get('status')!='completed':
  call('train_lora.py','--probe')
  call('train_lora.py')
 if not complete('smol500','peft',221):call('benchmark.py','--model','smol500','--adapter',str(ROOT/'results/peft/adapter'))
 call('summarize_peft.py')
 call('analyze_results.py')
 print('ALL CORE EXPERIMENTS COMPLETE',flush=True)
