"""Real multimodal generation benchmark. Every line is a raw observed response."""
import os,sys,json,time,hashlib,random,argparse,resource,platform,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache'/'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')

def load_json(path):return json.loads((ROOT/path).read_text())
def parse_answer(raw,n=65):
 try:
  v=json.loads(raw.strip())
  if not isinstance(v,dict) or set(v)!={'label_id','confidence'}:raise ValueError('schema')
  label=v['label_id'];confidence=v['confidence']
  if type(label) is not int or not 0<=label<n:raise ValueError('class')
  if type(confidence) not in (int,float) or not 0<=confidence<=1:raise ValueError('confidence')
  return {'valid':True,'prediction':label,'confidence':float(confidence)}
 except Exception as e:return {'valid':False,'prediction':-1,'confidence':None,'parse_error':str(e)}

def sampled(rows,split,per_class=None):
 result=[]
 for label in range(65):
  items=sorted([r for r in rows if r['split']==split and r['label']==label],key=lambda r:r['rgb_sha256'])
  result.extend(items if per_class is None else items[:per_class])
 return result

def prompt_text(classes,variant='main'):
 choices='; '.join(f'{i}={name.replace("_"," ")}' for i,name in enumerate(classes))
 if variant=='ru':
  instruction='Определи основной предмет на последнем изображении. Выбери ровно один идентификатор из списка классов.'
 elif variant=='paraphrase':
  instruction='Which listed object category best describes the main object in the final picture? Select exactly one category ID.'
 else:instruction='Classify the main object in the final image. Choose exactly one label ID from the list.'
 return instruction+'\nClasses: '+choices+'\nReturn only valid JSON with exactly two keys: "label_id" (integer) and "confidence" (number from 0 to 1, your estimated probability that your answer is correct). No explanations or markdown.'

def sync(torch,device):
 if device=='mps':torch.mps.synchronize()
 elif device=='cuda':torch.cuda.synchronize()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--application',action='store_true');ap.add_argument('--adapter');args=ap.parse_args()
 if args.adapter and (args.application or args.smoke):raise ValueError('Adapter evaluation is a separate exploratory experiment')
 import torch,transformers,psutil
 from PIL import Image
 from huggingface_hub import HfApi
 from transformers import AutoProcessor,AutoModelForImageTextToText
 cfg=load_json('configs/experiment.json');spec=next(s for s in cfg['models'] if s['key']==args.model)
 rows=load_json('data/manifest.json');classes=load_json('data/classes.json')
 revision_file=ROOT/'configs'/'model_revisions.json'
 revisions=json.loads(revision_file.read_text()) if revision_file.exists() else {}
 revision=revisions.get(spec['repo']) or HfApi().model_info(spec['repo']).sha
 revisions[spec['repo']]=revision;revision_file.write_text(json.dumps(revisions,indent=2))
 random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
 torch.set_num_threads(4)
 device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
 dtype=torch.float16 if device!='cpu' else torch.float32
 # Source-only diagnostics proved fp16 Qwen logits contain NaN on MPS.
 # Use the checkpoint's native bf16 precision; preserve the failed run separately.
 if 'Qwen2' in spec['repo'] and device!='cpu':dtype=torch.bfloat16
 print(f'Loading {spec["repo"]}@{revision} on {device}',flush=True)
 kwargs={}
 if 'Smol' in spec['repo']:kwargs={'size':{'longest_edge':512}}
 if 'Qwen2' in spec['repo']:kwargs={'min_pixels':56*56,'max_pixels':448*448}
 from runtime_utils import model_snapshot
 local=model_snapshot(spec['repo'],revision)
 processor=AutoProcessor.from_pretrained(local,local_files_only=True,**kwargs)
 model=AutoModelForImageTextToText.from_pretrained(local,local_files_only=True,torch_dtype=dtype,attn_implementation='eager',low_cpu_mem_usage=True).to(device).eval()
 if args.adapter:
  from peft import PeftModel
  model=PeftModel.from_pretrained(model,args.adapter).eval()
 examples=[]
 for label in [0,22,43]:
  ex=next(r for r in rows if r['split']=='source_train' and r['label']==label)
  examples.append(ex)
 def infer(row,mode,variant,check_numerics=False):
  messages=[];images=[]
  if mode=='few':
   for ex in examples:
    images.append(Image.open(ROOT/ex['path']).convert('RGB'))
    messages.append({'role':'user','content':[{'type':'image'},{'type':'text','text':'Identify the object using the class IDs in the final instruction.'}]})
    messages.append({'role':'assistant','content':[{'type':'text','text':json.dumps({'label_id':ex['label'],'confidence':1.0})}]})
  images.append(Image.open(ROOT/row['path']).convert('RGB'))
  instruction=prompt_text(classes,variant)
  messages.append({'role':'user','content':[{'type':'image'},{'type':'text','text':instruction}]})
  sync(torch,device);started=time.perf_counter()
  prompt=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
  inputs=processor(text=prompt,images=images,return_tensors='pt')
  inputs={k:v.to(device=device,dtype=dtype if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
  input_len=inputs['input_ids'].shape[-1]
  if device=='cuda':torch.cuda.reset_peak_memory_stats()
  with torch.inference_mode():
   generated=model.generate(**inputs,max_new_tokens=cfg['max_new_tokens'],do_sample=False,num_beams=1,use_cache=True,return_dict_in_generate=check_numerics,output_scores=check_numerics)
   if check_numerics:
    if any(torch.isnan(s).any().item() or torch.isposinf(s).any().item() for s in generated.scores):raise RuntimeError('Nonfinite warmup logits: numerical failure, not an invalid model answer')
    generated=generated.sequences
  sync(torch,device);elapsed=time.perf_counter()-started
  raw=processor.batch_decode(generated[:,input_len:],skip_special_tokens=True)[0].strip()
  result={'id':row['id'],'split':row['split'],'domain':row['domain'],'label':row['label'],'model':args.model,'repo':spec['repo'],'revision':revision,
  'mode':mode,'variant':variant,'raw':raw,'latency_s':elapsed,'input_tokens':input_len,'output_tokens':generated.shape[-1]-input_len,
  'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'example_ids':[r['id'] for r in examples] if mode=='few' else [],
  'rss_gb':psutil.Process().memory_info().rss/1e9,'process_peak_rss_gb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1e9 if sys.platform=='darwin' else 1e6),
  'mps_allocated_end_gb':torch.mps.driver_allocated_memory()/1e9 if device=='mps' else None,
  'cuda_peak_gb':torch.cuda.max_memory_allocated()/1e9 if device=='cuda' else None,
  'device':device,'image_width':row['width'],'image_height':row['height'],'instruction_language':'ru' if variant=='ru' else 'en',
  'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
  result.update(parse_answer(raw,len(classes)))
  return result
 warmup=next(r for r in rows if r['split']=='source_train' and r['id'] not in {e['id'] for e in examples})
 w=infer(warmup,'zero','main',check_numerics=True);print('WARMUP',w['raw'],round(w['latency_s'],2),flush=True)
 folder=ROOT/'results'/'raw';folder.mkdir(parents=True,exist_ok=True)
 if args.smoke:
  outputs=[w,infer(warmup,'few','main')]
  (folder/f'{args.model}_smoke.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2));print(outputs[-1]['raw'],outputs[-1]['latency_s'],flush=True);return
 tasks=[]
 if args.application:
  selection=load_json('results/selection.json')
  if selection['model']!=args.model:raise ValueError('Application must use model frozen by validation')
  for split in ['target_adapt','new_slice','external']:
   tasks.extend((r,selection['mode'],'main') for r in sampled(rows,split))
  tag='application'
 else:
  for split,n in [('validation',cfg['validation_per_class']),('test',cfg['test_per_class'])]:
   for mode in (['zero'] if args.adapter else ['zero','few']):tasks.extend((r,mode,'main') for r in sampled(rows,split,n))
  sensitivity=sampled(rows,'test',1)
  rng=random.Random(cfg['seed']);rng.shuffle(sensitivity);sensitivity=sensitivity[:13]
  for variant in ['paraphrase','ru']:tasks.extend((r,'zero',variant) for r in sensitivity)
  tag='peft' if args.adapter else 'benchmark'
 out=folder/f'{args.model}_{tag}.jsonl'
 done=set()
 if out.exists():
  for line in out.read_text().splitlines():
   r=json.loads(line);done.add((r['id'],r['mode'],r['variant']))
 metadata={'model':spec,'revision':revision,'torch':torch.__version__,'transformers':transformers.__version__,'python':sys.version,'platform':platform.platform(),'device':device,'dtype':str(dtype),'attention_backend':'eager','generation':cfg['generation'],'max_new_tokens':cfg['max_new_tokens'],'parameters':sum(p.numel() for p in model.parameters()),'processor_overrides':kwargs,'warmup_excluded':w,'warmup_numerics_passed':True,'adapter':args.adapter}
 (ROOT/'logs'/f'{args.model}_{tag}_environment.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
 with out.open('a',buffering=1) as f:
  for index,(row,mode,variant) in enumerate(tasks):
   if (row['id'],mode,variant) in done:continue
   r=infer(row,mode,variant);f.write(json.dumps(r,ensure_ascii=False)+'\n')
   if index%10==0:print(f'{args.model} {index+1}/{len(tasks)} {mode} {variant} {r["latency_s"]:.2f}s {r["raw"]}',flush=True)
 del model;gc.collect()
 if device=='mps':torch.mps.empty_cache()
 print('DONE',args.model,tag,len(tasks),flush=True)
if __name__=='__main__':
 from runtime_utils import accelerator_lock
 with accelerator_lock():main()
