"""Source-only numerical check; diagnostic outputs never enter benchmark metrics."""
import os,json,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
from runtime_utils import accelerator_lock,model_snapshot
from benchmark import prompt_text

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--dtype',choices=['float16','bfloat16'],required=True);ap.add_argument('--first-only',action='store_true');args=ap.parse_args()
 import torch
 from PIL import Image
 from transformers import AutoProcessor,AutoModelForImageTextToText
 torch.set_num_threads(4)
 cfg=json.loads((ROOT/'configs/experiment.json').read_text());spec=next(s for s in cfg['models'] if s['key']==args.model)
 pins=json.loads((ROOT/'configs/model_revisions.json').read_text());path=model_snapshot(spec['repo'],pins[spec['repo']])
 dtype=getattr(torch,args.dtype);device='mps' if torch.backends.mps.is_available() else 'cpu'
 row=next(r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']=='source_train' and r['label']==0)
 classes=json.loads((ROOT/'data/classes.json').read_text())
 kwargs={'min_pixels':56*56,'max_pixels':448*448} if 'Qwen2' in spec['repo'] else ({'size':{'longest_edge':512}} if 'Smol' in spec['repo'] else {})
 proc=AutoProcessor.from_pretrained(path,local_files_only=True,**kwargs)
 model=AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,dtype=dtype,attn_implementation='eager',low_cpu_mem_usage=True).to(device).eval()
 messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt_text(classes)}]}]
 prompt=proc.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
 inputs=proc(text=prompt,images=[Image.open(ROOT/row['path']).convert('RGB')],return_tensors='pt')
 inputs={k:v.to(device=device,dtype=dtype if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
 stages=[]
 def hook(name):
  def check(module,args,output):
   value=output[0] if isinstance(output,(tuple,list)) else output
   if isinstance(value,torch.Tensor):
    finite=bool(torch.isfinite(value).all().item());stages.append({'module':name,'finite':finite,'max_abs':float(value.detach().abs().max().item()) if finite else None})
  return check
 for name,module in model.named_modules():
  if name.endswith(('visual.patch_embed','visual.blocks.0','visual.merger','lm_head')):module.register_forward_hook(hook(name))
 with torch.inference_mode():
  first=model.generate(**inputs,max_new_tokens=1,do_sample=False,num_beams=1,return_dict_in_generate=True,output_scores=True)
  finite=bool(torch.isfinite(first.scores[0]).all().item())
  generated=model.generate(**inputs,max_new_tokens=40,do_sample=False,num_beams=1) if finite and not args.first_only else first.sequences
 result={'model':args.model,'revision':pins[spec['repo']],'dtype':args.dtype,'device':device,'source_id':row['id'],'first_logits_finite':finite,'stages':stages,'raw':proc.batch_decode(generated[:,inputs['input_ids'].shape[-1]:],skip_special_tokens=True)[0],'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'excluded_from_benchmark':True}
 out=ROOT/f'logs/{args.model}_{args.dtype}_numerics.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
 with accelerator_lock():main()
