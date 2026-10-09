"""Exploratory source-only LoRA extension. Kept out of the frozen seven-model selection."""
import os,json,time,random,resource,sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from benchmark import prompt_text,sampled
def main():
 import torch,peft,transformers
 from transformers import AutoModelForImageTextToText,AutoProcessor
 from peft import LoraConfig,get_peft_model
 from PIL import Image
 ap=argparse.ArgumentParser();ap.add_argument('--probe',action='store_true');args=ap.parse_args()
 cfg=json.loads((ROOT/'configs/experiment.json').read_text());rows=json.loads((ROOT/'data/manifest.json').read_text());classes=json.loads((ROOT/'data/classes.json').read_text())
 spec=next(s for s in cfg['models'] if s['key']=='smol500');revision=json.loads((ROOT/'configs/model_revisions.json').read_text())[spec['repo']]
 device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu');dtype=torch.float16 if device!='cpu' else torch.float32
 torch.set_num_threads(4);torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
 out=ROOT/'results/peft';out.mkdir(parents=True,exist_ok=True)
 if (out/'protocol.json').exists() and (out/'adapter/adapter_model.safetensors').exists():
  existing=json.loads((out/'protocol.json').read_text())
  if existing.get('status')=='completed':print('Reusing completed fixed-protocol LoRA adapter',flush=True);return
 protocol={'model':'smol500','revision':revision,'rank':8,'alpha':16,'dropout':0.05,'target':'text-model q_proj and v_proj only','epochs':2,'micro_batch':1,'gradient_accumulation':8,'learning_rate':.0001,'weight_decay':.01,'clip_norm':1.,'seed':cfg['seed'],'checkpoint':'lowest source-validation response-token cross entropy after each epoch','data':'520 source_train; 65 source validation; no target labels','exploratory_extension':True,'excluded_from_primary_model_selection':True,'training_confidence_target':1.0,'confidence_caveat':'Supervised targets teach format with confidence 1.0; this can inflate stated confidence and is not calibration training.','device':device,'peft':peft.__version__,'transformers':transformers.__version__,'torch':torch.__version__}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 from runtime_utils import model_snapshot
 local=model_snapshot(spec['repo'],revision)
 processor=AutoProcessor.from_pretrained(local,local_files_only=True,size={'longest_edge':512})
 base=AutoModelForImageTextToText.from_pretrained(local,local_files_only=True,torch_dtype=dtype,attn_implementation='eager').to(device)
 targets=[n for n,m in base.named_modules() if isinstance(m,torch.nn.Linear) and 'text_model' in n and n.endswith(('q_proj','v_proj'))]
 assert targets and not any('vision_model' in n for n in targets)
 model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=targets,bias='none'))
 model.peft_config['default'].base_model_name_or_path=spec['repo']
 params=[p for p in model.parameters() if p.requires_grad];protocol.update(trainable_parameters=sum(p.numel() for p in params),total_parameters=sum(p.numel() for p in model.parameters()),target_modules=targets)
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
 opt=torch.optim.AdamW(params,lr=.0001,weight_decay=.01)
 instruction=prompt_text(classes)
 def prepare(row):
  image=Image.open(ROOT/row['path']).convert('RGB')
  messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':instruction}]}]
  prefix=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
  pinputs=processor(text=prefix,images=[image],return_tensors='pt');n=pinputs['input_ids'].shape[-1]
  # Tokenize continuation separately so BPE cannot merge across the generation boundary.
  target=' '+json.dumps({'label_id':row['label'],'confidence':1.0})+'<end_of_utterance>'
  target_ids=processor.tokenizer(target,add_special_tokens=False,return_tensors='pt')['input_ids']
  inputs=dict(pinputs);inputs['input_ids']=torch.cat([pinputs['input_ids'],target_ids],dim=1)
  inputs['attention_mask']=torch.cat([pinputs['attention_mask'],torch.ones_like(target_ids)],dim=1)
  labels=inputs['input_ids'].clone();labels[:,:n]=-100
  assert (labels!=-100).sum()>0
  inputs={k:v.to(device=device,dtype=dtype if v.is_floating_point() else v.dtype) for k,v in inputs.items()};inputs['labels']=labels.to(device)
  return inputs
 train=sampled(rows,'source_train');val=sampled(rows,'validation',1)
 started=time.perf_counter();history=[];best=float('inf');opt.zero_grad()
 for epoch in range(2):
  random.Random(cfg['seed']+epoch).shuffle(train);model.train();losses=[]
  for i,row in enumerate(train):
   batch=prepare(row);loss=model(**batch,use_cache=False).loss
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
   (loss/8).backward();losses.append(float(loss.detach().cpu()))
   if (i+1)%8==0 or i+1==len(train):
    norm=torch.nn.utils.clip_grad_norm_(params,1.)
    if not torch.isfinite(norm):raise RuntimeError('Nonfinite adapter gradient')
    opt.step();opt.zero_grad()
   if i%25==0:print('LORA',epoch+1,i+1,float(loss.detach().cpu()),flush=True)
   if args.probe and i==7:
    if device=='mps':torch.mps.synchronize()
    info={'status':'probe_succeeded','micro_steps':8,'seconds':time.perf_counter()-started,'process_peak_rss_gb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1e9 if sys.platform=='darwin' else 1e6),'device':device,'loss':losses[-1]};(out/'probe.json').write_text(json.dumps(info,indent=2));print(info,flush=True);return
  model.eval();vl=[]
  with torch.inference_mode():
   for row in val:vl.append(float(model(**prepare(row),use_cache=False).loss.cpu()))
  avg=sum(vl)/len(vl);rec={'epoch':epoch+1,'training_ce':sum(losses)/len(losses),'validation_ce':avg,'elapsed_seconds':time.perf_counter()-started};history.append(rec)
  if avg<best:best=avg;model.save_pretrained(out/'adapter');processor.save_pretrained(out/'adapter');protocol['selected_epoch']=epoch+1
  (out/'history.json').write_text(json.dumps(history,indent=2));print('LORA EPOCH',rec,flush=True)
 protocol.update(status='completed',training_seconds=time.perf_counter()-started,peak_process_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1e9 if sys.platform=='darwin' else 1e6))
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
if __name__=='__main__':
 from runtime_utils import accelerator_lock
 with accelerator_lock():main()
