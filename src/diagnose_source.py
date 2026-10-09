"""Technical caption diagnostic on training images; excluded from evaluation."""
import os,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from PIL import Image
from transformers import AutoProcessor,AutoModelForImageTextToText
torch.set_num_threads(4)
repo='HuggingFaceTB/SmolVLM-500M-Instruct';revision=json.loads((ROOT/'configs/model_revisions.json').read_text())[repo]
device='mps' if torch.backends.mps.is_available() else 'cpu';dtype=torch.float16 if device=='mps' else torch.float32
processor=AutoProcessor.from_pretrained(repo,revision=revision,size={'longest_edge':512})
model=AutoModelForImageTextToText.from_pretrained(repo,revision=revision,torch_dtype=dtype,attn_implementation='eager').to(device).eval()
rows=json.loads((ROOT/'data/manifest.json').read_text());results=[]
for label in [0,10]:
 row=next(r for r in rows if r['split']=='source_train' and r['label']==label)
 for query in ['Describe this image in one sentence.','What is the main object in this image?']:
  messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':query}]}]
  prompt=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
  inputs=processor(text=prompt,images=[Image.open(ROOT/row['path']).convert('RGB')],return_tensors='pt');inputs={k:v.to(device=device,dtype=dtype if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
  with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=40,do_sample=False)
  answer=processor.batch_decode(generated[:,inputs['input_ids'].shape[-1]:],skip_special_tokens=True)[0]
  result={'id':row['id'],'split':'source_train','label':label,'query':query,'answer':answer,'excluded_from_all_metrics':True};results.append(result);print(result,flush=True)
(ROOT/'logs/source_caption_diagnostic.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
