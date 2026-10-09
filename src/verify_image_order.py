"""Source-only CPU check that mixed image sizes preserve order in Qwen few-shot input."""
import json,os,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
import torch
from PIL import Image
from transformers import AutoProcessor
from runtime_utils import model_snapshot
torch.set_num_threads(1)
rows=json.loads((ROOT/'data/manifest.json').read_text());pins=json.loads((ROOT/'configs/model_revisions.json').read_text())
examples=[next(r for r in rows if r['split']=='source_train' and r['label']==k) for k in [0,22,43]]
other=max((r for r in rows if r['split']=='source_train' and r['id'] not in {e['id'] for e in examples}),key=lambda r:max(r['width']/r['height'],r['height']/r['width']))
chosen=examples+[other];images=[Image.open(ROOT/r['path']).convert('RGB') for r in chosen]
results=[]
for repo in ['Qwen/Qwen2-VL-2B-Instruct','Qwen/Qwen2.5-VL-3B-Instruct']:
 proc=AutoProcessor.from_pretrained(model_snapshot(repo,pins[repo]),local_files_only=True,min_pixels=56*56,max_pixels=448*448)
 combined=proc.image_processor(images=images,return_tensors='pt')
 separate=[proc.image_processor(images=[im],return_tensors='pt') for im in images]
 expected=torch.cat([x['pixel_values'] for x in separate]);grids=torch.cat([x['image_grid_thw'] for x in separate])
 delta=float((combined['pixel_values']-expected).abs().max())
 assert torch.equal(combined['image_grid_thw'],grids)
 assert torch.allclose(combined['pixel_values'],expected,atol=1e-6,rtol=1e-6)
 results.append({'repo':repo,'revision':pins[repo],'source_ids':[r['id'] for r in chosen],'saved_image_sizes':[list(im.size) for im in images],'grids':grids.tolist(),'max_pixel_tensor_difference':delta,'order_matches_individual_processing':True,'excluded_from_model_metrics':True})
(ROOT/'logs/image_order_verification.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
