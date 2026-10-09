"""Stream Office-Home, audit all images, retain a deterministic experiment subset."""
from pathlib import Path
import os, json, hashlib, heapq, collections, time
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('HF_DATASETS_CACHE', str(ROOT / '.cache' / 'datasets'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
def dump(name,value):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2))
def main():
 from datasets import load_dataset
 from huggingface_hub import HfApi
 from PIL import Image,ImageOps
 import imagehash
 out=ROOT/'data';images=out/'images';images.mkdir(parents=True,exist_ok=True)
 manifest_path=out/'manifest.json'
 if manifest_path.exists():
  existing=json.loads(manifest_path.read_text())
  if all((ROOT/r['path']).exists() for r in existing):
   print('Preserving frozen manifest.',flush=True);return
  print('Restoring missing images from the pinned dataset revision.',flush=True)
  manifest_path.unlink()
 repo='flwrlabs/office-home'
 metadata_path=out/'download_metadata.json'
 revision=json.loads(metadata_path.read_text())['revision'] if metadata_path.exists() else HfApi().dataset_info(repo).sha
 ds=load_dataset(repo,revision=revision,split='train',streaming=True)
 classes=ds.features['label'].names;dump('data/classes.json',classes)
 records=[];pools=collections.defaultdict(list);counts=collections.Counter();errors=[];started=time.time()
 for idx,row in enumerate(ds):
  try:
   im=ImageOps.exif_transpose(row['image']).convert('RGB');w,h=im.size
   sha=hashlib.sha256(f'{w}x{h}:'.encode()+im.tobytes()).hexdigest();ph=str(imagehash.phash(im))
   domain=row['domain'].replace(' ','_').replace('-','_');label=int(row['label']);counts[(domain,label)]+=1
   rank=int(hashlib.sha256(('20261006:'+sha).encode()).hexdigest(),16)
   records.append({'id':idx,'domain':domain,'label':label,'class_name':classes[label],'width':w,'height':h,'rgb_sha256':sha,'phash':ph,'rank':rank})
   heap=pools[(domain,label)]
   if len(heap)<16 or rank < -heap[0][0]:
    im.thumbnail((448,448),Image.Resampling.LANCZOS);im.save(images/f'{idx:05d}.jpg',quality=95)
    if len(heap)>=16:
     _,old=heapq.heappop(heap);(images/f'{old:05d}.jpg').unlink(missing_ok=True)
    heapq.heappush(heap,(-rank,idx))
   if (idx+1)%1000==0:print(f'Audited {idx+1} images in {time.time()-started:.0f}s',flush=True)
  except Exception as e:errors.append({'index':idx,'error':str(e)})
 parent=list(range(len(records)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[b]=a
 seen={};exact=[];near=[];buckets=collections.defaultdict(list)
 bands=[(0,13),(13,13),(26,13),(39,13),(52,12)]
 for i,r in enumerate(records):
  sha=r['rgb_sha256']
  if sha in seen:union(i,seen[sha]);exact.append([records[seen[sha]]['id'],r['id']])
  else:seen[sha]=i
  value=int(r['phash'],16);candidates=set()
  for band,(shift,bits) in enumerate(bands):candidates.update(buckets[(band,(value>>shift)&((1<<bits)-1))])
  for j in candidates:
   other=records[j]
   if sha==other['rgb_sha256']:continue
   if (value^int(other['phash'],16)).bit_count()<=4:
    ratio=(r['width']/r['height'])/(other['width']/other['height'])
    if 0.9<=ratio<=1.1:union(i,j);near.append([other['id'],r['id']])
  for band,(shift,bits) in enumerate(bands):buckets[(band,(value>>shift)&((1<<bits)-1))].append(i)
 groups=collections.defaultdict(list)
 for i,r in enumerate(records):groups[find(i)].append(r)
 conflict={g for g,rs in groups.items() if len({r['label'] for r in rs})>1}
 id_group={r['id']:g for g,rs in groups.items() for r in rs};kept={i for pool in pools.values() for _,i in pool}
 candidates=collections.defaultdict(list)
 for r in records:
  if r['id'] in kept and id_group[r['id']] not in conflict:candidates[(r['domain'],r['label'])].append(r)
 allocation={'Product':[('source_train',8),('validation',2)],'Clipart':[('target_adapt',3),('test',2)],'Art':[('new_slice',2)],'Real_World':[('external',2)]}
 manifest=[];used=set();shortages=[]
 for domain,splits in allocation.items():
  for label in range(len(classes)):
   available=sorted(candidates[(domain,label)],key=lambda x:x['rank'])
   for split,n in splits:
    selected=[]
    for r in available:
     g=id_group[r['id']]
     if g in used:continue
     selected.append(r);used.add(g)
     if len(selected)==n:break
    if len(selected)!=n:shortages.append({'domain':domain,'label':label,'split':split,'requested':n,'actual':len(selected)})
    for r in selected:
     r=dict(r);r.pop('rank');r['split']=split;r['group']=id_group[r['id']];r['path']=f'data/images/{r["id"]:05d}.jpg'
     r['file_sha256']=hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest();manifest.append(r)
 selected_ids={r['id'] for r in manifest}
 for i in kept-selected_ids:(images/f'{i:05d}.jpg').unlink(missing_ok=True)
 dump('data/manifest.json',manifest);dump('data/full_audit.json',records)
 dump('data/duplicate_pairs.json',{'exact':exact,'near_candidates':near,'conflicting_groups':sorted(conflict)})
 audit={'repository':repo,'revision':revision,'streaming':True,'classes':classes,'rows_decoded':len(records),'errors':errors,
 'domain_counts':dict(collections.Counter(r['domain'] for r in records)),
 'class_domain_counts':[{'domain':d,'label':l,'n':n} for (d,l),n in sorted(counts.items())],
 'exact_duplicate_pairs':len(exact),'near_duplicate_candidate_pairs':len(near),'conflicting_groups_excluded':len(conflict),
 'selected_rows':len(manifest),'split_counts':dict(collections.Counter(r['split'] for r in manifest)),
 'shortages':shortages,'elapsed_s':time.time()-started,
 'near_duplicate_policy':'pHash Hamming<=4 and aspect ratio within 10%; conservative candidates, not proven duplicates',
 'license_note':'Original Office-Home Fair Use Notice; academic/non-commercial research',
 'official_source':'https://www.hemanthdv.org/officeHomeDataset.html'}
 dump('data/download_metadata.json',audit)
 dump('data/split_lock.json',{'manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),'seed':20261006})
 print(json.dumps({k:v for k,v in audit.items() if k not in ['classes','class_domain_counts']},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
