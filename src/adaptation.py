"""Controlled 3x2 alignment/pseudo-label experiment on frozen ResNet-18 features."""
import os,json,time,argparse,hashlib,random,csv,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCH_HOME',str(ROOT/'.cache/torch'))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')

def run():
 ap=argparse.ArgumentParser();ap.add_argument('--features-only',action='store_true');args=ap.parse_args()
 import numpy as np,torch
 from PIL import Image
 from torchvision.models import resnet18,ResNet18_Weights
 from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef
 torch.set_num_threads(4)
 manifest=json.loads((ROOT/'data/manifest.json').read_text());cfg=json.loads((ROOT/'configs/experiment.json').read_text())
 device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
 feature_path=ROOT/'data/resnet18_features.npz'
 if feature_path.exists():
  data=np.load(feature_path);x=data['features'];ids=data['ids'];assert list(ids)==[r['id'] for r in manifest]
 else:
  weights=ResNet18_Weights.IMAGENET1K_V1;net=resnet18(weights=weights).to(device).eval();net.fc=torch.nn.Identity();pre=weights.transforms()
  parts=[]
  for start in range(0,len(manifest),32):
   batch=torch.stack([pre(Image.open(ROOT/r['path']).convert('RGB')) for r in manifest[start:start+32]]).to(device)
   with torch.inference_mode():parts.append(net(batch).cpu().numpy())
   print('Features',start+len(batch),'/',len(manifest),flush=True)
  x=np.concatenate(parts);np.savez_compressed(feature_path,features=x,ids=np.array([r['id'] for r in manifest]));del net
  if device=='mps':torch.mps.empty_cache()
 if args.features_only:
  print('Frozen feature cache ready',len(x),x.shape,flush=True);return
 selection=json.loads((ROOT/'results/selection.json').read_text());key=selection['model'];raw=ROOT/f'results/raw/{key}_application.jsonl'
 predictions=[json.loads(line) for line in raw.read_text().splitlines()];pred_by_id={r['id']:r for r in predictions if r['split']=='target_adapt'}
 ix={s:np.array([i for i,r in enumerate(manifest) if r['split']==s]) for s in ['source_train','validation','target_adapt','test','new_slice']}
 if {manifest[i]['id'] for i in ix['target_adapt']}!=set(pred_by_id):raise ValueError('Incomplete pseudo-label application')
 # Statistics from the source training subset only.
 mean=x[ix['source_train']].mean(0);std=x[ix['source_train']].std(0).clip(0.1)
 x=((x-mean)/std).astype('float32');xt=torch.tensor(x,device=device)
 ys=torch.tensor([manifest[i]['label'] for i in ix['source_train']],device=device)
 target_rows=[manifest[i] for i in ix['target_adapt']]
 py=torch.tensor([pred_by_id[r['id']]['prediction'] for r in target_rows],device=device)
 mask=torch.tensor([pred_by_id[r['id']]['valid'] and pred_by_id[r['id']]['confidence']>=cfg['pseudo_label_confidence_threshold'] for r in target_rows],device=device)
 out=ROOT/'results/adaptation';out.mkdir(parents=True,exist_ok=True)
 heads=out/'heads';heads.mkdir(exist_ok=True)
 np.savez_compressed(out/'source_standardization.npz',mean=mean,std=std)
 accepted_rows=[pred_by_id[r['id']] for i,r in enumerate(target_rows) if mask[i].item()]
 predicted_counts=collections.Counter(r['prediction'] for r in accepted_rows)
 classes=json.loads((ROOT/'data/classes.json').read_text())
 (out/'pseudo_label_audit.json').write_text(json.dumps({'n_target_adapt':len(target_rows),'accepted':int(mask.sum()),'threshold':cfg['pseudo_label_confidence_threshold'],'model':key,'mode':selection['mode'],'accepted_accuracy_for_evaluation_only':float(np.mean([pred_by_id[r['id']]['prediction']==r['label'] for i,r in enumerate(target_rows) if mask[i].item()])) if mask.any() else None,'accepted_correct_for_evaluation_only':sum(r['prediction']==r['label'] for r in accepted_rows),'predicted_classes_present':len(predicted_counts),'most_frequent_predictions':[{'label_id':label,'class':classes[label],'count':count} for label,count in predicted_counts.most_common(5)]},indent=2))
 class Net(torch.nn.Module):
  def __init__(self):
   super().__init__();self.mapper=torch.nn.Sequential(torch.nn.Linear(512,128),torch.nn.ReLU());self.classifier=torch.nn.Linear(128,65)
  def forward(self,z):
   h=self.mapper(z);return h,self.classifier(h)
 def align(a,b,method):
  if method=='none':return a.sum()*0
  if method=='coral':
   a=a-a.mean(0);b=b-b.mean(0);ca=a.T@a/(len(a)-1);cb=b.T@b/(len(b)-1)
   return ((ca-cb)**2).mean()/4
  # Biased non-negative multi-scale Gaussian MMD. Fixed bandwidths, no test fitting.
  z=torch.cat([a,b]);sq=(z*z).sum(1,keepdim=True);dist=(sq+sq.T-2*z@z.T).clamp_min(0)/z.shape[1]
  kernel=sum(torch.exp(-dist/(2*s*s)) for s in [.5,1,2])/3;n=len(a)
  return kernel[:n,:n].mean()+kernel[n:,n:].mean()-2*kernel[:n,n:].mean()
 results=[];observed=[]
 for seed in [20261006,20261007,20261008]:
  for method in ['none','mmd','coral']:
   for use_pseudo in [False,True]:
    torch.manual_seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed)
    model=Net().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best=-1;best_state=None;best_epoch=0;history=[];started=time.time()
    for epoch in range(40):
     model.train();permutation=rng.permutation(len(ys));losses=[]
     for start in range(0,len(ys),64):
      si=permutation[start:start+64];ti=rng.choice(len(target_rows),len(si),replace=len(target_rows)<len(si))
      hs,logits=model(xt[ix['source_train'][si]]);ht,tlogits=model(xt[ix['target_adapt'][ti]])
      ce=torch.nn.functional.cross_entropy(logits,ys[si]);al=align(hs,ht,method)
      pmask=mask[ti];pl=torch.nn.functional.cross_entropy(tlogits[pmask],py[ti][pmask]) if use_pseudo and pmask.any() else logits.sum()*0
      loss=ce+.1*al+.3*pl;optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach().cpu()))
     model.eval()
     with torch.inference_mode():vp=model(xt[ix['validation']])[1].argmax(1).cpu().numpy()
     truth=[manifest[i]['label'] for i in ix['validation']];val=f1_score(truth,vp,labels=range(65),average='macro',zero_division=0)
     history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'source_validation_f1':float(val)})
     if val>best:best=float(val);best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch+1
    model.load_state_dict(best_state);model.eval()
    from safetensors.torch import save_file
    save_file({k:v.contiguous() for k,v in best_state.items()},str(heads/f'{seed}_{method}_{int(use_pseudo)}.safetensors'))
    for split in ['test','new_slice']:
     with torch.inference_mode():prob=torch.softmax(model(xt[ix[split]])[1],dim=-1).cpu().numpy()
     predicted=prob.argmax(1);truth=np.array([manifest[i]['label'] for i in ix[split]])
     results.append({'seed':seed,'alignment':method,'pseudo_labels':use_pseudo,'split':split,'n':len(truth),'accuracy':float(accuracy_score(truth,predicted)),'macro_f1':float(f1_score(truth,predicted,labels=range(65),average='macro',zero_division=0)),'mcc':float(matthews_corrcoef(truth,predicted)),'best_source_validation_f1':best,'best_epoch':best_epoch,'training_seconds':time.time()-started})
     for j,i in enumerate(ix[split]):observed.append({'seed':seed,'alignment':method,'pseudo_labels':use_pseudo,'split':split,'id':manifest[i]['id'],'label':int(truth[j]),'prediction':int(predicted[j]),'probabilities':prob[j].tolist()})
    (out/f'train_{seed}_{method}_{int(use_pseudo)}.json').write_text(json.dumps(history,indent=2))
    print('ADAPT',results[-2],flush=True)
    del model
    if device=='mps':torch.mps.empty_cache()
 with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in observed))
 (out/'protocol.json').write_text(json.dumps({'backbone':'torchvision ResNet18 IMAGENET1K_V1 frozen','feature_standardization':'source_train only','trainable_head':'512 -> 128 ReLU -> 65','epochs':40,'batch_size':64,'learning_rate':.001,'weight_decay':.0001,'alignment_weight':.1,'pseudo_ce_weight':.3,'checkpoint_selection':'source validation Macro-F1','target_labels_used_for_training':False,'test_images_used_for_adaptation':False,'device':device},indent=2))
if __name__=='__main__':
 from runtime_utils import accelerator_lock
 with accelerator_lock():run()
