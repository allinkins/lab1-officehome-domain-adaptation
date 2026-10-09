"""Metrics: invalid answers remain classification errors; calibration uses valid JSON only."""
import json,numpy as np
from sklearn.metrics import f1_score,matthews_corrcoef,accuracy_score,confusion_matrix

def evaluate(rows,nclasses=65):
 y=np.array([r['label'] for r in rows]);p=np.array([r['prediction'] for r in rows]);v=np.array([r['valid'] for r in rows])
 out={'n':len(y),'accuracy':float(np.mean(y==p)),'macro_f1':float(f1_score(y,p,labels=range(nclasses),average='macro',zero_division=0)),
 'weighted_f1':float(f1_score(y,p,labels=range(nclasses),average='weighted',zero_division=0)),
 'mcc':float(matthews_corrcoef(y,p)),'invalid_rate':float(1-v.mean()),'valid_n':int(v.sum()),
 'latency_median_s':float(np.median([r['latency_s'] for r in rows])),
 'latency_p95_s':float(np.percentile([r['latency_s'] for r in rows],95)),
 'peak_process_rss_gb':max(r.get('process_peak_rss_gb',0) for r in rows)}
 recalls=[float(np.mean(p[y==k]==k)) for k in range(nclasses) if (y==k).any()]
 out['balanced_accuracy']=float(np.mean(recalls))
 if v.any():
  conf=np.array([r['confidence'] for r in rows if r['valid']]);correct=(y[v]==p[v]).astype(float)
  ece=0;bins=[]
  bin_ids=np.minimum((conf*10).astype(int),9)
  for bin_id in range(10):
   lo=bin_id/10
   mask=bin_ids==bin_id
   if mask.any():
    acc=float(correct[mask].mean());c=float(conf[mask].mean());ece+=mask.mean()*abs(acc-c);bins.append({'lo':float(lo),'n':int(mask.sum()),'accuracy':acc,'confidence':c})
  out.update(ece=float(ece),binary_brier=float(np.mean((conf-correct)**2)),calibration_bins=bins)
 else:out.update(ece=None,binary_brier=None,calibration_bins=[])
 return out

def bootstrap_f1(rows,repeats=2000,seed=20261006):
 y=np.array([r['label'] for r in rows]);p=np.array([r['prediction'] for r in rows]);rng=np.random.default_rng(seed)
 groups=[np.where(y==k)[0] for k in np.unique(y)]
 if all(len(g)==1 for g in groups):return None
 values=[]
 for _ in range(repeats):
  ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
  values.append(f1_score(y[ix],p[ix],labels=range(65),average='macro',zero_division=0))
 return [float(x) for x in np.quantile(values,[0.025,0.975])]

def paired_bootstrap_difference(a,b,repeats=2000,seed=20261006):
 da={r['id']:r for r in a};db={r['id']:r for r in b}
 if set(da)!=set(db):raise ValueError('Paired comparison requires identical observations')
 ids=sorted(da);y=np.array([da[i]['label'] for i in ids]);pa=np.array([da[i]['prediction'] for i in ids]);pb=np.array([db[i]['prediction'] for i in ids])
 groups=[np.where(y==k)[0] for k in np.unique(y)];rng=np.random.default_rng(seed);v=[]
 for _ in range(repeats):
  ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
  v.append(f1_score(y[ix],pa[ix],labels=range(65),average='macro',zero_division=0)-f1_score(y[ix],pb[ix],labels=range(65),average='macro',zero_division=0))
 return [float(x) for x in np.quantile(v,[.025,.975])]
