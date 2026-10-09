"""Validate that tables represent complete, paired, correctly split real runs."""
import json,hashlib,collections,csv,math
from pathlib import Path
from benchmark import sampled,parse_answer
from metrics import evaluate
ROOT=Path(__file__).resolve().parents[1]
def j(p):return json.loads((ROOT/p).read_text())
def main():
 cfg=j('configs/experiment.json');manifest=j('data/manifest.json');lookup={r['id']:r for r in manifest};pins=j('configs/model_revisions.json')
 assert hashlib.sha256((ROOT/'configs/experiment.json').read_bytes()).hexdigest()==(ROOT/'configs/protocol_sha256.txt').read_text().strip()
 assert hashlib.sha256((ROOT/'data/manifest.json').read_bytes()).hexdigest()==j('data/split_lock.json')['manifest_sha256']
 counts={};validation_candidates=[]
 for spec in cfg['models']:
  rows=[json.loads(s) for s in (ROOT/f'results/raw/{spec["key"]}_benchmark.jsonl').read_text().splitlines()]
  keys={(r['id'],r['mode'],r['variant']) for r in rows};assert len(keys)==len(rows)==416
  groups=collections.defaultdict(list)
  for r in rows:
   original=lookup[r['id']];assert r['label']==original['label'] and r['split']==original['split'] and r['domain']==original['domain'];assert r['revision']==pins[spec['repo']]
   parsed=parse_answer(r['raw']);assert all(r[k]==parsed[k] for k in ['valid','prediction','confidence'])
   assert r['latency_s']>0 and r['output_tokens']<=cfg['max_new_tokens']
   assert all(lookup[i]['split']=='source_train' for i in r['example_ids'])
   groups[(r['split'],r['mode'],r['variant'])].append(r)
  for split,n in [('validation',1),('test',2)]:
   expected={r['id'] for r in sampled(manifest,split,n)}
   for mode in ['zero','few']:assert {r['id'] for r in groups[(split,mode,'main')]}==expected
  for mode in ['zero','few']:
   m=evaluate(groups[('validation',mode,'main')]);w=cfg['selection']['weights']
   m.update(model=spec['key'],mode=mode)
   m['selection_score']=w['macro_f1']*m['macro_f1']+w['one_minus_ece']*(1-m['ece'] if m['ece'] is not None else 0)+w['valid_rate']*(1-m['invalid_rate'])+w['speed_score']/(1+m['latency_median_s']/5)
   validation_candidates.append(m)
  assert len(groups[('test','zero','ru')])==len(groups[('test','zero','paraphrase')])==13
  counts[spec['key']]=len(rows)
 sel=j('results/selection.json');app=[json.loads(s) for s in (ROOT/f'results/raw/{sel["model"]}_application.jsonl').read_text().splitlines()]
 eligible=[r for r in validation_candidates if r['invalid_rate']<=cfg['selection']['invalid_rate_max']]
 expected_selection=max(eligible or validation_candidates,key=lambda r:(r['selection_score'],-r['latency_median_s'],r['model'],r['mode']))
 assert sel['model']==expected_selection['model'] and sel['mode']==expected_selection['mode']
 assert math.isclose(sel['selection_score'],expected_selection['selection_score']) and sel['validity_gate_passed']==bool(eligible)
 assert len(app)==len({r['id'] for r in app})==455
 assert {r['id'] for r in app}=={r['id'] for r in manifest if r['split'] in ['target_adapt','new_slice','external']}
 assert all(r['mode']==sel['mode'] for r in app)
 for r in app:
  original=lookup[r['id']];assert r['label']==original['label'] and r['split']==original['split']
  parsed=parse_answer(r['raw']);assert all(r[k]==parsed[k] for k in ['valid','prediction','confidence'])
  assert r['revision']==pins[r['repo']] and r['latency_s']>0 and r['output_tokens']<=cfg['max_new_tokens']
 peft=[json.loads(s) for s in (ROOT/'results/raw/smol500_peft.jsonl').read_text().splitlines()]
 assert len(peft)==len({(r['id'],r['variant']) for r in peft})==221
 assert all(r['mode']=='zero' and r['revision']==pins[r['repo']] for r in peft)
 for r in peft:
  original=lookup[r['id']];assert r['label']==original['label'] and r['split']==original['split']
  parsed=parse_answer(r['raw']);assert all(r[k]==parsed[k] for k in ['valid','prediction','confidence'])
 for split,n in [('validation',1),('test',2)]:
  assert {r['id'] for r in peft if r['split']==split and r['variant']=='main'}=={r['id'] for r in sampled(manifest,split,n)}
 with (ROOT/'results/adaptation/metrics.csv').open(encoding='utf-8-sig') as f:adapt=list(csv.DictReader(f))
 keys={(int(r['seed']),r['alignment'],r['pseudo_labels'],r['split']) for r in adapt}
 expected={(s,a,p,t) for s in [20261006,20261007,20261008] for a in ['none','mmd','coral'] for p in ['False','True'] for t in ['test','new_slice']}
 assert len(adapt)==36 and keys==expected
 assert len(list((ROOT/'results/adaptation/heads').glob('*.safetensors')))==18
 observed=[json.loads(s) for s in (ROOT/'results/adaptation/predictions.jsonl').read_text().splitlines()]
 assert len(observed)==18*260
 for r in observed:
  original=lookup[r['id']];assert r['label']==original['label'] and r['split']==original['split']
  assert len(r['probabilities'])==65 and all(math.isfinite(v) for v in r['probabilities'])
  assert abs(sum(r['probabilities'])-1)<1e-5 and r['prediction']==max(range(65),key=lambda i:r['probabilities'][i])
 # Independently recompute adaptation tables and replay all retained heads on CPU.
 import numpy as np,torch
 from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef
 from safetensors.torch import load_file
 torch.set_num_threads(4)
 feature_cache=np.load(ROOT/'data/resnet18_features.npz');normalization=np.load(ROOT/'results/adaptation/source_standardization.npz')
 features=feature_cache['features'];source_indices=[i for i,r in enumerate(manifest) if r['split']=='source_train']
 assert np.allclose(normalization['mean'],features[source_indices].mean(0))
 assert np.allclose(normalization['std'],features[source_indices].std(0).clip(.1))
 inputs=torch.from_numpy(((features-normalization['mean'])/normalization['std']).astype('float32'))
 index_by_id={r['id']:i for i,r in enumerate(manifest)}
 for metric in adapt:
  seed=int(metric['seed']);method=metric['alignment'];pseudo=metric['pseudo_labels']=='True';split=metric['split']
  part=[r for r in observed if r['seed']==seed and r['alignment']==method and r['pseudo_labels']==pseudo and r['split']==split]
  expected_ids={r['id'] for r in manifest if r['split']==split}
  assert len(part)==len(expected_ids)==len({r['id'] for r in part}) and {r['id'] for r in part}==expected_ids
  truth=[r['label'] for r in part];pred=[r['prediction'] for r in part]
  independent={'accuracy':accuracy_score(truth,pred),'macro_f1':f1_score(truth,pred,labels=range(65),average='macro',zero_division=0),'mcc':matthews_corrcoef(truth,pred)}
  assert all(math.isclose(float(metric[k]),v,rel_tol=1e-10,abs_tol=1e-12) for k,v in independent.items())
  state=load_file(str(ROOT/f'results/adaptation/heads/{seed}_{method}_{int(pseudo)}.safetensors'))
  with torch.inference_mode():
   z=inputs[[index_by_id[r['id']] for r in part]]
   hidden=torch.relu(torch.nn.functional.linear(z,state['mapper.0.weight'],state['mapper.0.bias']))
   replay=torch.softmax(torch.nn.functional.linear(hidden,state['classifier.weight'],state['classifier.bias']),dim=-1).numpy()
  assert np.allclose(replay,np.array([r['probabilities'] for r in part]),rtol=1e-4,atol=1e-5)
  assert np.array_equal(replay.argmax(1),pred)
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/resnet18_features.npz',ROOT/'results/peft/adapter/adapter_model.safetensors']}
 report={'status':'passed','benchmark_counts':counts,'application_count':len(app),'peft_count':len(peft),'adaptation_runs':18,'protocol_hash_unchanged':True,'raw_parse_recomputed':True,'test_id_sets_identical':True,'few_shot_from_source_train_only':True,'observed_times_positive':True,'selection_recomputed_from_source_validation_only':True,'adaptation_predictions_finite':True,'adaptation_metrics_independently_recomputed':True,'retained_heads_cpu_replay_matches_observed_predictions':True,'derived_artifact_sha256':hashes}
 (ROOT/'logs/results_verification.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
