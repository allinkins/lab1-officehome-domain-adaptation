from pathlib import Path
import json,csv,collections,argparse
from metrics import evaluate,bootstrap_f1,paired_bootstrap_difference
ROOT=Path(__file__).resolve().parents[1]
def write_csv(path,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
 cfg=json.loads((ROOT/'configs/experiment.json').read_text());groups=collections.defaultdict(list)
 for p in sorted((ROOT/'results/raw').glob('*_benchmark.jsonl')):
  for line in p.read_text().splitlines():
   r=json.loads(line);groups[(r['model'],r['split'],r['mode'],r['variant'])].append(r)
 table=[];candidates=[]
 for (model,split,mode,variant),rs in groups.items():
  m=evaluate(rs);m.update(model=model,split=split,mode=mode,variant=variant)
  ci=bootstrap_f1(rs,cfg['bootstrap_repeats']) if split=='test' and variant=='main' else None
  m['macro_f1_ci_low']=ci[0] if ci else None;m['macro_f1_ci_high']=ci[1] if ci else None
  m.pop('calibration_bins');table.append(m)
  if split=='validation' and variant=='main':
   w=cfg['selection']['weights'];m['selection_score']=w['macro_f1']*m['macro_f1']+w['one_minus_ece']*(1-m['ece'] if m['ece'] is not None else 0)+w['valid_rate']*(1-m['invalid_rate'])+w['speed_score']/(1+m['latency_median_s']/5)
   candidates.append(m)
 write_csv(ROOT/'results/model_metrics.csv',table)
 expected={s['key'] for s in cfg['models']}
 completed={m for m in expected if all(len(groups[(m,'validation',mode,'main')])==65*cfg['validation_per_class'] and len(groups[(m,'test',mode,'main')])==65*cfg['test_per_class'] for mode in ['zero','few'])}
 if completed!=expected:
  print('Incomplete models:',sorted(expected-completed));return
 # Only validation values enter the selection rule.
 eligible=[r for r in candidates if r['invalid_rate']<=cfg['selection']['invalid_rate_max']]
 chosen=max(eligible or candidates,key=lambda r:(r['selection_score'],-r['latency_median_s'],r['model'],r['mode']))
 chosen=dict(chosen);chosen['validity_gate_passed']=bool(eligible)
 (ROOT/'results/selection.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2))
 diffs=[]
 for model in sorted(expected):
  a=groups[(model,'test','few','main')];b=groups[(model,'test','zero','main')];ci=paired_bootstrap_difference(a,b,cfg['bootstrap_repeats'])
  diffs.append({'model':model,'few_minus_zero_macro_f1':evaluate(a)['macro_f1']-evaluate(b)['macro_f1'],'ci_low':ci[0],'ci_high':ci[1]})
 write_csv(ROOT/'results/paired_mode_comparisons.csv',diffs)
 errors=[];byclass=[];sensitivity=[]
 for (model,split,mode,variant),rs in groups.items():
  if split!='test':continue
  if variant=='main':
   for cls in range(65):
    part=[r for r in rs if r['label']==cls];byclass.append({'model':model,'mode':mode,'label':cls,'n':len(part),'correct':sum(r['prediction']==cls for r in part),'invalid':sum(not r['valid'] for r in part)})
   errors.extend({'model':model,'mode':mode,'id':r['id'],'label':r['label'],'prediction':r['prediction'],'valid':r['valid'],'raw':r['raw'],'confidence':r['confidence'],'output_tokens':r['output_tokens'],'width':r['image_width'],'height':r['image_height']} for r in rs if r['prediction']!=r['label'])
  else:
   base={r['id']:r for r in groups[(model,split,mode,'main')]};matched=[r for r in rs if r['id'] in base]
   base_accuracy=sum(base[r['id']]['prediction']==r['label'] for r in matched)/len(matched)
   accuracy=sum(r['prediction']==r['label'] for r in matched)/len(matched)
   both_valid=[r for r in matched if r['valid'] and base[r['id']]['valid']]
   sensitivity.append({'model':model,'variant':variant,'n':len(matched),'prediction_flip_rate':sum(r['prediction']!=base[r['id']]['prediction'] for r in matched)/len(matched),'accuracy':accuracy,'base_accuracy_same_images':base_accuracy,'accuracy_difference_same_images':accuracy-base_accuracy,'invalid_rate':sum(not r['valid'] for r in matched)/len(matched),'base_invalid_rate_same_images':sum(not base[r['id']]['valid'] for r in matched)/len(matched),'raw_answer_change_rate':sum(r['raw']!=base[r['id']]['raw'] for r in matched)/len(matched),'both_valid_n':len(both_valid),'prediction_flip_rate_both_valid':sum(r['prediction']!=base[r['id']]['prediction'] for r in both_valid)/len(both_valid) if both_valid else None})
 write_csv(ROOT/'results/errors.csv',errors);write_csv(ROOT/'results/per_class.csv',byclass);write_csv(ROOT/'results/prompt_sensitivity.csv',sensitivity)
 print('SELECTION',json.dumps(chosen,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
