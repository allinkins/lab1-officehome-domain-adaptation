"""Derive tables, diagnostics and scientific plots exclusively from observed logs."""
import json,csv,collections,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix
from metrics import evaluate,bootstrap_f1
from summarize import write_csv
ROOT=Path(__file__).resolve().parents[1]
def read_jsonl(path):return [json.loads(x) for x in path.read_text().splitlines()]
def main():
 cfg=json.loads((ROOT/'configs/experiment.json').read_text());sel=json.loads((ROOT/'results/selection.json').read_text())
 classes=json.loads((ROOT/'data/classes.json').read_text());manifest={r['id']:r for r in json.loads((ROOT/'data/manifest.json').read_text())}
 rows=sum([read_jsonl(ROOT/f'results/raw/{m["key"]}_benchmark.jsonl') for m in cfg['models']],[])
 assert len(rows)==7*416
 app=read_jsonl(ROOT/f'results/raw/{sel["model"]}_application.jsonl');assert len(app)==455
 mainrows=[r for r in rows if r['split']=='test' and r['variant']=='main']
 winner=[r for r in mainrows if r['model']==sel['model'] and r['mode']==sel['mode']]
 out=ROOT/'results';fig=out/'figures';fig.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
 with (out/'model_metrics.csv').open(encoding='utf-8-sig') as f:metrics=list(csv.DictReader(f))
 test=[r for r in metrics if r['split']=='test' and r['variant']=='main'];names=[m['key'] for m in cfg['models']]
 schema=[]
 for key in names:
  for mode in ['zero','few']:
   part=[r for r in mainrows if r['model']==key and r['mode']==mode];counts=collections.Counter()
   for r in part:
    if r['valid']:continue
    if r['raw'].lstrip().startswith('```'):category='markdown_fence'
    else:
     try:category='non_object_json' if not isinstance(json.loads(r['raw']),dict) else 'schema_type_or_range'
     except (ValueError,TypeError):category='malformed_or_extra_text'
    counts[category]+=1
   schema.append({'model':key,'mode':mode,'n':len(part),'invalid':sum(counts.values()),**{k:counts[k] for k in ['markdown_fence','non_object_json','schema_type_or_range','malformed_or_extra_text']},'invalid_at_token_limit':sum(not r['valid'] and r['output_tokens']==cfg['max_new_tokens'] for r in part)})
 write_csv(out/'schema_errors.csv',schema)
 memory=[]
 for key in names:
  for mode in ['zero','few']:
   part=[r for r in mainrows if r['model']==key and r['mode']==mode]
   snapshots=[r['mps_allocated_end_gb'] for r in part if r.get('mps_allocated_end_gb') is not None]
   memory.append({'model':key,'mode':mode,'peak_process_rss_gb':max(r['process_peak_rss_gb'] for r in part),'max_observed_mps_driver_end_gb':max(snapshots) if snapshots else None,'note':'Maximum of end-of-request snapshots, including driver/allocator caches; not a GPU peak measurement.'})
 write_csv(out/'memory_observations.csv',memory)
 f,ax=plt.subplots(figsize=(10,5));x=np.arange(7)
 for j,mode in enumerate(['zero','few']):
  subset=[next(r for r in test if r['model']==n and r['mode']==mode) for n in names];v=np.array([float(r['macro_f1']) for r in subset]);lo=np.array([float(r['macro_f1_ci_low']) for r in subset]);hi=np.array([float(r['macro_f1_ci_high']) for r in subset])
  ax.bar(x+(j-.5)*.35,v,.35,label=mode+'-shot');ax.errorbar(x+(j-.5)*.35,(hi+lo)/2,yerr=(hi-lo)/2,fmt='none',color='black',capsize=3)
 ax.set(xticks=x,xticklabels=names,ylabel='Macro-F1',ylim=(0,1),title='Office-Home Clipart · 130 images · 95% bootstrap intervals');ax.tick_params(axis='x',rotation=25);ax.legend();f.tight_layout();f.savefig(fig/'model_comparison.png',dpi=180);plt.close(f)
 f,axes=plt.subplots(1,2,figsize=(10,5));colors=dict(zip(names,plt.get_cmap('tab10').colors[:7]))
 for mode,marker in [('zero','o'),('few','s')]:
  part=[r for r in test if r['mode']==mode]
  axes[0].scatter([float(r['latency_median_s']) for r in part],[float(r['macro_f1']) for r in part],c=[colors[r['model']] for r in part],marker=marker)
 axes[0].set(xlabel='Median latency, seconds',ylabel='Macro-F1')
 axes[0].legend(handles=[Line2D([],[],color='gray',marker=m,linestyle='',label=s) for s,m in [('zero','o'),('few','s')]],fontsize=11)
 ev=evaluate(winner);bins=ev['calibration_bins'];axes[1].plot([0,1],[0,1],'--',color='gray')
 if bins:axes[1].scatter([b['confidence'] for b in bins],[b['accuracy'] for b in bins],s=[max(20,b['n']*4) for b in bins])
 axes[1].set(xlabel='Self-reported confidence',ylabel='Observed accuracy',xlim=(0,1),ylim=(0,1),title=f'Valid answers only: n={ev["valid_n"]}')
 f.legend(handles=[Line2D([],[],color=colors[n],marker='o',linestyle='',label=n) for n in names],loc='lower center',ncol=4,fontsize=11)
 f.tight_layout(rect=(0,.14,1,1));f.savefig(fig/'cost_calibration.png',dpi=180);plt.close(f)
 appmetrics=[]
 for split in ['target_adapt','new_slice','external']:
  part=[r for r in app if r['split']==split];m=evaluate(part);m.pop('calibration_bins');ci=bootstrap_f1(part,cfg['bootstrap_repeats']);m.update(split=split,domain=part[0]['domain'],macro_f1_ci_low=ci[0],macro_f1_ci_high=ci[1]);appmetrics.append(m)
 write_csv(out/'application_metrics.csv',appmetrics)
 grouped=[]
 for dimension,func in [('output_length',lambda r:'up to 15 tokens' if r['output_tokens']<=15 else ('16 to 30 tokens' if r['output_tokens']<=30 else '31 to 40 tokens')),('input_length',lambda r:'up to 1000 tokens' if r['input_tokens']<=1000 else 'over 1000 tokens'),('image_area',lambda r:'up to 250000 pixels' if r['image_width']*r['image_height']<=250000 else 'over 250000 pixels'),('language',lambda r:r['instruction_language']),('domain',lambda r:r['domain'])]:
  groups=collections.defaultdict(list)
  for r in rows+app:groups[(r['model'],r['mode'],r['split'],dimension,func(r))].append(r)
  for key,part in groups.items():
   model,mode,split,dim,val=key
   grouped.append({'model':model,'mode':mode,'split':split,'dimension':dim,'value':val,'n':len(part),'accuracy':np.mean([r['label']==r['prediction'] for r in part]),'invalid_rate':np.mean([not r['valid'] for r in part]),'note':'Descriptive strata; variants may repeat the same image; no causal interpretation'})
 write_csv(out/'error_strata.csv',grouped)
 # Select ten diverse observed errors, prefer the chosen model, then other models.
 candidates=[r for r in winner if r['label']!=r['prediction']]+[r for r in mainrows if r['label']!=r['prediction']]
 examples=[];seen=set();seen_classes=set()
 for r in candidates:
  if r['id'] in seen or r['label'] in seen_classes:continue
  seen.add(r['id']);seen_classes.add(r['label']);q=dict(r);q['true_class']=classes[r['label']];q['predicted_class']=classes[r['prediction']] if r['valid'] else 'INVALID';q['error_type']='schema / invalid JSON' if not r['valid'] else ('confident wrong class' if r['confidence']>=.7 else 'wrong class with low confidence');q['image_path']=manifest[r['id']]['path'];examples.append(q)
  if len(examples)==10:break
 assert len(examples)==10, 'Ten distinct error classes must exist in the observed run'
 (out/'ten_errors.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
 valid_errors=[r for r in winner if r['valid'] and r['prediction']!=r['label']]
 pairs=collections.Counter((r['label'],r['prediction']) for r in valid_errors)
 error_summary={'model':sel['model'],'mode':sel['mode'],'valid_errors':len(valid_errors),'confident_wrong_answers':sum(r['confidence']>=cfg['pseudo_label_confidence_threshold'] for r in valid_errors),'confidence_threshold':cfg['pseudo_label_confidence_threshold'],'top_confusions':[{'true_class':classes[a],'predicted_class':classes[b],'n':n} for (a,b),n in pairs.most_common(5)],'note':'At most two observations per true class; confusion frequencies are exploratory.'}
 (out/'selected_error_summary.json').write_text(json.dumps(error_summary,ensure_ascii=False,indent=2))
 from PIL import Image
 f,axes=plt.subplots(2,5,figsize=(10,6))
 for ax,r in zip(axes.flat,examples):
  ax.imshow(Image.open(ROOT/r['image_path']));ax.axis('off');ax.set_title(f'#{r["id"]} {r["true_class"]}\n→ {r["predicted_class"]}',fontsize=12)
 f.tight_layout();f.savefig(fig/'ten_errors.png',dpi=180);plt.close(f)
 matrix=confusion_matrix([r['label'] for r in winner],[r['prediction'] for r in winner],labels=list(range(65))+[-1])[:65]
 np.savetxt(out/'confusion_matrix.csv',matrix,delimiter=',',fmt='%d',header=','.join(classes+['INVALID']),comments='')
 f,ax=plt.subplots(figsize=(13,12));im=ax.imshow(matrix,cmap='Blues');ax.set_xticks(range(66),classes+['INVALID'],rotation=90,fontsize=5);ax.set_yticks(range(65),classes,fontsize=5);ax.set(xlabel='Predicted class',ylabel='True class',title=f'{sel["model"]} {sel["mode"]} · Clipart test');f.colorbar(im,ax=ax,shrink=.6);f.tight_layout();f.savefig(fig/'confusion_matrix.png',dpi=180);plt.close(f)
 with (out/'adaptation/metrics.csv').open(encoding='utf-8-sig') as f:adapt=list(csv.DictReader(f))
 aggregation=[]
 for split in ['test','new_slice']:
  for method in ['none','mmd','coral']:
   for pseudo in ['False','True']:
    part=[r for r in adapt if r['split']==split and r['alignment']==method and r['pseudo_labels']==pseudo];assert len(part)==3
    rec={'split':split,'alignment':method,'pseudo_labels':pseudo,'seeds':3}
    for metric in ['accuracy','macro_f1','mcc']:
     val=np.array([float(r[metric]) for r in part]);rec[metric+'_mean']=float(val.mean());rec[metric+'_sd']=float(val.std(ddof=1))
    aggregation.append(rec)
 write_csv(out/'adaptation/summary.csv',aggregation)
 effects=[]
 for split in ['test','new_slice']:
  for method in ['none','mmd','coral']:
   diffs=[]
   for seed in ['20261006','20261007','20261008']:
    a=next(r for r in adapt if r['split']==split and r['alignment']==method and r['seed']==seed and r['pseudo_labels']=='True');b=next(r for r in adapt if r['split']==split and r['alignment']==method and r['seed']==seed and r['pseudo_labels']=='False');diffs.append(float(a['macro_f1'])-float(b['macro_f1']))
   effects.append({'split':split,'alignment':method,'pseudo_effect_mean':np.mean(diffs),'pseudo_effect_sd':np.std(diffs,ddof=1),'paired_seed_differences':json.dumps(diffs)})
 write_csv(out/'adaptation/paired_effects.csv',effects)
 alignment_effects=[]
 for split in ['test','new_slice']:
  for pseudo in ['False','True']:
   for method in ['mmd','coral']:
    differences=[]
    for seed in ['20261006','20261007','20261008']:
     method_row=next(r for r in adapt if r['split']==split and r['alignment']==method and r['seed']==seed and r['pseudo_labels']==pseudo)
     baseline=next(r for r in adapt if r['split']==split and r['alignment']=='none' and r['seed']==seed and r['pseudo_labels']==pseudo)
     differences.append(float(method_row['macro_f1'])-float(baseline['macro_f1']))
    alignment_effects.append({'split':split,'alignment':method,'pseudo_labels':pseudo,'alignment_effect_mean':np.mean(differences),'alignment_effect_sd':np.std(differences,ddof=1),'paired_seed_differences':json.dumps(differences)})
 write_csv(out/'adaptation/alignment_effects.csv',alignment_effects)
 f,ax=plt.subplots(figsize=(8,4));x=np.arange(3)
 for j,pseudo in enumerate(['False','True']):
  part=[r for r in aggregation if r['split']=='test' and r['pseudo_labels']==pseudo]
  ax.bar(x+(j-.5)*.35,[r['macro_f1_mean'] for r in part],.35,yerr=[r['macro_f1_sd'] for r in part],capsize=4,label='With pseudo-labels' if j else 'Without pseudo-labels')
 ax.set(xticks=x,xticklabels=['Source only','MMD','CORAL'],ylabel='Macro-F1 · mean ± SD',ylim=(0,1),title='Product → Clipart · three training seeds');ax.legend();f.tight_layout();f.savefig(fig/'adaptation_comparison.png',dpi=180);plt.close(f)
 (out/'analysis_complete.json').write_text(json.dumps({'benchmark_responses':len(rows),'application_responses':len(app),'adaptation_runs':18,'error_examples':len(examples),'all_numbers_from_real_logs':True},indent=2))
 print('Analysis artifacts complete')
if __name__=='__main__':main()
