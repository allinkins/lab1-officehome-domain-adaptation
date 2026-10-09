from pathlib import Path
import os,json,csv,collections,math
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
a=json.loads((ROOT/'data/download_metadata.json').read_text());rows=json.loads((ROOT/'data/full_audit.json').read_text());m=json.loads((ROOT/'data/manifest.json').read_text())
out=ROOT/'results';fig=out/'figures';fig.mkdir(parents=True,exist_ok=True)
domains=['Art','Clipart','Product','Real_World'];classes=a['classes']
counts=np.array([[sum(r['domain']==d and r['label']==c for r in rows) for c in range(65)] for d in domains])
fig1,ax=plt.subplots(figsize=(7.2,3.5));bars=ax.bar(domains,counts.sum(1),color=['#5575a6','#8b77a9','#5b927c','#be9866']);ax.bar_label(bars,padding=3);ax.set_ylim(0,5000);ax.set_ylabel('Количество изображений');ax.set_title('Office-Home: фактически прочитанные изображения');ax.spines[['top','right']].set_visible(False);fig1.tight_layout();fig1.savefig(fig/'domain_counts.png',dpi=200);plt.close(fig1)
fig2,ax=plt.subplots(figsize=(10,3.2));im=ax.imshow(counts,aspect='auto',cmap='Blues');ax.set_yticks(range(4),domains);ax.set_xticks(range(0,65,5));ax.set_xlabel('Идентификатор класса');ax.set_title('Число изображений по доменам и классам');fig2.colorbar(im,ax=ax,label='Изображения');fig2.tight_layout();fig2.savefig(fig/'class_domain_heatmap.png',dpi=200);plt.close(fig2)
summary=[]
for di,d in enumerate(domains):
 part=[r for r in rows if r['domain']==d];vals=counts[di]
 summary.append({'domain':d,'n':len(part),'classes':len(set(r['label'] for r in part)),'class_min':int(vals.min()),'class_max':int(vals.max()),'imbalance_ratio':float(vals.max()/vals.min()),'median_width':float(np.median([r['width'] for r in part])),'median_height':float(np.median([r['height'] for r in part])),'min_width':min(r['width'] for r in part),'min_height':min(r['height'] for r in part),'max_width':max(r['width'] for r in part),'max_height':max(r['height'] for r in part)})
with (out/'data_statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
# Confirm absence of duplicate groups across partitions.
bygroup=collections.defaultdict(set)
for r in m:bygroup[r['group']].add(r['split'])
assert all(len(v)==1 for v in bygroup.values())
assert all(sum(r['split']==s and r['label']==c for r in m)==n for s,n in [('source_train',8),('validation',2),('target_adapt',3),('test',2),('new_slice',2),('external',2)] for c in range(65))
report={'statistics':summary,'no_group_leakage':True,'selected_missing_files':[r['path'] for r in m if not (ROOT/r['path']).exists()],'time_structure':'No chronological metadata supplied; no time-based split claimed.','missing_images':len(a['errors'])}
(out/'data_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
# Montage from preserved experiment images; not externally generated media.
fig3,axes=plt.subplots(4,4,figsize=(8,8))
for i,d in enumerate(domains):
 for j,c in enumerate([0,4,10,32]):
  r=next(r for r in m if r['domain']==d and r['label']==c);axes[i,j].imshow(Image.open(ROOT/r['path']));axes[i,j].axis('off')
  axes[i,j].set_title(f'{d}\n{classes[c]}',fontsize=9)
fig3.tight_layout();fig3.savefig(fig/'domain_examples.png',dpi=150);plt.close(fig3)
print(json.dumps(report,ensure_ascii=False,indent=2))
