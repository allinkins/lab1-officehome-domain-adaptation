from pathlib import Path
import json
from metrics import evaluate,bootstrap_f1,paired_bootstrap_difference
from summarize import write_csv
ROOT=Path(__file__).resolve().parents[1]
def main():
 base=[json.loads(s) for s in (ROOT/'results/raw/smol500_benchmark.jsonl').read_text().splitlines()]
 adapted=[json.loads(s) for s in (ROOT/'results/raw/smol500_peft.jsonl').read_text().splitlines()];assert len(adapted)==221
 table=[]
 for mode,rs in [('zero',base),('few',base),('lora',adapted)]:
  part=[r for r in rs if r['split']=='test' and r['variant']=='main' and (mode=='lora' or r['mode']==mode)];assert len(part)==130
  m=evaluate(part);m.pop('calibration_bins');ci=bootstrap_f1(part);m.update(mode=mode,ci_low=ci[0],ci_high=ci[1]);table.append(m)
 a=[r for r in adapted if r['split']=='test' and r['variant']=='main'];b=[r for r in base if r['split']=='test' and r['variant']=='main' and r['mode']=='zero']
 ci=paired_bootstrap_difference(a,b)
 write_csv(ROOT/'results/peft/metrics.csv',table)
 (ROOT/'results/peft/paired_difference.json').write_text(json.dumps({'lora_minus_zero_macro_f1':evaluate(a)['macro_f1']-evaluate(b)['macro_f1'],'ci_low':ci[0],'ci_high':ci[1]},indent=2))
if __name__=='__main__':main()
