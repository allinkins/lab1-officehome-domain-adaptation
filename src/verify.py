"""Integrity and metric checks. Synthetic fixtures here never enter experiment results."""
from pathlib import Path
import json,hashlib,sys
from benchmark import parse_answer
from metrics import evaluate
ROOT=Path(__file__).resolve().parents[1]
def main():
 rows=json.loads((ROOT/'data/manifest.json').read_text());groups={};ids=set()
 for r in rows:
  assert r['id'] not in ids;ids.add(r['id'])
  assert groups.setdefault(r['group'],r['split'])==r['split']
  assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['file_sha256']
 assert parse_answer('{"label_id":4,"confidence":0.3}')['valid']
 assert not parse_answer('{"label_id":"4","confidence":0.3}')['valid']
 assert not parse_answer('{"label_id":4,"confidence":1.3}')['valid']
 assert not parse_answer('```json\n{"label_id":4,"confidence":0.3}\n```')['valid']
 fixtures=[{'label':0,'prediction':0,'valid':True,'confidence':.3,'latency_s':1},{'label':1,'prediction':-1,'valid':False,'confidence':None,'latency_s':1}]
 m=evaluate(fixtures,nclasses=2)
 assert abs(m['accuracy']-.5)<1e-9 and abs(m['macro_f1']-.5)<1e-9
 assert abs(m['ece']-.7)<1e-9 and abs(m['binary_brier']-.49)<1e-9
 assert sum(b['n'] for b in m['calibration_bins'])==m['valid_n']
 status={'manifest_rows':len(rows),'file_hashes_valid':True,'split_group_leakage':False,'metric_unit_checks_passed':True,'fixtures_used_for_model_results':False}
 (ROOT/'logs/verification.json').write_text(json.dumps(status,indent=2));print(status)
if __name__=='__main__':main()
