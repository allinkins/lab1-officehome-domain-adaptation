"""Read-only progress from observed files; does not declare unverified work complete."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'configs/experiment.json').read_text())
def count(path):
 if not path.exists():return 0
 with path.open() as f:return sum(1 for _ in f)
counts={m['key']:count(ROOT/f'results/raw/{m["key"]}_benchmark.jsonl') for m in cfg['models']}
status={'benchmark_counts':counts,'benchmark_responses':sum(counts.values()),'benchmark_expected':2912,'complete_models':sum(n==416 for n in counts.values()),'peft_responses':count(ROOT/'results/raw/smol500_peft.jsonl'),'application_responses':sum(count(p) for p in (ROOT/'results/raw').glob('*_application.jsonl')),'adaptation_runs_recorded':len(list((ROOT/'results/adaptation').glob('train_*.json'))),'core_analysis_complete':(ROOT/'results/analysis_complete.json').exists(),'artifacts_rendered':(ROOT/'logs/artifacts_prepared.json').exists()}
fail=ROOT/'logs/coordinator_failures.json'
status['failures']=json.loads(fail.read_text()) if fail.exists() else []
print(json.dumps(status,ensure_ascii=False))
