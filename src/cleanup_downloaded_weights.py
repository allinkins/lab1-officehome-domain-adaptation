"""Remove only this project's downloaded base weights after verified experiments."""
from pathlib import Path
import json, shutil, datetime

ROOT = Path(__file__).resolve().parents[1]
REPOS = {
    'HuggingFaceTB/SmolVLM-256M-Instruct',
    'HuggingFaceTB/SmolVLM-500M-Instruct',
    'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
    'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
    'llava-hf/llava-interleave-qwen-0.5b-hf',
    'Qwen/Qwen2-VL-2B-Instruct',
    'Qwen/Qwen2.5-VL-3B-Instruct',
}


def main():
    complete = json.loads((ROOT / 'results/analysis_complete.json').read_text())
    verified = json.loads((ROOT / 'logs/results_verification.json').read_text())
    assert complete['benchmark_responses'] == 2912
    assert complete['application_responses'] == 455
    assert verified['status'] == 'passed'
    assert verified['adaptation_runs'] == 18
    assert (ROOT / 'results/peft/adapter/adapter_model.safetensors').is_file()
    assert len(list((ROOT / 'results/adaptation/heads').glob('*.safetensors'))) == 18
    configured = {s['repo'] for s in json.loads((ROOT / 'configs/experiment.json').read_text())['models']}
    assert configured == REPOS
    targets = [ROOT / '.cache/huggingface/hub' / ('models--' + repo.replace('/', '--')) for repo in sorted(REPOS)]
    targets.append(ROOT / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
    records = []
    for path in targets:
        if not path.exists():
            continue
        assert not path.is_symlink()
        assert path.resolve().is_relative_to((ROOT / '.cache').resolve())
        files = [p for p in path.rglob('*') if p.is_file() and not p.is_symlink()] if path.is_dir() else [path]
        size = sum(p.stat().st_size for p in files)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        records.append({'path':str(path.relative_to(ROOT)), 'regular_file_bytes_removed':size})
    audit_path = ROOT / 'logs/weight_cleanup.json'
    previous = json.loads(audit_path.read_text()) if audit_path.exists() else {'removed':[]}
    all_records = previous['removed'] + records
    audit = {
        'status':'completed',
        'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Seven project-local downloaded base model caches and ResNet18 checkpoint only',
        'removed':all_records,
        'total_regular_file_bytes_removed':sum(r['regular_file_bytes_removed'] for r in all_records),
        'trained_adapter_and_heads_preserved':True,
        'data_features_raw_responses_and_environments_preserved':True,
    }
    audit_path.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
