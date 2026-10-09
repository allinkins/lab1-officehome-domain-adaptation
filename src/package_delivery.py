"""Hash and package the verified project; large runtime caches are excluded."""
from pathlib import Path
import argparse, datetime, hashlib, json, zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {'.git', '.cache', '.runtime', '.venv', '__pycache__', 'qa', 'logs', 'lab1_officehome', 'lab1_officehome 2', '.DS_Store'}
ARCHIVE = ROOT / 'Лабораторная_1_Office_Home_готовые_файлы.zip'
INVENTORY = ROOT / 'ARTIFACT_MANIFEST.json'


def included(path):
    rel = path.relative_to(ROOT)
    if str(rel) in {'src/prepare_delivery.py', 'configs/lab_requirements_source.txt',
                    'results/raw/smol256_smoke.json', 'results/analysis_complete.json'}:
        return False
    if rel.parts[:3] == ('results', 'peft', 'adapter') and path.name not in {
        'README.md', 'adapter_config.json', 'adapter_model.safetensors'
    }:
        return False
    return (not any(part in EXCLUDED_PARTS for part in rel.parts)
            and not path.name.startswith('~$')
            and path.suffix not in {'.zip', '.pyc', '.lock'}
            and path != INVENTORY and path != ROOT / 'WORKLOG.md')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify():
    inventory = json.loads(INVENTORY.read_text())
    for row in inventory['files']:
        path = ROOT / row['path']
        assert path.is_file(), row['path']
        assert path.stat().st_size == row['bytes'], row['path']
        assert sha256(path) == row['sha256'], row['path']
    print(f"Verified {len(inventory['files'])} delivered files against SHA256 inventory")


def package():
    verified = json.loads((ROOT / 'logs/results_verification.json').read_text())
    qa = json.loads((ROOT / 'logs/document_visual_review.json').read_text())
    cleanup = json.loads((ROOT / 'logs/weight_cleanup.json').read_text())
    assert verified['status'] == 'passed' and cleanup['status'] == 'completed'
    assert qa['status'] == 'passed' and qa['all_rendered_pages_reviewed']
    assert qa['docx_sha256'] == sha256(ROOT / 'Лабораторная_1_Office_Home.docx')
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and included(p))
    inventory = {
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'thesis_topic':'Сравнительный анализ методов выравнивания распределений для адаптации домена в задачах компьютерного зрения',
        'experiments':{'base_responses':2912, 'application_responses':455, 'lora_responses':221, 'adaptation_runs':18},
        'base_weights_removed':True,
        'repository_url':(ROOT / 'configs/repository_url.txt').read_text().strip() if (ROOT / 'configs/repository_url.txt').exists() else None,
        'files':[{'path':str(p.relative_to(ROOT)), 'bytes':p.stat().st_size, 'sha256':sha256(p)} for p in files],
    }
    INVENTORY.write_text(json.dumps(inventory, ensure_ascii=False, indent=2))
    verify()
    with zipfile.ZipFile(ARCHIVE, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files + [INVENTORY]:
            archive.write(path, str(Path('lab1_officehome') / path.relative_to(ROOT)))
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        expected = {str(Path('lab1_officehome') / p.relative_to(ROOT)) for p in files + [INVENTORY]}
        assert set(archive.namelist()) == expected
        for row in inventory['files']:
            assert hashlib.sha256(archive.read('lab1_officehome/' + row['path'])).hexdigest() == row['sha256']
    print(json.dumps({'archive':str(ARCHIVE), 'bytes':ARCHIVE.stat().st_size, 'sha256':sha256(ARCHIVE), 'archived_files':len(files)+1}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    verify() if args.verify else package()
