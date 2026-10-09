"""Produce an executed results notebook and a separate portable rerun notebook."""
from pathlib import Path
import json,io,contextlib,os,base64

try:
 import nbformat as nbf
except ModuleNotFoundError:
 # Minimal standards-compliant notebook writer. This avoids making artifact
 # creation depend on an optional package after the ML experiments finish.
 class Obj(dict):
  __getattr__=dict.__getitem__
  __setattr__=dict.__setitem__
 class V4:
  @staticmethod
  def new_notebook():return Obj(nbformat=4,nbformat_minor=5,metadata=Obj(),cells=[])
  @staticmethod
  def new_markdown_cell(source):return Obj(cell_type='markdown',metadata=Obj(),source=source)
  @staticmethod
  def new_code_cell(source):return Obj(cell_type='code',execution_count=None,metadata=Obj(),outputs=[],source=source)
  @staticmethod
  def new_output(output_type,**kwargs):return Obj(output_type=output_type,**kwargs)
 class MiniNB:
  v4=V4()
  @staticmethod
  def validate(notebook):
   assert notebook['nbformat']==4 and isinstance(notebook['cells'],list)
   assert all(c['cell_type'] in {'markdown','code'} for c in notebook['cells'])
  @staticmethod
  def write(notebook,path):Path(path).write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
 nbf=MiniNB()
ROOT=Path(__file__).resolve().parents[1]
def main():
 nb=nbf.v4.new_notebook();nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
 cells=[];ns={};execution=0
 def md(s):cells.append(nbf.v4.new_markdown_cell(s))
 def code(s):
  nonlocal execution
  c=nbf.v4.new_code_cell(s);buf=io.StringIO()
  with contextlib.redirect_stdout(buf):exec(compile(s,'<notebook>','exec'),ns)
  execution+=1;c.execution_count=execution;c.outputs=[nbf.v4.new_output('stream',name='stdout',text=buf.getvalue())] if buf.getvalue() else [];cells.append(c)
 old=os.getcwd();os.chdir(ROOT)
 md('# Лабораторная работа 1 на Office-Home\n\nТема ВКР: **Сравнительный анализ методов выравнивания распределений для адаптации домена в задачах компьютерного зрения**.\n\nАнализ результатов сравнения мультимодальных моделей и методов адаптации домена. Первичные ответы моделей сохранены в `results/raw`; код экспериментов находится в `src/benchmark.py`, `src/adaptation.py` и `src/train_lora.py`.')
 code("from pathlib import Path\nimport json, csv, hashlib\nROOT = Path.cwd()\nassert (ROOT / 'configs/experiment.json').exists(), 'Откройте notebook из корня проекта'\ndef read_json(path): return json.loads((ROOT / path).read_text())\ndef read_csv(path):\n    with (ROOT / path).open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))\nprint('Наблюдений в корпусе:', read_json('data/download_metadata.json')['rows_decoded'])\nprint('Выбранная модель:', read_json('results/selection.json')['model'])")
 md('## Данные и отсутствие утечки\nПолный корпус прошёл потоковый аудит; разбиение зафиксировано до сравнения. Дубликаты объединялись до разбиения. Все показатели ниже получены из сохранённого аудита.')
 code("audit = read_json('data/download_metadata.json')\nprint('Изображений в корпусе:', audit['rows_decoded'])\nprint('Домены:', audit['domain_counts'])\nprint('Размеры частей:', audit['split_counts'])\nprint('Точные пары:', audit['exact_duplicate_pairs'])\nprint('Кандидаты похожих пар:', audit['near_duplicate_candidate_pairs'])\nprint('Ошибок чтения:', len(audit['errors']))\nmanifest = read_json('data/manifest.json')\ngroups = {}\nfor r in manifest:\n    assert groups.setdefault(r['group'], r['split']) == r['split']\nprint('Пересечений групп между частями нет')")
 md('## Единый протокол\n40 новых токенов, жадная генерация, строгий JSON; zero-shot и три фиксированных source-примера. Invalid считается ошибкой, без исправлений. Калибровка — по текстовой уверенности среди валидных ответов. Полный текст инструкции доступен в `configs/prompt_main.txt`.\n\nЧисленная ошибка Qwen float16/MPS подтверждена на source_train и исправлена штатной точностью bfloat16. 156 технически некорректных ответов исключены и сохранены в `logs/invalid_runs`; изменение зарегистрировано в `configs/runtime_amendment.json`. Остальные модели используют float16. Различие точности ограничивает интерпретацию latency.')
 code("cfg = read_json('configs/experiment.json')\nfor model in cfg['models']: print(model['key'], model['repo'])\nprint('SHA256 протокола:', hashlib.sha256((ROOT/'configs/experiment.json').read_bytes()).hexdigest())\nprint('Правило выбора:', cfg['selection'])")
 md('## Фактические тестовые метрики\nДоверительный интервал — 2000 стратифицированных bootstrap повторов. Малый тест ограничивает интерпретацию.')
 code("metrics = read_csv('results/model_metrics.csv')\nprint('model | mode | Macro-F1 | 95% CI | invalid rate | p50 seconds')\nfor r in metrics:\n    if r['split']=='test' and r['variant']=='main':\n        print(r['model'], r['mode'], round(float(r['macro_f1']),4), (r['macro_f1_ci_low'],r['macro_f1_ci_high']), r['invalid_rate'],r['latency_median_s'], sep=' | ')")
 md('## Выбор и новые срезы\nМодель выбирается по source validation, а не по максимуму test.')
 code("selection = read_json('results/selection.json')\nprint('Выбрана:', selection['model'], selection['mode'])\nprint('Порог валидности пройден:', selection['validity_gate_passed'])\ntest = [r for r in metrics if r['split']=='test' and r['variant']=='main']\nchosen = next(r for r in test if r['model']==selection['model'] and r['mode']==selection['mode'])\nbest_test = max(test,key=lambda r:float(r['macro_f1']))\nprint('Выбранный режим на test:',chosen['macro_f1'])\nprint('Максимум на test:',best_test['model'],best_test['mode'],best_test['macro_f1'])\nprint('Test не изменяет выбор, сделанный по source validation')\nfor r in read_csv('results/application_metrics.csv'): print(r['domain'],r['n'],r['macro_f1'],r['invalid_rate'])")
 md('## Вклад псевдометок в MMD и CORAL\nТри seed, общий backbone, общий классификатор и source validation. Разброс между seed — SD, не доверительный интервал.')
 code("print(read_json('results/adaptation/pseudo_label_audit.json'))\nfor r in read_csv('results/adaptation/summary.csv'):\n    print(r['split'],r['alignment'],r['pseudo_labels'],r['macro_f1_mean'],r['macro_f1_sd'])\nprint('Парный вклад псевдометок:')\nfor r in read_csv('results/adaptation/paired_effects.csv'): print(r)\nprint('Парный вклад выравнивания относительно none:')\nfor r in read_csv('results/adaptation/alignment_effects.csv'): print(r)")
 md('## LoRA как отдельное расширение\nОбучается только языковой адаптер на source_train. Этот опыт не подменяет сравнение семи базовых моделей и не участвует задним числом в их выборе.')
 code("for r in read_csv('results/peft/metrics.csv'): print(r)\nprint(read_json('results/peft/paired_difference.json'))")
 md('## Примеры ошибок\nИсходные ответы сохраняются без редактирования. Причины ошибок нельзя устанавливать только по корреляции с фоном или длиной.')
 code("for r in read_json('results/ten_errors.json'):\n    print('ID',r['id'],'true',r['true_class'],'predicted',r['predicted_class'],'model',r['model'])\n    print(r['raw'])")
 md('## Графики')
 for name in ['model_comparison.png','cost_calibration.png','adaptation_comparison.png','ten_errors.png']:
  encoded=base64.b64encode((ROOT/'results/figures'/name).read_bytes()).decode();c=nbf.v4.new_markdown_cell(f'![{name}](attachment:{name})');c.attachments={name:{'image/png':encoded}};cells.append(c)
 md('## Повторение вычислений\nДля пересчёта анализа используйте `python src/summarize.py`, `python src/summarize_peft.py`, `python src/analyze_results.py`, `python src/verify_results.py`. Для повторного вывода моделей и обучения см. README. Скачанные веса удаляются после экспериментов, поэтому полный повтор требует их загрузки. Новый протокол должен записывать результаты в отдельный каталог.')
 nb.cells=cells;nbf.validate(nb);nbf.write(nb,ROOT/'Лабораторная_1_анализ.ipynb');os.chdir(old)
 cloud=nbf.v4.new_notebook();cloud.metadata.kernelspec=nb.metadata.kernelspec
 cloud.cells=[
  nbf.v4.new_markdown_cell('# Повторный запуск\n\nНужны Python 3.12, доступ к интернету и GPU с нативной поддержкой bfloat16 и достаточной памятью. Поместите каталог проекта в `/content/lab1_officehome` или `/kaggle/working/lab1_officehome`. Первая ячейка создаёт отдельную рабочую копию кода и конфигураций; изображения загружаются из зафиксированной версии Office-Home. При возобновлении расчёта используйте ранее созданный рабочий каталог и переходите к следующей ячейке.'),
  nbf.v4.new_code_cell("from pathlib import Path\nimport os, subprocess, sys, shutil\nSOURCE = (Path('/content') if Path('/content').exists() else Path('/kaggle/working')) / 'lab1_officehome'\nassert (SOURCE/'configs/experiment.json').exists()\nassert sys.version_info[:2] == (3,12), 'Для зафиксированной среды нужен Python 3.12'\nROOT = SOURCE.with_name('lab1_officehome_reproduction')\nassert not ROOT.exists(), 'Каталог уже существует: для возобновления задайте ROOT и пропустите копирование'\nROOT.mkdir()\nfor name in ['src','configs','literature']:\n    shutil.copytree(SOURCE/name,ROOT/name)\n(ROOT/'data').mkdir()\nshutil.copy2(SOURCE/'data/download_metadata.json',ROOT/'data/download_metadata.json')\nfor name in ['requirements.txt','README.md','REQUIREMENTS.md','report_background.md','DATA_AND_MODEL_NOTICES.md','Паспорт_данных.md']:\n    shutil.copy2(SOURCE/name,ROOT/name)\n(ROOT/'logs').mkdir()\n(ROOT/'results').mkdir()\nos.chdir(ROOT)\nsubprocess.check_call([sys.executable,'-m','pip','install','-r','requirements.txt'])"),
  nbf.v4.new_code_cell("import torch\nprint(torch.__version__,torch.cuda.is_available())\nassert torch.cuda.is_available(), 'Выберите GPU runtime'\nprint(torch.cuda.get_device_name(0))\nassert torch.cuda.is_bf16_supported(including_emulation=False), 'Для текущего протокола Qwen нужен GPU с нативным bfloat16; другой режим точности требует отдельной проверки'\nprint('GPU bytes:',torch.cuda.get_device_properties(0).total_memory)\n# Меньшая память может вызвать OOM: модель few-shot обрабатывает четыре изображения."),
  nbf.v4.new_code_cell("import json\nsubprocess.check_call([sys.executable,'src/download_data.py'])\nsubprocess.check_call([sys.executable,'src/verify.py'])\nsubprocess.check_call([sys.executable,'src/analyze_data.py'])\nwith Path('configs/environment_freeze.txt').open('w') as out:\n    subprocess.check_call([sys.executable,'-m','pip','freeze'],stdout=out)"),
  nbf.v4.new_code_cell("cfg=json.loads(Path('configs/experiment.json').read_text())\nfor spec in cfg['models']:\n    subprocess.check_call([sys.executable,'src/download_models.py','--model',spec['key']])\n    subprocess.check_call([sys.executable,'src/benchmark.py','--model',spec['key']])\nsubprocess.check_call([sys.executable,'src/summarize.py'])\nselected=json.loads(Path('results/selection.json').read_text())\nsubprocess.check_call([sys.executable,'src/benchmark.py','--model',selected['model'],'--application'])\nsubprocess.check_call([sys.executable,'src/adaptation.py'])"),
  nbf.v4.new_code_cell("subprocess.check_call([sys.executable,'src/train_lora.py','--probe'])\nsubprocess.check_call([sys.executable,'src/train_lora.py'])\nsubprocess.check_call([sys.executable,'src/benchmark.py','--model','smol500','--adapter','results/peft/adapter'])\nfor script in ['summarize_peft.py','analyze_results.py','verify_results.py']:\n    subprocess.check_call([sys.executable,'src/'+script])"),
  nbf.v4.new_markdown_cell('Все новые ответы и журналы находятся в `lab1_officehome_reproduction`; исходный комплект не изменяется. Локальные показатели latency нельзя считать облачными. В случае OOM следует выбрать устройство с большей памятью; изменение разрешения или few-shot состава создаёт другой протокол, который нельзя объединять с исходными результатами.')
 ]
 nbf.validate(cloud);nbf.write(cloud,ROOT/'Повторный запуск.ipynb');print('Notebooks built')
if __name__=='__main__':main()
