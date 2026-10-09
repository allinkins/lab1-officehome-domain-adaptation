# LoRA-адаптер SmolVLM-500M для Office-Home

Базовая модель: [HuggingFaceTB/SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct), ревизия `a7da5b986cb59b408707209984f360a5f4ad7e47`.

Адаптер обучен на `Product/source_train` поднабора Office-Home. Параметры обучения и результаты сохранены в `results/peft/protocol.json` и `results/peft/metrics.csv`. Для применения требуется базовая модель указанной ревизии; её веса в этот каталог не входят. Условия использования базовой модели приведены в её карточке и `DATA_AND_MODEL_NOTICES.md`.
