# TVSum Highlights (Stage 1)

Краткий репозиторий по промежуточному этапу проекта поиска хайлайтов в видео.

## Что внутри
- `tvsum_highlight_detection.ipynb` - основной ноутбук.
- `stage1_tvsum_report.md` - текст отчета для рендера в PDF.
- `images/` - графики и визуализации.

## Быстрый старт
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Рендер отчета в PDF
```bash
pandoc stage1_tvsum_report.md -o stage1_tvsum_report_from_md.pdf --pdf-engine=xelatex -V mainfont='PT Sans' -V sansfont='PT Sans' -V monofont='PT Mono'
```
