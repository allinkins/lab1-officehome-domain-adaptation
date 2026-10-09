"""Synchronize verified citation metadata, the research matrix and reference exports."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIT = ROOT / "literature"
ACCESS_DATE = "09.10.2026"


def initials(name: str) -> tuple[str, str]:
    words = name.split()
    return words[-1], " ".join(f"{word[0].upper()}." for word in words[:-1])


def author_heading(name: str) -> str:
    surname, given = initials(name)
    return f"{surname}, {given}"


def author_statement(name: str) -> str:
    surname, given = initials(name)
    return f"{given} {surname}"


def gost_reference(meta: dict, title: str) -> str:
    authors = meta["authors"]
    credit = ", ".join(author_statement(name) for name in authors)
    if meta["more_authors"]:
        credit += " [et al.]"
    # ГОСТ Р 7.0.80-2023: four or more authors are described under the title.
    lead = title if meta["more_authors"] else f"{author_heading(authors[0])} {title}"
    body = f"{lead} / {credit} // {meta['container']}. — {meta['year']}."
    if meta.get("volume"):
        body += " — Vol. " + meta["volume"] + "."
    if meta.get("number"):
        body += " — No. " + meta["number"] + "."
    if meta.get("pages"):
        body += " — P. " + meta["pages"] + "."
    if meta.get("eprint"):
        body += " — arXiv:" + meta["eprint"] + "."
    if meta.get("doi"):
        body += " — DOI: " + meta["doi"] + "."
    return body + " — URL: " + meta["url"] + f" (дата обращения: {ACCESS_DATE}). — Текст : электронный."


def bib_reference(meta: dict, title: str) -> str:
    values = [
        ("title", title),
        ("author", " and ".join(meta["authors"] + (["others"] if meta["more_authors"] else []))),
        ("year", str(meta["year"])),
        ("journal" if meta["type"] == "article" else "booktitle" if meta["type"] == "inproceedings" else "howpublished", meta["container"]),
    ]
    for key in ("volume", "number", "pages", "doi", "eprint"):
        if meta.get(key):
            val = meta[key].replace("–", "--") if key == "pages" else meta[key]
            values.append((key, val))
    if meta.get("eprint"):
        values.append(("archivePrefix", "arXiv"))
    values.append(("url", meta["url"]))
    return "@" + meta["type"] + "{" + meta["id"] + ",\n" + "\n".join(f"  {key} = {{{value}}}," for key, value in values) + "\n}"


def main() -> None:
    metadata = json.loads((LIT / "citation_metadata.json").read_text())
    matrix = json.loads((LIT / "matrix.json").read_text())
    assert len(metadata) == len(matrix) == 15
    assert [m["id"] for m in metadata] == [r["id"] for r in matrix]
    by_id = {r["id"]: r for r in matrix}
    for meta in metadata:
        row = by_id[meta["id"]]
        row["url"] = meta["url"]
        row["year"] = str(meta["year"])
    by_id["R09"]["venue_status"] = "COLM 2025; доступная версия arXiv"
    by_id["R09"]["venue_rank_note"] = "Принята на COLM 2025; подтверждено официальным списком принятых работ"
    by_id["R09"]["venue_rank_source"] = "https://colmweb.org/2025/AcceptedPapers.html"
    by_id["R02"]["data"] = "Office-31 и другие визуальные наборы, рассматриваемые в статье DAN"
    by_id["R05"]["data"] = "Визуальные бенчмарки адаптации, включая Office-Home и Office-31"
    by_id["R08"]["data"] = "Визуальные инструкции и ScienceQA; оценка следования инструкциям"
    by_id["R09"]["metrics"] = "Показатели визуальных benchmark, скорость и память модели"
    by_id["R13"]["metrics"] = "ECE, диаграммы надёжности, калибровка вероятностей"
    (LIT / "matrix.json").write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n")
    with (LIT / "matrix.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(matrix)
    (LIT / "references.bib").write_text("\n\n".join(bib_reference(m, by_id[m["id"]]["title"]) for m in metadata) + "\n")
    (LIT / "references_gost.txt").write_text("\n\n".join(f"[{i}] {gost_reference(m, by_id[m['id']]['title'])}" for i, m in enumerate(metadata, 1)) + "\n")
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (2100, 570), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 39)
    font_small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 31)
    boxes = [(60, 120, 610, 355), (775, 120, 1325, 355), (1490, 120, 2040, 355)]
    labels = [
        "Зарегистрировано\n16 уникальных работ",
        "Исключена 1 работа\nBrown et al. (2020)\nпо меньшей близости к задаче",
        "В аналитический обзор\nвключено 15 работ",
    ]
    for box, label in zip(boxes, labels):
        draw.rectangle(box, fill="white", outline="black", width=3)
        draw.multiline_text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), label, font=font,
                            fill="black", anchor="mm", align="center", spacing=12)
    for x in (655, 1370):
        draw.line((x, 237, x + 65, 237), fill="black", width=5)
        draw.polygon([(x + 65, 223), (x + 65, 251), (x + 83, 237)], fill="black")
    draw.text((1050, 465), "Общий объём поисковой выдачи и этап полного скрининга не фиксировались.",
              font=font_small, anchor="mm", fill="black")
    canvas.save(ROOT / "results/figures/literature_flow.png")


if __name__ == "__main__":
    main()
