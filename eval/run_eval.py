"""Evaluation harness for SemanticBookAI.

Retrieval (no LLM needed):
  - passage retrieval over the full texts: book Recall@1/@5, chapter Recall@5, evidence Recall@5, MRR@10
  - catalog search over book metadata: Recall@5, MRR@10
  each for dense-only, BM25-only and hybrid (RRF).
Answers (needs Ollama): citation validity, refusal behaviour, LLM-as-judge faithfulness and correctness,
latency and throughput per model.

Usage:
  python eval/run_eval.py                       # retrieval only
  python eval/run_eval.py --models gemma4:12b   # + answer metrics for one or more local models
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import llm  # noqa: E402
from app.chat import answer_stream  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.store import BOOKS, PASSAGES, search  # noqa: E402

EVAL_DIR = ROOT / "eval"
MODES = ["dense", "sparse", "hybrid"]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'").replace("ß", "ss")).lower()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def gold_locations(qa: list[dict]) -> dict[str, dict]:
    """Chapters and passages of the gold book that contain the evidence phrase."""
    passages = load_jsonl(get_settings().processed_dir / "passages.jsonl")
    by_book = defaultdict(list)
    for p in passages:
        by_book[p["book_id"]].append(p)
    out = {}
    for q in qa:
        if not q.get("answerable", True):
            continue
        ev = norm(q["evidence"])
        hits = [p for p in by_book[q["book_id"]] if ev in norm(p["text"])]
        if not hits:
            raise SystemExit(f"evidence for {q['id']!r} not found in {q['book_id']}: {q['evidence']!r}")
        out[q["id"]] = {
            "chapters": {p["chapter_idx"] for p in hits},
            "passages": {(p["chapter_idx"], p["passage_idx"]) for p in hits},
        }
    return out


def rr(ranked: list[bool]) -> float:
    return next((1 / (i + 1) for i, ok in enumerate(ranked) if ok), 0.0)


def eval_passage_retrieval(qa: list[dict], gold: dict) -> dict:
    results = {}
    for mode in MODES:
        rows = []
        for q in qa:
            if not q.get("answerable", True):
                continue
            t0 = time.perf_counter()
            hits = search(PASSAGES, q["question"], mode=mode, limit=10)
            ms = (time.perf_counter() - t0) * 1000
            book_ok = [h.payload["book_id"] == q["book_id"] for h in hits]
            ch_ok = [ok and h.payload["chapter_idx"] in gold[q["id"]]["chapters"] for ok, h in zip(book_ok, hits)]
            ev_ok = [ok and (h.payload["chapter_idx"], h.payload["passage_idx"]) in gold[q["id"]]["passages"]
                     for ok, h in zip(book_ok, hits)]
            rows.append({"id": q["id"], "book@1": book_ok[0] if hits else False, "book@5": any(book_ok[:5]),
                         "chapter@5": any(ch_ok[:5]), "evidence@5": any(ev_ok[:5]), "mrr": rr(book_ok), "ms": ms})
        results[mode] = {
            "book_recall@1": mean(r["book@1"] for r in rows),
            "book_recall@5": mean(r["book@5"] for r in rows),
            "chapter_recall@5": mean(r["chapter@5"] for r in rows),
            "evidence_recall@5": mean(r["evidence@5"] for r in rows),
            "mrr@10": mean(r["mrr"] for r in rows),
            "p50_ms": statistics.median(r["ms"] for r in rows),
            "per_question": rows,
        }
    return results


def eval_catalog(catalog: list[dict]) -> dict:
    results = {}
    for mode in MODES:
        rows = []
        for c in catalog:
            t0 = time.perf_counter()
            hits = search(BOOKS, c["query"], mode=mode, limit=10)
            ms = (time.perf_counter() - t0) * 1000
            ok = [h.payload["book_id"] in c["book_ids"] for h in hits]
            rows.append({"id": c["id"], "recall@5": any(ok[:5]), "mrr": rr(ok), "ms": ms,
                         "top3": [h.payload["title"] for h in hits[:3]]})
        results[mode] = {
            "recall@5": mean(r["recall@5"] for r in rows),
            "mrr@10": mean(r["mrr"] for r in rows),
            "p50_ms": statistics.median(r["ms"] for r in rows),
            "per_query": rows,
        }
    return results


JUDGE_PROMPT = """You are grading an answer produced by a retrieval-augmented assistant.

Sources given to the assistant:
{sources}

Question: {question}
Reference answer: {reference}
Assistant answer: {answer}

Grade two things:
- faithfulness: 1 if every claim in the assistant answer is supported by the sources, 0.5 if mostly, 0 if it contains unsupported claims.
- correctness: 1 if the assistant answer agrees with the reference answer, 0.5 if partially, 0 if not.
Reply with JSON only: {{"faithfulness": <0|0.5|1>, "correctness": <0|0.5|1>, "reason": "<one short sentence>"}}"""


async def judge(question: str, reference: str, answer: str, sources: list[dict], model: str) -> dict:
    src = "\n\n".join(f"[{s['n']}] {s['text']}" for s in sources)
    out = await llm.complete([{"role": "user", "content": JUDGE_PROMPT.format(
        sources=src, question=question, reference=reference, answer=answer)}], model=model)
    m = re.search(r"\{.*\}", out, re.S)
    try:
        return json.loads(m.group(0)) if m else {"faithfulness": 0, "correctness": 0, "reason": "unparseable"}
    except json.JSONDecodeError:
        return {"faithfulness": 0, "correctness": 0, "reason": "unparseable"}


async def run_question(q: dict, model: str) -> dict:
    sources, done, refusal = [], None, None
    async for ev in answer_stream(q["question"], model=model):
        if ev["event"] == "sources":
            sources = ev["sources"]
        elif ev["event"] == "done":
            done = ev
        elif ev["event"] == "refusal":
            refusal = ev
    return {"sources": sources, "done": done, "refusal": refusal}


async def eval_answers(qa: list[dict], models: list[str], judge_model: str) -> dict:
    # Pass 1: every candidate model answers every question (one model load per candidate).
    generated: dict[str, list[tuple[dict, dict, list[dict]]]] = {}
    for model in models:
        await llm.complete([{"role": "user", "content": "Say OK."}], model=model)  # warm-up, not timed
        generated[model] = []
        for q in qa:
            r = await run_question(q, model)
            row = {"id": q["id"], "answerable": q.get("answerable", True)}
            if r["refusal"]:
                row.update(refused=True, reason=r["refusal"]["reason"], answer=r["refusal"]["text"])
            elif r["done"]:
                d = r["done"]
                cited_books = {s["book_id"] for s in r["sources"] if s["n"] in d["citations"]}
                row.update(refused=False, answer=d["text"], citations=d["citations"],
                           invalid_citations=d["invalid_citations"], uncited=d.get("uncited", False),
                           cites_gold_book=q.get("book_id") in cited_books, ttft_s=d.get("ttft_s"),
                           tokens_per_s=d.get("tokens_per_s"), latency_s=d["latency_s"])
            generated[model].append((q, row, r["sources"]))
            print(f"    {model:14} {q['id']:16} {'REFUSED' if row.get('refused') else 'answered'}", flush=True)

    # Pass 2: one judge model grades all answers (one model load).
    for model in models:
        for q, row, sources in generated[model]:
            if row["answerable"] and not row.get("refused") and "answer" in row:
                row.update(await judge(q["question"], q["reference"], row["answer"], sources, judge_model))
        print(f"    judged {model}", flush=True)

    results = {}
    for model in models:
        rows = [row for _, row, _ in generated[model]]
        answerable = [r for r in rows if r["answerable"]]
        answered = [r for r in answerable if not r["refused"]]
        gen = [r for r in rows if not r.get("refused")]
        n_cit = sum(len(r["citations"]) + len(r["invalid_citations"]) for r in gen)
        n_bad = sum(len(r["invalid_citations"]) for r in gen)
        results[model] = {
            "citation_validity": round(1 - n_bad / n_cit, 3) if n_cit else 1.0,
            "answers_with_citation": mean(not r["uncited"] for r in gen) if gen else 0,
            "cites_gold_book": mean(r["cites_gold_book"] for r in answered) if answered else 0,
            "unanswerable_refused": f"{sum(r['refused'] for r in rows if not r['answerable'])}/"
                                    f"{sum(not r['answerable'] for r in rows)}",
            "false_refusals": f"{sum(r['refused'] for r in answerable)}/{len(answerable)}",
            "faithfulness": mean(r.get("faithfulness", 0) for r in answered) if answered else 0,
            "correctness": mean(r.get("correctness", 0) for r in answered) if answered else 0,
            "p50_ttft_s": median([r["ttft_s"] for r in gen if r.get("ttft_s")]),
            "p50_tokens_per_s": median([r["tokens_per_s"] for r in gen if r.get("tokens_per_s")]),
            "per_question": rows,
        }
    return results


def mean(values) -> float:
    values = [float(v) for v in values]
    return round(sum(values) / len(values), 3) if values else 0.0


def median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def fmt(x) -> str:
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def write_report(res: dict, path: Path) -> None:
    s = get_settings()
    lines = ["# Evaluation results", "",
             f"Corpus: {res['corpus']['books']} books, {res['corpus']['passages']} passages · "
             f"dense model `{s.embed_model}` (Ollama) · sparse `{s.sparse_model}` · RRF k={s.rrf_k}", ""]
    lines += ["## Passage retrieval (chat)", "",
              f"{res['n_retrieval']} answerable questions (EN + DE, incl. one cross-lingual), no book filter, top-10.", "",
              "| Mode | Book R@1 | Book R@5 | Chapter R@5 | Evidence R@5 | MRR@10 | p50 latency |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for mode in MODES:
        r = res["passages"][mode]
        lines.append(f"| {mode} | {fmt(r['book_recall@1'])} | {fmt(r['book_recall@5'])} | {fmt(r['chapter_recall@5'])} | "
                     f"{fmt(r['evidence_recall@5'])} | {fmt(r['mrr@10'])} | {r['p50_ms']:.0f} ms |")
    lines += ["", "## Catalog search", "", f"{res['n_catalog']} descriptive queries (EN + DE).", "",
              "| Mode | Recall@5 | MRR@10 | p50 latency |", "| --- | --- | --- | --- |"]
    for mode in MODES:
        r = res["catalog"][mode]
        lines.append(f"| {mode} | {fmt(r['recall@5'])} | {fmt(r['mrr@10'])} | {r['p50_ms']:.0f} ms |")
    if res.get("answers"):
        lines += ["", "## Grounded answers (local LLMs via Ollama)", "",
                  f"{res['n_retrieval']} answerable + {res['n_unanswerable']} unanswerable questions. "
                  f"Judge: `{res['judge_model']}` (local LLM-as-judge).", "",
                  "| Model | Citation validity | Answers with citation | Cites gold book | Faithfulness | Correctness | "
                  "Unanswerable refused | False refusals | p50 TTFT | p50 tok/s |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for model, r in res["answers"].items():
            lines.append(f"| `{model}` | {fmt(r['citation_validity'])} | {fmt(r['answers_with_citation'])} | "
                         f"{fmt(r['cites_gold_book'])} | {fmt(r['faithfulness'])} | {fmt(r['correctness'])} | "
                         f"{r['unanswerable_refused']} | {r['false_refusals']} | {r['p50_ttft_s']} s | {r['p50_tokens_per_s']} |")
    lines += ["", f"Generated by `eval/run_eval.py` on {time.strftime('%Y-%m-%d')}."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="Ollama models for answer evaluation")
    ap.add_argument("--judge", default=None, help="judge model (default: largest of --models)")
    args = ap.parse_args()

    qa = load_jsonl(EVAL_DIR / "qa.jsonl")
    catalog = load_jsonl(EVAL_DIR / "catalog.jsonl")
    answerable = [q for q in qa if q.get("answerable", True)]
    gold = gold_locations(qa)
    s = get_settings()
    corpus = {"books": len(load_jsonl(s.processed_dir / "books.jsonl")),
              "passages": sum(1 for _ in (s.processed_dir / "passages.jsonl").open(encoding="utf-8"))}

    print("passage retrieval …", flush=True)
    res = {"corpus": corpus, "n_retrieval": len(answerable), "n_unanswerable": len(qa) - len(answerable),
           "n_catalog": len(catalog), "passages": eval_passage_retrieval(qa, gold)}
    print("catalog search …", flush=True)
    res["catalog"] = eval_catalog(catalog)
    if args.models:
        res["judge_model"] = args.judge or args.models[-1]
        print("answers …", flush=True)
        res["answers"] = asyncio.run(eval_answers(qa, args.models, res["judge_model"]))

    (EVAL_DIR / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=list), encoding="utf-8")
    write_report(res, EVAL_DIR / "results.md")
    print((EVAL_DIR / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
