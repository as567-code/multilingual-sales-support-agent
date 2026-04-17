"""Two-pass human-in-the-loop reviewer for the gold QA eval set.

Usage:
  python -m ingest.annotate review data/eval/gold_qa_en.jsonl --pass 1
  python -m ingest.annotate review data/eval/gold_qa_en.jsonl --pass 2
  python -m ingest.annotate status data/eval/gold_qa_en.jsonl

The reviewer walks through each unreviewed query, shows the LLM-generated
candidate gold_answer + relevant_faq_ids + difficulty, and records one of
four decisions: accept / edit / reject / skip. Decisions are stored in a
sidecar file `<source>.reviewed.jsonl` (same schema plus _review_status,
_review_pass, _reviewer, _review_ts). The source file is never mutated.

The PRD calls for solo two-pass annotation with a 24h gap between passes;
this tool tracks which pass each decision came from, enabling a deferred
Cohen's kappa calculation once the second pass is done.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

VALID_DECISIONS = {"accept", "edit", "reject", "skip"}


# ---- IO -----------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _reviewed_path(src: Path) -> Path:
    return src.with_suffix(".reviewed.jsonl")


def _reviewed_ids(path: Path, pass_num: int) -> set[str]:
    return {r["id"] for r in _load_jsonl(path) if r.get("_review_pass") == pass_num}


# ---- Review loop --------------------------------------------------------------


def _render(row: dict) -> None:
    print("\n" + "=" * 78)
    print(f"ID:          {row['id']}")
    print(f"Lang:        {row['lang']}")
    print(f"Category:    {row['category']}    Difficulty: {row['difficulty']}")
    print(f"Source FAQs: {', '.join(row['relevant_faq_ids']) or '(none — out-of-domain)'}")
    print("-" * 78)
    print(f"QUERY:       {row['query']}")
    print(f"GOLD ANSWER: {row['gold_answer']}")
    print("=" * 78)


def _prompt(prompt_msg: str) -> str:
    try:
        return input(prompt_msg).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nInterrupted.")
        sys.exit(130)


def review(src: Path, pass_num: int, reviewer: str, limit: int | None) -> None:
    rows = _load_jsonl(src)
    out = _reviewed_path(src)
    done = _reviewed_ids(out, pass_num)
    remaining = [r for r in rows if r["id"] not in done]
    print(f"Source: {src}")
    print(f"Pass:   {pass_num}   Reviewer: {reviewer}")
    print(f"Done:   {len(done)}   Remaining: {len(remaining)}")

    reviewed = 0
    for row in remaining:
        if limit is not None and reviewed >= limit:
            print(f"Reached --limit {limit}. Stopping.")
            break
        _render(row)
        while True:
            raw = _prompt("Decision [accept/edit/reject/skip/quit]: ")
            if raw in ("q", "quit", "exit"):
                print("Exiting. Progress saved.")
                return
            if raw in VALID_DECISIONS:
                decision = raw
                break
            print(f"  ? not valid — choose from {sorted(VALID_DECISIONS)}")

        edited_gold: str | None = None
        edited_query: str | None = None
        if decision == "edit":
            nq = _prompt("  new QUERY (blank = keep): ")
            if nq:
                edited_query = nq
            ng = _prompt("  new GOLD_ANSWER (blank = keep): ")
            if ng:
                edited_gold = ng

        record: dict[str, Any] = dict(row)
        if edited_query:
            record["query"] = edited_query
        if edited_gold:
            record["gold_answer"] = edited_gold
        record["_review_status"] = decision
        record["_review_pass"] = pass_num
        record["_reviewer"] = reviewer
        record["_review_ts"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        _append_jsonl(out, record)
        reviewed += 1

    print(f"\nWrote {reviewed} decisions to {out}")


# ---- Status -------------------------------------------------------------------


def status(src: Path) -> None:
    rows = _load_jsonl(src)
    rev = _load_jsonl(_reviewed_path(src))
    by_pass: dict[int, list[dict]] = {}
    for r in rev:
        by_pass.setdefault(r.get("_review_pass", 0), []).append(r)

    print(f"Source:   {src}   ({len(rows)} rows)")
    print(f"Reviewed: {_reviewed_path(src)}")
    for p, items in sorted(by_pass.items()):
        dec: dict[str, int] = {}
        for it in items:
            dec[it["_review_status"]] = dec.get(it["_review_status"], 0) + 1
        print(f"  pass {p}: {len(items)} decisions → {dec}")

    if {1, 2}.issubset(by_pass):
        p1 = {r["id"]: r["_review_status"] for r in by_pass[1]}
        p2 = {r["id"]: r["_review_status"] for r in by_pass[2]}
        shared = set(p1) & set(p2)
        if shared:
            agree = sum(1 for i in shared if p1[i] == p2[i])
            rate = agree / len(shared)
            print(f"  pairwise agreement over {len(shared)} shared IDs: "
                  f"{agree}/{len(shared)} = {rate:.3f}")
            print("  (Compute Cohen's kappa post-hoc; see data/eval/annotation_pipeline.md)")


# ---- Entrypoint ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("review")
    r.add_argument("src", type=Path)
    r.add_argument("--pass", dest="pass_num", type=int, choices=[1, 2], default=1)
    r.add_argument("--reviewer", default=os.environ.get("USER", "solo"))
    r.add_argument("--limit", type=int, default=None)

    s = sub.add_parser("status")
    s.add_argument("src", type=Path)

    args = parser.parse_args(argv)
    if args.cmd == "review":
        review(args.src, args.pass_num, args.reviewer, args.limit)
    elif args.cmd == "status":
        status(args.src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
