#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the lightweight RAG QA agent using /mnt/data/rag_eval_questions.csv

CSV expected columns:
question,expected_answer_contains

We mark correct if the agent answer (lowercased) contains the expected string (lowercased).
"""

import csv
from pathlib import Path
from rag_agent import build_default_agent

CSV_PATH = Path("/mnt/data/rag_eval_questions.csv")

def evaluate():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return
    agent = build_default_agent()
    total = 0
    correct = 0
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            q = row.get("question", "").strip()
            exp = row.get("expected_answer_contains", "").strip().lower()
            if not q:
                continue
            total += 1
            result = agent.answer(q)
            ans = (result.get("answer") or "").strip()
            ok = exp in ans.lower()
            correct += 1 if ok else 0
            rows.append({
                "question": q,
                "expected": exp,
                "answer": ans,
                "ok": ok,
                "sources": "; ".join(result.get("sources", []))
            })
    # print a small report
    print(f"Score: {correct}/{total} correct")
    for row in rows:
        flag = "✅" if row["ok"] else "❌"
        print(f"{flag} Q: {row['question']}\n   A: {row['answer']}\n   Expected contains: {row['expected']}\n   Sources: {row['sources']}\n")

if __name__ == "__main__":
    evaluate()
