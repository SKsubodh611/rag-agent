#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def normalize_ws(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def sentences_from_markdown(md_text: str) -> List[str]:
    # Remove code blocks & inline code
    txt = re.sub(r'```.*?```', ' ', md_text, flags=re.S)
    txt = re.sub(r'`[^`]*`', ' ', txt)

    lines = [normalize_ws(line) for line in md_text.splitlines()]
    lines = [l for l in lines if l]  # drop empty

    sents: List[str] = []
    for line in lines:
        # Treat headings and bullets as standalone sentences
        if re.match(r'^\s*(#{1,6}\s+|[-*•]\s+|\d+\.\s+)', line):
            sents.append(re.sub(r'^\s*(#{1,6}\s+|[-*•]\s+|\d+\.\s+)', '', line).strip())
        else:
            parts = SENT_SPLIT_RE.split(line)
            parts = [normalize_ws(s) for s in parts if s and not s.isspace()]
            sents.extend(parts)
    return sents

def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r'[a-z0-9%\-/]+', text)

@dataclass
class Chunk:
    doc_id: int
    doc_path: str
    text: str
    tokens: List[str]
    tf: Dict[str, float]
    vec: Dict[str, float]
    sent_spans: List[Tuple[int, int]]

class TFIDFIndex:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.vocab: set[str] = set()

    def add_document(self, doc_id: int, path: str, sents: List[str], chunk_size: int = 1):
        for i in range(0, len(sents), chunk_size):
            block = sents[i:i+chunk_size]
            if not block:
                continue
            text = ' '.join(block)
            tokens = tokenize(text)
            if not tokens:
                continue
            tf: Dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0.0) + 1.0
            sent_spans = []
            offset = 0
            for s in block:
                s_clean = s.strip()
                if not s_clean:
                    continue
                start = text.find(s_clean, offset)
                if start == -1:
                    start = offset
                end = start + len(s_clean)
                sent_spans.append((start, end))
                offset = end
            self.chunks.append(Chunk(doc_id, path, text, tokens, tf, {}, sent_spans))
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1
                self.vocab.add(term)

    def finalize(self):
        N = len(self.chunks) if self.chunks else 1
        for term, df in self.df.items():
            self.idf[term] = math.log((1 + N) / (1 + df)) + 1.0
        for ch in self.chunks:
            vec: Dict[str, float] = {}
            length = 0.0
            max_tf = max(ch.tf.values()) if ch.tf else 1.0
            for term, freq in ch.tf.items():
                tf = 0.5 + 0.5 * (freq / max_tf)
                idf = self.idf.get(term, 0.0)
                w = tf * idf
                if w > 0:
                    vec[term] = w
                    length += w * w
            norm = math.sqrt(length) if length > 0 else 1.0
            for term in list(vec.keys()):
                vec[term] = vec[term] / norm
            ch.vec = vec

    def query(self, text: str, top_k: int = 5) -> List[Tuple[float, Chunk]]:
        q_tokens = tokenize(text)
        if not q_tokens:
            return []
        q_tf: Dict[str, float] = {}
        for t in q_tokens:
            q_tf[t] = q_tf.get(t, 0.0) + 1.0
        max_tf = max(q_tf.values()) if q_tf else 1.0
        q_vec: Dict[str, float] = {}
        q_len = 0.0
        for term, freq in q_tf.items():
            tf = 0.5 + 0.5 * (freq / max_tf)
            idf = self.idf.get(term, 0.0)
            w = tf * idf
            if w > 0:
                q_vec[term] = w
                q_len += w * w
        q_norm = math.sqrt(q_len) if q_len > 0 else 1.0
        for term in list(q_vec.keys()):
            q_vec[term] = q_vec[term] / q_norm

        def cosine(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
            if not vec_a or not vec_b:
                return 0.0
            if len(vec_a) > len(vec_b):
                vec_a, vec_b = vec_b, vec_a
            s = 0.0
            for term, wa in vec_a.items():
                wb = vec_b.get(term)
                if wb:
                    s += wa * wb
            return s

        scored = [(cosine(q_vec, ch.vec), ch) for ch in self.chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

class RAGQA:
    def _segment_on_separators(self, text: str):
        parts = re.split(r'\s+(?:[-–—•]|&bull;)\s+', text)
        return [p.strip() for p in parts if p.strip()]

    def _refine_to_best_segment(self, query: str, text: str) -> str:
        parts = self._segment_on_separators(text)
        if len(parts) <= 1:
            return text.strip()
        best, best_score = "", -1.0
        for seg in parts:
            sc = self._sentence_score(query, seg)
            if sc > best_score:
                best, best_score = seg, sc
        return best.strip()

    def _maybe_shorten_entity_answer(self, query: str, sent: str) -> str:
        q = set(tokenize(query))
        if 'which' in q:
            m = re.search(r'\bWidget\s+[A-Za-z0-9]+\b', sent)
            if m:
                return m.group(0)
        return sent.strip()

    def __init__(self, doc_paths: List[str], chunk_size: int = 1):
        self.doc_paths = [str(p) for p in doc_paths if Path(p).exists()]
        if not self.doc_paths:
            raise FileNotFoundError("No documents found at provided paths.")
        self.index = TFIDFIndex()
        for doc_id, p in enumerate(self.doc_paths):
            sents = sentences_from_markdown(read_text(p))
            self.index.add_document(doc_id, p, sents, chunk_size=chunk_size)
        self.index.finalize()

    @staticmethod
    def _sentence_score(query: str, sent: str) -> float:
        q_tokens = set(tokenize(query))
        s_tokens = set(tokenize(sent))
        if not s_tokens:
            return 0.0
        overlap = len(q_tokens & s_tokens) / (len(q_tokens) + 1e-9)
        has_number = bool(re.search(r'\b\d+(\.\d+)?\b', sent))
        has_percent = '%' in sent
        has_days = bool(re.search(r'\bday(s)?\b', sent.lower()))
        boost = 0.0
        boost += 0.15 if has_number else 0.0
        boost += 0.15 if has_percent else 0.0
        boost += 0.10 if has_days else 0.0
        return overlap + boost

    def _best_sentence(self, query: str, chunk) -> str:
        best_s, best_score = "", -1.0
        for (start, end) in chunk.sent_spans:
            s = chunk.text[start:end].strip()
            if not s:
                continue
            s = self._refine_to_best_segment(query, s)
            sc = self._sentence_score(query, s)
            if sc > best_score:
                best_s, best_score = s, sc
        return best_s.strip() if best_s else self._refine_to_best_segment(query, chunk.text.strip())

    def answer(self, query: str, top_k: int = 8) -> Dict[str, Any]:
        candidates = self.index.query(query, top_k=top_k)
        if not candidates:
            return {"answer": "Sorry, I couldn't find that in the docs.", "sources": []}

        # Diversify by document
        best_per_doc = {}
        for score, ch in candidates:
            current = best_per_doc.get(ch.doc_path)
            if (current is None) or (score > current[0]):
                best_per_doc[ch.doc_path] = (score, ch)
        diversified = sorted(best_per_doc.values(), key=lambda x: x[0], reverse=True)[:4]

        best = {"sent": "", "score": -1.0, "path": ""}
        for score, ch in diversified:
            sent = self._best_sentence(query, ch)
            s_score = self._sentence_score(query, sent) + score
            if s_score > best["score"]:
                best = {"sent": sent, "score": s_score, "path": ch.doc_path}

        source_paths = [os.path.basename(ch.doc_path) for _, ch in diversified]
        return {
            "answer": self._maybe_shorten_entity_answer(query, best["sent"]),
            "sources": source_paths
        }

def build_default_agent() -> "RAGQA":
    base = Path(__file__).resolve().parent
    docs = [
        "onboarding_guide.md",
        "pricing.md",
        "product_specs_widget.md",
        "refund_policy.md",
        "release_notes.md",
        "troubleshooting.md",
    ]
    defaults = [str(base / doc) for doc in docs]
    return RAGQA(defaults, chunk_size=1)

def main():
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", type=str, help="Ask a single question")
    parser.add_argument("--batch", type=str, help="TXT file with one question per line")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    agent = build_default_agent()

    if args.ask:
        out = agent.answer(args.ask)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.batch:
        with open(args.batch, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
        results = []
        for q in questions:
            results.append({"question": q, **agent.answer(q)})
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    parser.print_help()

if __name__ == "__main__":
    main()


