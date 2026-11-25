"""Tier-2 Controlled Evaluation Harness for Paper 1 (Spoke 1 — ucai-verify v0.2.0).

Compares five auditing regimes across all 120 items of UCAI-CitationAudit-Bench-v1:
  1. ucai-verify Stage 1 (Deterministic Citation Cascade + PDF Numeric/Polarity Matcher, $0)
  2. Hybrid Cascade — Local NLI (Stage 1 + DeBERTa-v3 Cross-Encoder on survivors, $0)
  3. Hybrid Cascade — Scoped LLM (Stage 1 + Grounded Gemini Stage-2 on survivors)
  4. Pure Self-Critique (Zero-shot LLM on full 120 items)
  5. Pure LLM-as-a-Judge (Structured Reviewer Prompt on full 120 items)

Includes per-item checkpointing, exponential rate-limit backoff, 95% percentile
bootstrap confidence intervals (1,000 resamples), and exact McNemar's test p-values.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .bench_builder import BenchItem, VERIFIED_ANCHORS
from .citation_auditor import CitationAuditor
from .hybrid_auditor import HybridCascadeAuditor
from .numeric_auditor import NumericAuditor


def compute_binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "n": len(y_true),
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_bootstrap_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Computes 95% percentile bootstrap confidence intervals for key binary metrics."""
    n = len(y_true)
    if n == 0:
        return {}
    rng = random.Random(seed)
    keys = ("accuracy", "precision", "recall", "f1", "fpr")
    samples: Dict[str, List[float]] = {k: [] for k in keys}
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        m = compute_binary_metrics(yt, yp)
        for k in keys:
            samples[k].append(float(m[k]))
    ci: Dict[str, List[float]] = {}
    lo_idx = int(0.025 * n_resamples)
    hi_idx = min(n_resamples - 1, int(0.975 * n_resamples))
    for k in keys:
        arr = sorted(samples[k])
        ci[f"{k}_ci95"] = [round(arr[lo_idx], 4), round(arr[hi_idx], 4)]
    return ci


def mcnemar_exact_pvalue(
    y_true: Sequence[int], y_pred_a: Sequence[int], y_pred_b: Sequence[int]
) -> Dict[str, Any]:
    """Exact two-sided binomial McNemar test comparing correctness of Method A vs Method B."""
    a_right_b_wrong = 0
    a_wrong_b_right = 0
    for yt, pa, pb in zip(y_true, y_pred_a, y_pred_b):
        ca = pa == yt
        cb = pb == yt
        if ca and not cb:
            a_right_b_wrong += 1
        elif cb and not ca:
            a_wrong_b_right += 1
    n_disc = a_right_b_wrong + a_wrong_b_right
    if n_disc == 0:
        return {"b": 0, "c": 0, "discordant": 0, "p_value": 1.0}
    k = min(a_right_b_wrong, a_wrong_b_right)
    tail = sum(math.comb(n_disc, i) for i in range(k + 1)) / (2**n_disc)
    p_val = min(1.0, 2.0 * tail)
    return {
        "a_right_b_wrong": a_right_b_wrong,
        "a_wrong_b_right": a_wrong_b_right,
        "discordant": n_disc,
        "p_value": round(p_val, 6),
    }


def _per_class_breakdown(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    per_class: Dict[str, Dict[str, Any]] = {}
    for cls in sorted({r["label_class"] for r in records}):
        sub = [r for r in records if r["label_class"] == cls]
        correct = sum(1 for r in sub if r["is_corrupted"] == r["pred_corrupted"])
        per_class[cls] = {
            "n": len(sub),
            "correct": correct,
            "acc": round(correct / len(sub), 4) if sub else 0.0,
        }
    return per_class


class ExperimentEvaluator:
    """Runs deterministic, hybrid, and full-120 LLM baseline comparisons."""

    def __init__(self, pdf_dir: Optional[str | Path] = None) -> None:
        self.cit_auditor = CitationAuditor()
        self.num_auditor = NumericAuditor()
        self.hybrid_auditor = HybridCascadeAuditor()

        # Index real PDFs if available, else register grounded anchor source text
        indexed_ids = set()
        if pdf_dir and Path(pdf_dir).exists():
            for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
                for anc in VERIFIED_ANCHORS:
                    if anc["arxiv_id"] in pdf_path.name:
                        try:
                            self.num_auditor.index_pdf(anc["doc_id"], pdf_path)
                            indexed_ids.add(anc["doc_id"])
                        except Exception:
                            pass
        # Ensure all 12 anchor documents have indexed text fallback for offline/unit-test reproducibility
        for anc in VERIFIED_ANCHORS:
            if anc["doc_id"] not in self.num_auditor.corpus_pages:
                self.num_auditor.index_text_pages(anc["doc_id"], [(1, anc["source_text"])])
        self.pdf_loaded_count = len(indexed_ids)

    def run_ucai_verify(
        self, items: List[BenchItem], use_live_apis: bool = True
    ) -> Dict[str, Any]:
        """Evaluates Stage 1 ucai-verify across all 120 benchmark items."""
        t_start = time.perf_counter()
        y_true: List[int] = []
        y_pred: List[int] = []
        records: List[Dict[str, Any]] = []
        cit_cache: Dict[tuple, Any] = {}

        for it in items:
            if use_live_apis:
                cache_key = (it.title, it.author, it.year, it.doi, it.arxiv_id)
                if cache_key not in cit_cache:
                    cit_cache[cache_key] = self.cit_auditor.verify_reference(
                        title=it.title,
                        author=it.author,
                        year=it.year,
                        doi=it.doi,
                        arxiv_id=it.arxiv_id,
                    )
                cit_res = cit_cache[cache_key]
                cit_flagged = cit_res.status != "VERIFIED_EXACT"
                cit_status = cit_res.status
            else:
                anc_map = {a["arxiv_id"]: a for a in VERIFIED_ANCHORS}
                rec = anc_map.get(it.arxiv_id)
                if rec and rec["title"] == it.title and rec["doi"] == it.doi:
                    cit_flagged, cit_status = False, "VERIFIED_EXACT"
                elif rec:
                    cit_flagged, cit_status = True, "IDENTIFIER_HIJACKED"
                else:
                    cit_flagged, cit_status = True, "SUSPECTED_FAKE"

            num_res = self.num_auditor.audit_claim(it.claim_text, it.cited_doc_id)
            num_flagged = num_res.status not in ("NUMERIC_VERIFIED", "DOC_NOT_INDEXED")

            pred = 1 if (cit_flagged or num_flagged) else 0
            y_true.append(it.is_corrupted)
            y_pred.append(pred)
            records.append(
                {
                    "item_id": it.item_id,
                    "label_class": it.label_class,
                    "is_corrupted": it.is_corrupted,
                    "pred_corrupted": pred,
                    "citation_status": cit_status,
                    "numeric_status": num_res.status,
                    "matched_doc_id": num_res.matched_doc_id,
                    "matched_page": num_res.matched_page,
                    "evidence_snippet": num_res.evidence_snippet,
                    "evidence_context": num_res.best_evidence_sentence or num_res.evidence_snippet,
                    "notes": num_res.notes,
                }
            )

        elapsed = round(time.perf_counter() - t_start, 3)
        metrics = compute_binary_metrics(y_true, y_pred)
        metrics.update(compute_bootstrap_ci(y_true, y_pred))
        metrics["total_latency_sec"] = elapsed
        metrics["sec_per_item"] = round(elapsed / len(items), 4) if items else 0.0
        metrics["llm_api_calls"] = 0

        return {
            "method": "ucai-verify Stage 1 (Deterministic Cascade + PDF Numeric)",
            "live_apis": use_live_apis,
            "pdfs_indexed": self.pdf_loaded_count,
            "metrics": metrics,
            "per_class": _per_class_breakdown(records),
            "records": records,
        }

    def run_hybrid_local_nli(
        self,
        items: List[BenchItem],
        ucai_res: Dict[str, Any],
        contradiction_threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """Runs Real Stage 2 Open-Weight NLI Hybrid Cascade ($0 API cost)."""
        s1_records = ucai_res["records"]
        s1_lat = float(ucai_res["metrics"]["total_latency_sec"])
        raw = self.hybrid_auditor.run_local_nli_cascade(
            items=items,
            stage1_records=s1_records,
            stage1_latency_sec=s1_lat,
            contradiction_threshold=contradiction_threshold,
        )
        y_true = [r["is_corrupted"] for r in raw["records"]]
        y_pred = [r["pred_corrupted"] for r in raw["records"]]
        metrics = compute_binary_metrics(y_true, y_pred)
        metrics.update(compute_bootstrap_ci(y_true, y_pred))
        metrics["stage1_latency_sec"] = raw["stage1_latency_sec"]
        metrics["stage2_latency_sec"] = raw["stage2_latency_sec"]
        metrics["total_latency_sec"] = raw["total_latency_sec"]
        metrics["sec_per_item"] = round(raw["total_latency_sec"] / len(items), 4) if items else 0.0
        metrics["stage2_nli_evals"] = raw["stage2_calls"]
        metrics["stage2_call_reduction_pct"] = raw["stage2_call_reduction_pct"]
        metrics["llm_api_calls"] = 0
        return {
            "method": raw["method"],
            "metrics": metrics,
            "per_class": _per_class_breakdown(raw["records"]),
            "records": raw["records"],
        }

    def run_hybrid_scoped_gemini(
        self,
        items: List[BenchItem],
        ucai_res: Dict[str, Any],
        api_key: str,
        model_name: str = "gemini-flash-lite-latest",
        checkpoint_path: Optional[str | Path] = None,
        sleep_sec: float = 4.2,
    ) -> Dict[str, Any]:
        """Runs Real Stage 2 Scoped Gemini Hybrid Cascade on Stage-1 survivors."""
        from google import genai

        client = genai.Client(api_key=api_key)
        model_name = self._pick_working_gemini_model(client, model_name)
        s1_records = ucai_res["records"]
        s1_lat = float(ucai_res["metrics"]["total_latency_sec"])
        ckpt = Path(checkpoint_path) if checkpoint_path else None
        raw = self.hybrid_auditor.run_scoped_gemini_cascade(
            items=items,
            stage1_records=s1_records,
            stage1_latency_sec=s1_lat,
            client=client,
            model_name=model_name,
            checkpoint_path=ckpt,
            sleep_sec=sleep_sec,
        )
        y_true = [r["is_corrupted"] for r in raw["records"]]
        y_pred = [r["pred_corrupted"] for r in raw["records"]]
        metrics = compute_binary_metrics(y_true, y_pred)
        metrics.update(compute_bootstrap_ci(y_true, y_pred))
        metrics["stage1_latency_sec"] = raw["stage1_latency_sec"]
        metrics["stage2_latency_sec"] = raw["stage2_latency_sec"]
        metrics["total_latency_sec"] = raw["total_latency_sec"]
        metrics["sec_per_item"] = round(raw["total_latency_sec"] / len(items), 4) if items else 0.0
        metrics["llm_api_calls"] = raw["stage2_calls"]
        metrics["clean_ok"] = raw["stage2_clean_ok"]
        metrics["error_count"] = raw["stage2_error_count"]
        metrics["stage2_call_reduction_pct"] = raw["stage2_call_reduction_pct"]
        return {
            "method": raw["method"],
            "metrics": metrics,
            "per_class": _per_class_breakdown(raw["records"]),
            "records": raw["records"],
        }

    @staticmethod
    def _pick_working_gemini_model(client: Any, preferred: str) -> str:
        """Selects the first working Gemini model via a 1-token probe, preferring stable flash-lite."""
        candidates = [
            "gemini-flash-lite-latest",
            preferred,
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.5-flash",
        ]
        skip_unstable = {"gemini-3.8-flash", "gemini-flash-latest"}
        try:
            for m in client.models.list():
                name = getattr(m, "name", "") or ""
                short = name.replace("models/", "")
                if "gemini" in short and "embedding" not in short and "tts" not in short and "image" not in short:
                    if short not in candidates and short not in skip_unstable:
                        candidates.append(short)
        except Exception:
            pass

        for cand in candidates:
            if cand in skip_unstable:
                continue
            try:
                client.models.generate_content(model=cand, contents="Reply OK")
                print(f"  [Gemini Probe] Using model: {cand}", flush=True)
                return cand
            except Exception as exc:
                print(f"  [Gemini Probe] {cand} unavailable: {str(exc)[:80]}", flush=True)
        return preferred

    @staticmethod
    def _stratified_subset(items: List[BenchItem], max_items: Optional[int]) -> List[BenchItem]:
        if not max_items or max_items >= len(items):
            return items
        buckets: Dict[str, List[BenchItem]] = {}
        for it in items:
            buckets.setdefault(it.label_class, []).append(it)
        selected: List[BenchItem] = []
        idx = 0
        keys = list(buckets.keys())
        while len(selected) < max_items:
            added = False
            for k in keys:
                if idx < len(buckets[k]) and len(selected) < max_items:
                    selected.append(buckets[k][idx])
                    added = True
            if not added:
                break
            idx += 1
        return selected

    def run_gemini_baseline(
        self,
        items: List[BenchItem],
        api_key: str,
        mode: str = "self_critique",
        model_name: str = "gemini-flash-lite-latest",
        max_items: Optional[int] = 120,
        sleep_sec: float = 4.2,
        checkpoint_path: Optional[str | Path] = None,
        seed_clean_paths: Optional[Sequence[str | Path]] = None,
    ) -> Dict[str, Any]:
        """Runs Baseline A (Self-Critique) or Baseline B (LLM-as-a-Judge) across up to 120 items."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        model_name = self._pick_working_gemini_model(client, model_name)
        subset = self._stratified_subset(items, max_items)

        # Load verified clean checkpoints (never reuse any ERR: record!)
        clean_cache: Dict[str, Dict[str, Any]] = {}
        paths_to_load: List[Path] = []
        if seed_clean_paths:
            paths_to_load.extend(Path(p) for p in seed_clean_paths)
        if checkpoint_path:
            paths_to_load.append(Path(checkpoint_path))

        for p in paths_to_load:
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    for rec in data.get("records", []):
                        raw_rep = str(rec.get("raw_reply", ""))
                        if raw_rep and not raw_rep.startswith("ERR:"):
                            clean_cache[rec["item_id"]] = rec
                except Exception:
                    pass

        if clean_cache:
            print(
                f"  [{mode}] Loaded {len(clean_cache)} verified-clean cached items; running remaining {len(subset) - len(clean_cache)} items...",
                flush=True,
            )

        y_true: List[int] = []
        y_pred: List[int] = []
        records: List[Dict[str, Any]] = []
        clean_ok = 0
        fresh_latencies: List[float] = []
        t_start = time.perf_counter()

        gen_cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        )
        ckpt_file = Path(checkpoint_path) if checkpoint_path else None

        for idx, it in enumerate(subset, 1):
            if it.item_id in clean_cache:
                cached_rec = clean_cache[it.item_id]
                pred = int(cached_rec["pred_corrupted"])
                clean_ok += 1
                y_true.append(it.is_corrupted)
                y_pred.append(pred)
                records.append(cached_rec)
                continue

            if mode == "self_critique":
                prompt = (
                    "Check whether this academic citation and claim look accurate and real. "
                    'Reply ONLY with JSON: {"corrupted": true/false, "reason": "..."}.\n'
                    f"Title: {it.title}\nAuthors: {it.author} ({it.year})\n"
                    f"DOI: {it.doi} | arXiv: {it.arxiv_id}\nClaim: {it.claim_text}"
                )
            else:
                prompt = (
                    "Act as a strict Academic Peer Reviewer auditing for 4 defect types: "
                    "(1) fabricated author/title, (2) mismatched DOI/arXiv ID, "
                    "(3) fabricated or misattributed numeric stats, (4) polarity reversal. "
                    "You do NOT have live internet search; rely only on internal knowledge. "
                    'Reply ONLY with JSON: {"corrupted": true/false, "defect_type": "...", "reason": "..."}.\n'
                    f"Title: {it.title}\nAuthors: {it.author} ({it.year})\n"
                    f"DOI: {it.doi} | arXiv: {it.arxiv_id}\nClaim: {it.claim_text}"
                )

            pred = 0
            raw_reply = ""
            ok_this = False
            call_t0 = time.perf_counter()
            for attempt in range(4):
                try:
                    resp = client.models.generate_content(
                        model=model_name, contents=prompt, config=gen_cfg
                    )
                    raw_reply = (resp.text or "").strip()
                    clean = raw_reply.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(clean)
                    pred = 1 if parsed.get("corrupted") else 0
                    ok_this = True
                    break
                except Exception as exc:
                    raw_reply = f"ERR: {exc}"
                    pred = 0
                    if attempt < 3 and any(
                        k in raw_reply for k in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "500")
                    ):
                        wait = 15.0 * (2**attempt)
                        print(f"  [{mode} Retry {attempt+1}/3] Waiting {wait:.0f}s after rate/server limit...", flush=True)
                        time.sleep(wait)
                        continue
                    break

            call_dt = time.perf_counter() - call_t0
            fresh_latencies.append(call_dt)

            if ok_this:
                clean_ok += 1

            rec_out = {
                "item_id": it.item_id,
                "label_class": it.label_class,
                "is_corrupted": it.is_corrupted,
                "pred_corrupted": pred,
                "clean_ok": ok_this,
                "latency_sec": round(call_dt, 3),
                "raw_reply": raw_reply[:240],
            }
            records.append(rec_out)
            y_true.append(it.is_corrupted)
            y_pred.append(pred)

            if idx == 1 or idx % 5 == 0 or idx == len(subset):
                preview = raw_reply.replace("\n", " ")[:68]
                print(f"  [{mode}] {idx}/{len(subset)} (clean={clean_ok}) -> pred={pred} | {preview}", flush=True)
                if ckpt_file:
                    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
                    ckpt_file.write_text(
                        json.dumps({"records": records}, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

            time.sleep(sleep_sec)

        wall_elapsed = round(time.perf_counter() - t_start, 3)
        mean_api_sec = (
            round(sum(fresh_latencies) / len(fresh_latencies) + sleep_sec, 4)
            if fresh_latencies
            else round(wall_elapsed / max(len(subset), 1), 4)
        )
        metrics = compute_binary_metrics(y_true, y_pred)
        metrics.update(compute_bootstrap_ci(y_true, y_pred))
        metrics["clean_ok"] = clean_ok
        metrics["error_count"] = len(subset) - clean_ok
        metrics["llm_api_calls"] = len(subset)
        metrics["wall_elapsed_sec"] = wall_elapsed
        metrics["sec_per_item"] = mean_api_sec

        out_payload = {
            "method": f"{mode} ({model_name})",
            "metrics": metrics,
            "per_class": _per_class_breakdown(records),
            "records": records,
        }
        if ckpt_file:
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            ckpt_file.write_text(
                json.dumps(out_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return out_payload
