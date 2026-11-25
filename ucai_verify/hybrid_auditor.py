"""Real Two-Stage Deterministic-First Hybrid Cascade Auditor (v0.2.0).

Executes Stage 2 semantic/polarity verification ONLY on claims that pass
Stage 1 deterministic bibliographic & numeric verification (VERIFIED_EXACT +
NUMERIC_VERIFIED), supporting two executable Stage-2 backends:
  1. Local Open-Weight Cross-Encoder NLI (`cross-encoder/nli-deberta-v3-base`, $0 API cost)
  2. Scoped Cloud LLM Polarity Verifier (`gemini-flash-lite-latest` grounded on PDF snippet)
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .bench_builder import BenchItem


def _softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


class HybridCascadeAuditor:
    """Two-Stage Hybrid Verification Engine (Stage 1 Deterministic -> Stage 2 Semantic Polarity)."""

    def __init__(self, nli_model_name: str = "cross-encoder/nli-deberta-v3-base") -> None:
        self.nli_model_name = nli_model_name
        self._nli_encoder: Any = None
        self._contradiction_idx: int = 0
        self._entailment_idx: int = 1

    def _load_local_nli(self) -> Any:
        if self._nli_encoder is not None:
            return self._nli_encoder
        from sentence_transformers import CrossEncoder

        encoder = CrossEncoder(self.nli_model_name)
        id2label = getattr(encoder.model.config, "id2label", {}) or {}
        for idx, lbl in id2label.items():
            low = str(lbl).lower()
            if "contradict" in low:
                self._contradiction_idx = int(idx)
            elif "entail" in low:
                self._entailment_idx = int(idx)
        self._nli_encoder = encoder
        return encoder

    def run_local_nli_cascade(
        self,
        items: List[BenchItem],
        stage1_records: List[Dict[str, Any]],
        stage1_latency_sec: float,
        contradiction_threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """Runs Stage 2 Open-Weight NLI ONLY on items passing Stage 1 ($0 API cost)."""
        encoder = self._load_local_nli()
        t0 = time.perf_counter()

        survivor_indices: List[int] = []
        pair_item_indices: List[int] = []
        pairs: List[tuple[str, str]] = []
        for idx, rec in enumerate(stage1_records):
            if rec["pred_corrupted"] == 0:
                evidence = rec.get("evidence_context") or rec.get("evidence_snippet") or ""
                windows = [w.strip() for w in evidence.split(" ... ") if w.strip()][:5]
                if windows:
                    survivor_indices.append(idx)
                    for win in windows:
                        pair_item_indices.append(idx)
                        pairs.append((win, items[idx].claim_text))

        probs_by_idx: Dict[int, Dict[str, float]] = {}
        if pairs:
            try:
                raw_scores = encoder.predict(pairs, apply_softmax=False)
            except TypeError:
                raw_scores = encoder.predict(pairs)
            grouped: Dict[int, List[Tuple[float, float]]] = {}
            for s_idx, row in zip(pair_item_indices, raw_scores):
                row_list = [float(x) for x in row]
                if all(0.0 <= x <= 1.0 for x in row_list) and abs(sum(row_list) - 1.0) < 1e-3:
                    probs = row_list
                else:
                    probs = _softmax(row_list)
                grouped.setdefault(s_idx, []).append(
                    (probs[self._contradiction_idx], probs[self._entailment_idx])
                )
            for s_idx, scores_list in grouped.items():
                max_con = max(c for c, _ in scores_list)
                max_ent = max(e for _, e in scores_list)
                probs_by_idx[s_idx] = {
                    "p_contradiction": round(max_con, 4),
                    "p_entailment": round(max_ent, 4),
                }

        stage2_elapsed = round(time.perf_counter() - t0, 3)
        records: List[Dict[str, Any]] = []
        for idx, (it, s1) in enumerate(zip(items, stage1_records)):
            if s1["pred_corrupted"] == 1:
                records.append(
                    {
                        **s1,
                        "stage2_invoked": False,
                        "stage2_verdict": "SKIPPED_CAUGHT_BY_STAGE1",
                        "pred_corrupted": 1,
                    }
                )
            elif idx in probs_by_idx:
                p_con = probs_by_idx[idx]["p_contradiction"]
                p_ent = probs_by_idx[idx]["p_entailment"]
                flagged = (p_con >= 0.70 and p_con > p_ent) or (
                    p_con >= contradiction_threshold and p_con > p_ent and p_ent < 0.70
                )
                records.append(
                    {
                        **s1,
                        "stage2_invoked": True,
                        "stage2_verdict": "STAGE2_NLI_CONTRADICTION" if flagged else "STAGE2_NLI_ENTAILED",
                        "p_contradiction": p_con,
                        "p_entailment": p_ent,
                        "pred_corrupted": 1 if flagged else 0,
                    }
                )
            else:
                records.append(
                    {
                        **s1,
                        "stage2_invoked": False,
                        "stage2_verdict": "ABSTAINED_NO_EVIDENCE",
                        "pred_corrupted": 0,
                    }
                )

        return {
            "method": f"Hybrid Cascade — Local NLI ({self.nli_model_name})",
            "stage1_calls": len(items),
            "stage2_calls": len(pairs),
            "stage2_call_reduction_pct": round(100.0 * (1.0 - len(pairs) / max(len(items), 1)), 2),
            "stage1_latency_sec": stage1_latency_sec,
            "stage2_latency_sec": stage2_elapsed,
            "total_latency_sec": round(stage1_latency_sec + stage2_elapsed, 3),
            "records": records,
        }

    def run_scoped_gemini_cascade(
        self,
        items: List[BenchItem],
        stage1_records: List[Dict[str, Any]],
        stage1_latency_sec: float,
        client: Any,
        model_name: str,
        checkpoint_path: Optional[Path] = None,
        sleep_sec: float = 4.2,
    ) -> Dict[str, Any]:
        """Runs Stage 2 Scoped Gemini Judge ONLY on Stage-1 pass items with checkpointing."""
        from google.genai import types

        cached_map: Dict[str, Dict[str, Any]] = {}
        if checkpoint_path and checkpoint_path.exists():
            try:
                old = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                for r in old.get("records", []):
                    if r.get("stage2_ok"):
                        cached_map[r["item_id"]] = r
            except Exception:
                pass

        gen_cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        )

        t0 = time.perf_counter()
        stage2_calls = 0
        stage2_clean_ok = 0
        records: List[Dict[str, Any]] = []

        survivors_total = sum(1 for r in stage1_records if r["pred_corrupted"] == 0)
        surv_seen = 0

        for it, s1 in zip(items, stage1_records):
            if s1["pred_corrupted"] == 1:
                records.append(
                    {
                        **s1,
                        "stage2_invoked": False,
                        "stage2_ok": True,
                        "stage2_verdict": "SKIPPED_CAUGHT_BY_STAGE1",
                        "pred_corrupted": 1,
                    }
                )
                continue

            evidence = (s1.get("evidence_context") or s1.get("evidence_snippet") or "").strip()
            if not evidence:
                records.append(
                    {
                        **s1,
                        "stage2_invoked": False,
                        "stage2_ok": True,
                        "stage2_verdict": "ABSTAINED_NO_EVIDENCE",
                        "pred_corrupted": 0,
                    }
                )
                continue

            surv_seen += 1
            stage2_calls += 1

            if it.item_id in cached_map:
                prev = cached_map[it.item_id]
                stage2_clean_ok += 1
                records.append(prev)
                continue

            prompt = (
                "You are Stage-2 of a scientific verification pipeline. "
                "Stage-1 has ALREADY deterministically verified that the cited paper exists and its numbers appear in the PDF. "
                "Your ONLY task is to check whether the CLAIM reverses, contradicts, or distorts the directional conclusion "
                "or magnitude meaning of the PRIMARY SOURCE EVIDENCE snippet below.\n"
                "- Do NOT flag minor paraphrasing or rounded summaries if the scientific direction matches.\n"
                "- Flag contradicts_evidence=true ONLY if the claim states the opposite outcome, swaps which model won, "
                "or claims failure/degradation where the source reports success/improvement.\n"
                'Reply ONLY with JSON: {"contradicts_evidence": true/false, "reason": "..."}.\n\n'
                f"PRIMARY SOURCE EVIDENCE (from cited PDF): {evidence}\n"
                f"CLAIM TO CHECK: {it.claim_text}"
            )

            pred = 0
            raw_reply = ""
            ok_this = False
            for attempt in range(4):
                try:
                    resp = client.models.generate_content(
                        model=model_name, contents=prompt, config=gen_cfg
                    )
                    raw_reply = (resp.text or "").strip()
                    clean = raw_reply.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(clean)
                    pred = 1 if parsed.get("contradicts_evidence") else 0
                    ok_this = True
                    break
                except Exception as exc:
                    raw_reply = f"ERR: {exc}"
                    if attempt < 3 and any(
                        k in raw_reply for k in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "500")
                    ):
                        wait = 15.0 * (2**attempt)
                        print(f"  [Stage2 Retry {attempt+1}/3] Waiting {wait:.0f}s after transient API limit...", flush=True)
                        time.sleep(wait)
                        continue
                    break

            if ok_this:
                stage2_clean_ok += 1

            rec_out = {
                **s1,
                "stage2_invoked": True,
                "stage2_ok": ok_this,
                "stage2_verdict": "STAGE2_LLM_CONTRADICTION" if pred == 1 else "STAGE2_LLM_CONSISTENT",
                "stage2_raw_reply": raw_reply[:240],
                "pred_corrupted": pred,
            }
            records.append(rec_out)

            if surv_seen == 1 or surv_seen % 5 == 0 or surv_seen == survivors_total:
                print(
                    f"  [Hybrid-Stage2] {surv_seen}/{survivors_total} Stage-1 survivors checked -> pred={pred}",
                    flush=True,
                )
                if checkpoint_path:
                    checkpoint_path.write_text(
                        json.dumps({"records": records}, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

            time.sleep(sleep_sec)

        stage2_elapsed = round(time.perf_counter() - t0, 3)
        if checkpoint_path:
            checkpoint_path.write_text(
                json.dumps({"records": records}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        return {
            "method": f"Hybrid Cascade — Scoped LLM ({model_name})",
            "stage1_calls": len(items),
            "stage2_calls": stage2_calls,
            "stage2_clean_ok": stage2_clean_ok,
            "stage2_error_count": stage2_calls - stage2_clean_ok,
            "stage2_call_reduction_pct": round(100.0 * (1.0 - stage2_calls / max(len(items), 1)), 2),
            "stage1_latency_sec": stage1_latency_sec,
            "stage2_latency_sec": stage2_elapsed,
            "total_latency_sec": round(stage1_latency_sec + stage2_elapsed, 3),
            "records": records,
        }
