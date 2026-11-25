"""Single-command CLI for running UCAI-Verify v0.2.0 & Tier-2 120-Item Experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ucai_verify.bench_builder import build_benchmark
from ucai_verify.evaluator import ExperimentEvaluator, mcnemar_exact_pvalue


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UCAI-Verify v0.2.0 & UCAI-CitationAudit-Bench-v1")
    parser.add_argument("--pdf-dir", type=str, default="", help="Folder containing real PDFs")
    parser.add_argument("--out-dir", type=str, default="results_tier2", help="Directory for JSON/CSV logs")
    parser.add_argument("--offline", action="store_true", help="Skip live arXiv/Crossref/OpenAlex API calls")
    parser.add_argument("--run-hybrid-nli", action="store_true", help="Run Stage 2 Local NLI Cross-Encoder")
    parser.add_argument("--run-hybrid-gemini", action="store_true", help="Run Stage 2 Scoped Gemini Judge")
    parser.add_argument("--run-gemini-baselines", action="store_true", help="Run Self-Critique & LLM-as-a-Judge")
    parser.add_argument("--max-llm-items", type=int, default=120, help="Items for LLM baselines (default: 120)")
    parser.add_argument("--sleep-sec", type=float, default=4.2, help="Pacing between Gemini API calls")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_path = out_dir / "UCAI-CitationAudit-Bench-v1.json"
    items = build_benchmark(bench_path)
    print(f"[1/4] Built UCAI-CitationAudit-Bench-v1: {len(items)} items -> {bench_path}")

    evaluator = ExperimentEvaluator(pdf_dir=args.pdf_dir or None)
    print(f"[2/4] Indexed real PDFs: {evaluator.pdf_loaded_count}/12 anchors (plus grounded fallbacks)")

    ucai_res = evaluator.run_ucai_verify(items, use_live_apis=not args.offline)
    (out_dir / "ucai_verify_results.json").write_text(
        json.dumps(ucai_res, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[3/4] ucai-verify Stage 1 metrics:", json.dumps(ucai_res["metrics"], indent=2))
    print("Per-class accuracy:", json.dumps(ucai_res["per_class"], indent=2))

    if args.run_hybrid_nli:
        hnli_res = evaluator.run_hybrid_local_nli(items, ucai_res)
        (out_dir / "hybrid_local_nli_results.json").write_text(
            json.dumps(hnli_res, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("[Hybrid Local NLI] metrics:", json.dumps(hnli_res["metrics"], indent=2))

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if args.run_hybrid_gemini:
        if not api_key:
            print("[WARN] GEMINI_API_KEY not set; skipping Hybrid Scoped Gemini.")
        else:
            hgem_res = evaluator.run_hybrid_scoped_gemini(
                items,
                ucai_res,
                api_key=api_key,
                checkpoint_path=out_dir / "hybrid_gemini_results.json",
                sleep_sec=args.sleep_sec,
            )
            (out_dir / "hybrid_gemini_results.json").write_text(
                json.dumps(hgem_res, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print("[Hybrid Scoped Gemini] metrics:", json.dumps(hgem_res["metrics"], indent=2))

    if args.run_gemini_baselines:
        if not api_key:
            print("[WARN] GEMINI_API_KEY not set; skipping Gemini baselines.")
            return
        sc_res = evaluator.run_gemini_baseline(
            items,
            api_key=api_key,
            mode="self_critique",
            max_items=args.max_llm_items,
            sleep_sec=args.sleep_sec,
            checkpoint_path=out_dir / "baseline_self_critique_120.json",
        )
        lj_res = evaluator.run_gemini_baseline(
            items,
            api_key=api_key,
            mode="llm_judge",
            max_items=args.max_llm_items,
            sleep_sec=args.sleep_sec,
            checkpoint_path=out_dir / "baseline_llm_judge_120.json",
        )
        y_true = [it.is_corrupted for it in items[: len(lj_res["records"])]]
        sig = mcnemar_exact_pvalue(
            y_true,
            [r["pred_corrupted"] for r in ucai_res["records"][: len(y_true)]],
            [r["pred_corrupted"] for r in lj_res["records"]],
        )
        print("[4/4] McNemar (ucai-verify vs LLM-Judge):", sig)


if __name__ == "__main__":
    main()
