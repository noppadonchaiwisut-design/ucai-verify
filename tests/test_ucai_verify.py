"""Offline deterministic unit tests for ucai-verify v0.2.0."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucai_verify.bench_builder import build_benchmark
from ucai_verify.citation_auditor import _title_containment, _title_similarity
from ucai_verify.evaluator import (
    ExperimentEvaluator,
    compute_binary_metrics,
    compute_bootstrap_ci,
    mcnemar_exact_pvalue,
)
from ucai_verify.numeric_auditor import NumericAuditor, extract_numeric_tokens


class TestUcaiVerifyV2(unittest.TestCase):
    def test_title_similarity_and_containment(self) -> None:
        s_exact = _title_similarity(
            "Towards an AI co-scientist", "Towards an AI co-scientist"
        )
        self.assertGreaterEqual(s_exact, 0.99)
        cont = _title_containment(
            "Towards an AI co-scientist",
            "Towards an AI co-scientist: Accelerating discovery with multi-agent debate",
        )
        self.assertGreaterEqual(cont, 0.75)

    def test_comma_and_proportion_matching(self) -> None:
        aud = NumericAuditor()
        aud.index_text_pages(
            "2605.07723",
            [(1, "Auditing 111 million references across 2.5 million papers reveals 146,932 ghost citations; 85.3% persist.")],
        )
        aud.index_text_pages(
            "2610.23480",
            [(1, "On 199 LitQA2 questions, Munin Qwen3.8-27B reaches 0.874 accuracy and 1.000 [0.963, 1.000] refusal across 100 items.")],
        )

        r1 = aud.audit_claim(
            "Auditing 111 million references in 2.5 million papers reveals 146,932 conservative ghost citations, with 85.3% persisting.",
            "2605.07723",
        )
        self.assertEqual(r1.status, "NUMERIC_VERIFIED")

        # Test percentage-to-proportion equivalence (87.4% in prose vs 0.874 in PDF table)
        r2 = aud.audit_claim(
            "MuninAI Qwen3.8-27B scores 87.4% accuracy on LitQA2 and 100% refusal on 100 items.",
            "2610.23480",
        )
        self.assertEqual(r2.status, "NUMERIC_VERIFIED")

    def test_cross_artifact_misattribution_detection(self) -> None:
        aud = NumericAuditor()
        aud.index_text_pages(
            "2410.05080",
            [(27, "ScienceAgentBench: Claude-3.5-Sonnet achieves 22.9% SR and 84.0% valid execution rate.")],
        )
        aud.index_text_pages(
            "2506.08140",
            [(2, "AutoSDT-5K human experts verified 92.2% program correctness across 5,404 tasks.")],
        )
        res = aud.audit_claim(
            "ScienceAgentBench reports 92.2% valid execution rate for Claude-3.5-Sonnet.",
            "2410.05080",
        )
        self.assertEqual(res.status, "CROSS_ARTIFACT_MISATTRIBUTED")
        self.assertEqual(res.matched_doc_id, "2506.08140")
        self.assertEqual(res.matched_page, 2)

    def test_benchmark_and_offline_evaluator_and_stats(self) -> None:
        items = build_benchmark(None)
        self.assertEqual(len(items), 120)

        ev = ExperimentEvaluator(pdf_dir=None)
        res = ev.run_ucai_verify(items, use_live_apis=False)
        self.assertEqual(res["metrics"]["n"], 120)
        self.assertEqual(res["per_class"]["VALID"]["acc"], 1.0)
        self.assertEqual(res["per_class"]["FAKE_AUTHOR_TITLE"]["acc"], 1.0)
        self.assertEqual(res["per_class"]["BAD_DOI_ARXIV_ID"]["acc"], 1.0)
        self.assertGreaterEqual(res["per_class"]["NUMERIC_MISATTRIBUTION"]["acc"], 0.90)

        y_true = [r["is_corrupted"] for r in res["records"]]
        y_pred = [r["pred_corrupted"] for r in res["records"]]
        ci = compute_bootstrap_ci(y_true, y_pred, n_resamples=200)
        self.assertIn("f1_ci95", ci)
        self.assertEqual(len(ci["f1_ci95"]), 2)

        all_ones = [1] * 120
        mc = mcnemar_exact_pvalue(y_true, y_pred, all_ones)
        self.assertIn("p_value", mc)
        self.assertGreaterEqual(mc["p_value"], 0.0)
        self.assertLessEqual(mc["p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
