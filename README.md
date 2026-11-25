# `ucai-verify` (v0.2.0 — Tier-2 Edition)

[![License: MIT](https://img.shields.io/badge/Code_License-MIT-blue.svg)](LICENSE)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/Benchmark-CC_BY_4.0-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](pyproject.toml)

**Zero-Fake-Citation & Cross-Document Numeric Audit Engine** with **Two-Stage Hybrid Semantic Polarity Verification** for eliminating ghost citations and cross-artifact numeric swaps in AI-generated scientific manuscripts.

Developed as **Paper 1 (Spoke 1)** of the **UCAI Flagship 4** monograph (*Open Verifiable AI Co-Scientist & Research-in-a-Box*).

---

## Key Capabilities (v0.2.0 Tier-2)

1. **Stage 1A — Deterministic Multi-Source Citation Cascade (`$0` Token Cost):**
   - Resolves persistent identifiers (`arXiv Atom API`, `Crossref REST API`) with **version-aware preprint title resolution** (`v1` -> `v2+`) and falls back to title search across `Crossref` and `OpenAlex`.
   - Separates `VERIFIED_EXACT`, `METADATA_DRIFT`, `IDENTIFIER_HIJACKED`, and `SUSPECTED_FAKE`.
2. **Stage 1B — Page-Indexed Cross-PDF Numeric Matcher (`PyMuPDF`):**
   - Indexes full-text PDFs page-by-page, normalizes thousand separators (`146,932` <-> `146932`) and table proportions (`87.4%` <-> `0.874`), and scans the entire local library to detect **`CROSS_ARTIFACT_MISATTRIBUTED`** numbers with exact `(doc_id, page_#)` provenance.
3. **Stage 2 — Scoped Hybrid Polarity Verifier (Dual Executable Backends):**
   - Invoked **only** on claims that pass Stage 1 (~63%–68% call reduction), avoiding the Pure-LLM False-Positive Trap (`FPR > 80%`):
     - **`Hybrid-Local-NLI` (`$0` API Cost, 100% Sovereign):** Open-weight `cross-encoder/nli-deberta-v3-base` checking `(PDF evidence, claim)` contradiction locally on CPU/GPU.
     - **`Hybrid-Scoped-Gemini`:** Grounded cloud verifier checking directional alignment strictly against the retrieved PDF snippet.
4. **`UCAI-CitationAudit-Bench-v1` (120 Ground-Truth Items, `CC BY 4.0`):**
   - 5 balanced classes (`24 items x 5 = 120`): `VALID`, `FAKE_AUTHOR_TITLE`, `BAD_DOI_ARXIV_ID`, `NUMERIC_MISATTRIBUTION`, and `POLARITY_REVERSAL` anchored on 12 open-access papers (2024–2026) with 95% Bootstrap CIs and exact McNemar's test.

---

## Empirical Benchmark Results (`UCAI-CitationAudit-Bench-v1`, Full `N = 120`)

| Verification Regime (`N = 120`, 100% Clean) | Accuracy | Precision | Recall | **F1 [95% CI]** | **FPR ↓** | LLM Calls | Sec / Item |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ucai-verify` Stage 1 (`$0`)** | 0.8000 | **1.0000** | 0.7500 | 0.8571 `[0.800, 0.911]` | **0.0000** | **0** | **0.15 s** |
| **`Hybrid — Local NLI` (`$0`)** | 0.9000 | 0.9565 | 0.9167 | **0.9362** `[0.898, 0.968]`* | 0.1667 | **0** | 0.98 s |
| **`Hybrid — Scoped Gemini` ★** | **0.9667** | 0.9600 | **1.0000** | **0.9796** `[0.959, 0.995]`*** | 0.1667 | **48 (-60%)** | 3.03 s |
| `Pure Self-Critique` (120) | 0.8083 | 0.8230 | 0.9688 | 0.8900 `[0.843, 0.931]` | 0.8333 | 120 | 9.07 s |
| `Pure LLM-as-a-Judge` (120) | 0.7917 | 0.8087 | 0.9688 | 0.8815 `[0.828, 0.927]` | 0.9167 | 120 | 7.07 s |

> `*p = 0.0146` (`19:6` discordant wins) and `***p = 1.0e-6` (`21:0` discordant wins) vs. `Pure LLM-as-a-Judge` by exact two-sided McNemar's test. Full raw artifacts are stored in [`results_tier2/`](results_tier2/) and the English LaTeX paper is in [`manuscript_en/main.tex`](manuscript_en/main.tex).

---

## Repository Structure

```text
ucai-verify/
├── ucai_verify/
│   ├── __init__.py           # Package exports (v0.2.0, MIT)
│   ├── citation_auditor.py   # Multi-API bibliographic & version-aware ID verifier
│   ├── numeric_auditor.py    # Page-indexed PDF numeric & cross-artifact matcher
│   ├── hybrid_auditor.py     # Real Stage-2 Local NLI & Scoped Gemini cascade
│   ├── bench_builder.py      # Generates UCAI-CitationAudit-Bench-v1 (120 items)
│   └── evaluator.py          # Full-120 evaluation, checkpointing, Bootstrap CI & McNemar
├── tests/
│   └── test_ucai_verify.py   # Offline deterministic unit test suite
├── manuscript_en/            # International English LaTeX & BibTeX publication bundle
├── cli.py                    # Single-command CLI runner
├── UCAI_Paper1_ucai_verify_Colab.ipynb  # 1-Click Reproducible Google Colab Notebook
├── pyproject.toml            # Standard PEP 621 Python package config
├── requirements.txt          # Pinned runtime & evaluation dependencies
├── CITATION.cff              # GitHub / Zenodo citation metadata
├── .zenodo.json              # CERN Zenodo DOI deposit metadata
└── LICENSE                   # MIT (Code) + CC BY 4.0 (Dataset)
```

---

## Quick Start

### 1. Local Installation & Unit Tests
```bash
pip install -e ".[eval]"
python -m unittest discover -s tests -v
```

### 2. Run Full 120-Item Benchmark via CLI
```bash
# Deterministic Stage 1 + Local Open-Weight NLI Hybrid ($0 API cost):
python cli.py --pdf-dir /path/to/pdfs --out-dir results_tier2 --run-hybrid-nli

# Full 5-way comparison including Scoped Hybrid + Pure LLM Baselines (120 items):
export GEMINI_API_KEY="your_key_here"
python cli.py --pdf-dir /path/to/pdfs --out-dir results_tier2 \
  --run-hybrid-nli --run-hybrid-gemini --run-gemini-baselines --max-llm-items 120
```

### 3. One-Click Reproduction on Google Colab
1. Open `UCAI_Paper1_ucai_verify_Colab.ipynb` in **Google Colaboratory**.
2. *(Optional for Cloud LLM baselines)* Add `GEMINI_API_KEY` in Colab **Secrets** (left key icon).
3. Click **`Runtime -> Run all`** — checkpoints save automatically every 5 items to `results_tier2/` on Google Drive.

---

## Governance & Responsible Use (`L2 HITL`)

`ucai-verify` operates strictly as an **`L2 (Assistive Evidence & Human Approval)`** tool. Automated flags (`SUSPECTED_FAKE`, `CROSS_ARTIFACT_MISATTRIBUTED`, `STAGE2_NLI_CONTRADICTION`) generate page-level audit evidence for human editors and authors; they must never be used to issue automated misconduct sanctions without human verification.

---

## Citation

See [`CITATION.cff`](CITATION.cff) or cite:

```bibtex
@software{chaiwisut2026ucaiverify,
  author       = {Chaiwisut, Noppadon and {UCAI Autonomous Research Group}},
  title        = {{ucai-verify: Zero-Fake-Citation \& Cross-Document Numeric Audit Engine}},
  version      = {0.2.0},
  year         = {2026},
  license      = {MIT}
}
```
