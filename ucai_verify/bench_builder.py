"""Builder for UCAI-CitationAudit-Bench-v1 (120 controlled ground-truth cases).

Generates 5 balanced classes (24 items each = 120 total):
  0. VALID                   - Real verified citations + exact source numbers
  1. FAKE_AUTHOR_TITLE       - Fabricated paper titles & non-existent authors
  2. BAD_DOI_ARXIV_ID        - Hijacked or dead DOI / arXiv identifiers
  3. NUMERIC_MISATTRIBUTION  - Real metrics swapped across papers or digit-mutated
  4. POLARITY_REVERSAL       - Real citations paired with inverted directional claims
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List


# Grounded anchor records verified from Paper 0 corpus (2024-2026)
VERIFIED_ANCHORS: List[Dict[str, Any]] = [
    {
        "doc_id": "2408.06292",
        "title": "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery",
        "author": "Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, David Ha",
        "year": 2024,
        "arxiv_id": "2408.06292",
        "doi": "10.48550/arXiv.2408.06292",
        "valid_claim": "The AI Scientist generates a full ML research paper for under $15 in API cost and achieves 0.65 balanced accuracy on ICLR 2022.",
        "source_text": "The AI Scientist generates a complete paper at a cost of less than $15 per paper, achieving 0.65 balanced accuracy on ICLR 2022 reviews compared to 0.66 for humans.",
        "reversed_claim": "The AI Scientist requires over $1500 per paper and underperforms random guessing on ICLR 2022 reviews.",
    },
    {
        "doc_id": "2502.18864",
        "title": "Towards an AI co-scientist",
        "author": "Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Anil Palepu",
        "year": 2025,
        "arxiv_id": "2502.18864",
        "doi": "10.48550/arXiv.2502.18864",
        "valid_claim": "AI Co-Scientist achieves 78.4% top-1 accuracy on GPQA Diamond and demonstrates an 18-fold therapeutic window for KIRA6 in AML.",
        "source_text": "Co-Scientist achieves 78.4% top-1 accuracy on GPQA Diamond. In vitro wet-lab validation of KIRA6 yields an 18-fold therapeutic window (0.01 uM vs 0.18 uM).",
        "reversed_claim": "AI Co-Scientist achieves 78.4% accuracy on GPQA Diamond and fails wet-lab validation with zero therapeutic window for KIRA6.",
    },
    {
        "doc_id": "2409.13740",
        "title": "Language Agents Achieve Superhuman Synthesis of Scientific Knowledge",
        "author": "Michael D. Skarlinski, Sam Cox, Jon M. Laurent, James D. Braza, Andrew D. White",
        "year": 2024,
        "arxiv_id": "2409.13740",
        "doi": "10.48550/arXiv.2409.13740",
        "valid_claim": "On LitQA2, PaperQA2 surpasses the human baseline precision (85.2% vs 73.8%) with 66.0% accuracy.",
        "source_text": "On LitQA2, PaperQA2 achieves 66.0% accuracy and 85.2% precision, which surpasses the human expert baseline precision of 73.8% and 64.1% accuracy.",
        "reversed_claim": "On LitQA2, PaperQA2 underperforms the human baseline precision (85.2% vs 73.8%) and fails literature retrieval.",
    },
    {
        "doc_id": "2410.05080",
        "title": "ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery",
        "author": "Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Huan Sun",
        "year": 2025,
        "arxiv_id": "2410.05080",
        "doi": "10.48550/arXiv.2410.05080",
        "valid_claim": "Under Self-Debug without expert knowledge, Claude-3.5-Sonnet achieves 22.9% success rate and 84.0% valid execution rate.",
        "source_text": "Table E.1: Claude-3.5-Sonnet Self-Debug achieves 22.9% +/- 2.0% SR and 84.0% +/- 1.2% VER ($0.066), while o1-preview achieves 27.1% SR and 84.6% VER.",
        "reversed_claim": "Under Self-Debug without expert knowledge, Claude-3.5-Sonnet achieves 22.9% success rate and degrades below 10% valid execution rate.",
    },
    {
        "doc_id": "2506.08140",
        "title": "AutoSDT: Scaling Data-Driven Discovery Toward Open Co-Scientists",
        "author": "Yizhe Li, Hanyuan Ning Moussa, Ziru Chen, Huan Sun",
        "year": 2025,
        "arxiv_id": "2506.08140",
        "doi": "10.48550/arXiv.2506.08140",
        "valid_claim": "AutoSDT-5K contains 5,404 tasks across 1,325 repositories, with 92.2% human-verified program correctness at $0.55 per task.",
        "source_text": "AutoSDT-5K scales to 5,404 tasks from 1,325 repositories; human experts verified 92.2% program correctness with total construction cost $2,955 (~$0.55/task).",
        "reversed_claim": "AutoSDT-5K contains 5,404 tasks where human experts report 92.2% failure and costs exceed $50 per task.",
    },
    {
        "doc_id": "2505.19955",
        "title": "MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research",
        "author": "Hui Chen, Miao Xiong, Yujie Lu, Wei Han, Bryan Hooi",
        "year": 2025,
        "arxiv_id": "2505.19955",
        "doi": "10.48550/arXiv.2505.19955",
        "valid_claim": "Across 10 ICLR 2025 Trustworthy AI tasks, AI Scientist V2 produced 100% faked experimental results and 90% hallucinated methodology.",
        "source_text": "In our 10-task end-to-end audit (Figure 6, p. 21), AI Scientist V2 (o4-mini-high) produced 100% faked results and 90% hallucinated methodology; MLR-Agent produced 80% faked results.",
        "reversed_claim": "Across 10 ICLR 2025 tasks, AI Scientist V2 eliminates faked results and reduces hallucinated methodology to zero.",
    },
    {
        "doc_id": "2605.07723",
        "title": "LLM hallucinations in the wild: Large-scale evidence from non-existent citations",
        "author": "Z. Zhao, Y. Wang, T. Stuart, M. De Vaan, P. Ginsparg, Y. Yin",
        "year": 2026,
        "arxiv_id": "2605.07723",
        "doi": "10.48550/arXiv.2605.07723",
        "valid_claim": "Auditing 111 million references in 2.5 million papers reveals 146,932 conservative ghost citations in 2025, with 85.3% in bioRxiv persisting into PMC.",
        "source_text": "Auditing 111 million references across 2.5 million papers conservatively estimates 146,932 non-existent citations in 2025; 85.3% of bioRxiv ghost citations persist into PMC.",
        "reversed_claim": "Auditing 111 million references shows ghost citations completely disappear before publication into PMC.",
    },
    {
        "doc_id": "2605.27700",
        "title": "CITECHECK: Retrieval-Grounded Detection of LLM Citation Hallucinations in Scientific Text",
        "author": "Khashayar Khajavi, Shaghayegh Sadeghi, Rise Adhikari, Alexander Tessier",
        "year": 2026,
        "arxiv_id": "2605.27700",
        "doi": "10.48550/arXiv.2605.27700",
        "valid_claim": "CiteCheck achieves 88.7% Macro-F1 and 88.9% accuracy across 982 physics references with an average latency of 8.72 seconds per item.",
        "source_text": "CiteCheck achieves 88.7 Macro-F1 and 88.9% accuracy on 982 physics citations (MAJOR F1 98.3%), averaging 8.72 seconds latency per citation.",
        "reversed_claim": "CiteCheck underperforms Claude Sonnet 4.6 on 982 physics references and reduces Macro-F1 below 50%.",
    },
    {
        "doc_id": "2605.25415",
        "title": "LLM-as-a-Reviewer: Benchmarking Their Ability, Divergence, and Prompt Injection Resistance as Paper Reviewers",
        "author": "L. Li, J. Xiong, C. Zhu, R. Yu, Z. Lu",
        "year": 2026,
        "arxiv_id": "2605.25415",
        "doi": "10.48550/arXiv.2605.25415",
        "valid_claim": "Across 898 papers, invisible PDF prompt injections flip Reject decisions to Accept at rates of 70.89% to 76.80% on GPT-5.4 and Gemini.",
        "source_text": "Evaluating 898 stratified ICLR/NeurIPS papers shows hidden white-text prompt injections flip Reject to Accept at 70.89%-76.80% rates on GPT-5.4 and Gemini.",
        "reversed_claim": "Across 898 papers, GPT-5.4 and Gemini resist hidden prompt injections and block 100% of adversarial flips.",
    },
    {
        "doc_id": "2605.26730",
        "title": "PRISM: A Multi-Dimensional Benchmark for Evaluating LLM Peer Reviewers",
        "author": "N. P. P. Loc, T. H. L. Viet, T. Nguyen, B. T. Nguyen",
        "year": 2026,
        "arxiv_id": "2605.26730",
        "doi": "10.48550/arXiv.2605.26730",
        "valid_claim": "On 600 PRISM papers, Reviewer2 exhibits a ~3.6% hallucination rate compared to ~16.2% for CycleReviewer.",
        "source_text": "Section 4.2.2 (p. 8, 600 papers): Reviewer2 achieves a lower hallucination rate (~3.6%, Critical Recall 0.644), whereas CycleReviewer reaches a higher rate (~16.2%).",
        "reversed_claim": "On 600 PRISM papers, CycleReviewer achieves a lower hallucination rate than Reviewer2 (~3.6% vs ~16.2%).",
    },
    {
        "doc_id": "2610.23480",
        "title": "MuninAI: A Self-Hosted Agentic Framework for Independent Research Groups",
        "author": "M. Fischer, J. Meiler, M. Elgeti",
        "year": 2026,
        "arxiv_id": "2610.23480",
        "doi": "10.48550/arXiv.2610.23480",
        "valid_claim": "Running offline on a single GPU, MuninAI with Qwen3.8-27B scores 87.4% accuracy on LitQA2 and 100% correct refusal on 100 fabricated papers.",
        "source_text": "MuninAI Qwen3.8-27B attains 87.4% accuracy (94.6% precision) on LitQA2 and 100% (100/100) correct refusal on the fabricated-paper test with zero confabulated citations.",
        "reversed_claim": "Running offline, MuninAI falls below PaperQA2 on LitQA2 and fabricates citations on 100% of fake-paper tests.",
    },
    {
        "doc_id": "2511.02824",
        "title": "Kosmos: An AI Scientist for Autonomous Discovery",
        "author": "Ludovico Mitchener, Angela Yiu, Benjamin Chang, Andrew D. White",
        "year": 2025,
        "arxiv_id": "2511.02824",
        "doi": "10.48550/arXiv.2511.02824",
        "valid_claim": "Kosmos maintains 85.5% accuracy on data analysis statements, but accuracy drops to 57.9% on synthesis and interpretation statements.",
        "source_text": "Kosmos achieves 79.4% overall statement accuracy (n=102): 85.5% on data analysis, 82.1% on literature review, dropping to 57.9% on synthesis/interpretation.",
        "reversed_claim": "Kosmos improves accuracy from 85.5% in data analysis up to 99.0% on synthesis and interpretation statements.",
    },
]

FAKE_TITLES = [
    ("OmniVerify: Universal Quantum Proof Auditing for Autonomous Labs", "H. Vance, K. Moriarty, R. Sterling", 2025),
    ("ZeroHalluc: End-to-End Self-Correcting Bibliographies via Hyperbolic Graphs", "T. Lindqvist, M. Okafor, S. Patel", 2026),
    ("SciGuard-XL: Eliminating All Citation Errors Without External APIs", "J. Kowalski, L. Beaumont, Y. Tanaka", 2025),
    ("NeuroCite: Neural Checksums for Peer Review Integrity in Biomedicine", "A. Deshmukh, P. Sorensen, C. Alvarez", 2026),
    ("DeepRef: Zero-Shot Detection of Synthetic Tables in Nature Journals", "R. Gallagher, N. Takahashi, D. Haddad", 2024),
    ("AutoScholarBench: Evaluating 10,000 Autonomous AI Scientists in Wet Labs", "E. Novak, F. Rossi, G. Lindholm", 2026),
    ("BiblioProof: Formal SMT Verification of BibTeX Entries at Scale", "M. Petrov, S. Chatterjee, K. Wright", 2025),
    ("TrustSci-2026: A Global Registry of Verified Multi-Agent Manuscripts", "L. Goldstein, H. Park, O. Adeyemi", 2026),
    ("CiteShield: Cryptographic Watermarking for LLM Literature Synthesis", "V. Subramanian, T. Müller, J. Ortiz", 2025),
    ("EpistemicNet: Predicting Hallucination Rates from Token Entropy Alone", "C. Moreau, B. Al-Mansoor, K. Watanabe", 2026),
    ("PeerOracle: Replacing Human Program Chairs with Consensus Transformers", "D. Kaczmarek, R. Singh, M. Dupont", 2025),
    ("LabTwin-Pro: Autonomous Wet-Lab Robotics with Deterministic Guarantee", "S. Alverez, P. Lindberg, T. Nguyen-Tran", 2026),
]


@dataclass
class BenchItem:
    item_id: str
    label_class: str  # VALID | FAKE_AUTHOR_TITLE | BAD_DOI_ARXIV_ID | NUMERIC_MISATTRIBUTION | POLARITY_REVERSAL
    is_corrupted: int  # 0 = clean, 1 = corrupted
    cited_doc_id: str
    title: str
    author: str
    year: int
    doi: str
    arxiv_id: str
    claim_text: str
    ground_truth_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_benchmark(output_path: str | Path | None = None) -> List[BenchItem]:
    """Builds the 120-item UCAI-CitationAudit-Bench-v1 dataset (24 items x 5 classes)."""
    items: List[BenchItem] = []
    n_anchors = len(VERIFIED_ANCHORS)

    # Class 0: VALID (24 items: 2 variants per anchor)
    for i in range(24):
        anc = VERIFIED_ANCHORS[i % n_anchors]
        items.append(
            BenchItem(
                item_id=f"UCAI-BENCH-{len(items)+1:03d}",
                label_class="VALID",
                is_corrupted=0,
                cited_doc_id=anc["doc_id"],
                title=anc["title"],
                author=anc["author"],
                year=anc["year"],
                doi=anc["doi"],
                arxiv_id=anc["arxiv_id"],
                claim_text=anc["valid_claim"],
                ground_truth_reason="Verified bibliographic record and numeric claim match primary source.",
            )
        )

    # Class 1: FAKE_AUTHOR_TITLE (24 items: 2 variants per fake paper)
    for i in range(24):
        ft, fa, fy = FAKE_TITLES[i % len(FAKE_TITLES)]
        suffix = "" if i < 12 else " (Extended Empirical Edition)"
        items.append(
            BenchItem(
                item_id=f"UCAI-BENCH-{len(items)+1:03d}",
                label_class="FAKE_AUTHOR_TITLE",
                is_corrupted=1,
                cited_doc_id=VERIFIED_ANCHORS[i % n_anchors]["doc_id"],
                title=ft + suffix,
                author=fa,
                year=fy,
                doi=f"10.48550/arXiv.2699.{90000 + i}",
                arxiv_id=f"2699.{90000 + i}",
                claim_text=f"According to {fa.split(',')[0]}, {ft} achieves 99.1% audit accuracy across 500 papers.",
                ground_truth_reason="Completely fabricated title, authors, and synthetic identifier.",
            )
        )

    # Class 2: BAD_DOI_ARXIV_ID (24 items: real title paired with hijacked or non-existent ID)
    for i in range(24):
        anc = VERIFIED_ANCHORS[i % n_anchors]
        other = VERIFIED_ANCHORS[(i + 3) % n_anchors]
        hijacked = (i % 2 == 0)
        bad_arxiv = other["arxiv_id"] if hijacked else f"2599.{88000 + i}"
        bad_doi = other["doi"] if hijacked else f"10.1109/FAKE.{202500 + i}"
        items.append(
            BenchItem(
                item_id=f"UCAI-BENCH-{len(items)+1:03d}",
                label_class="BAD_DOI_ARXIV_ID",
                is_corrupted=1,
                cited_doc_id=anc["doc_id"],
                title=anc["title"],
                author=anc["author"],
                year=anc["year"],
                doi=bad_doi,
                arxiv_id=bad_arxiv,
                claim_text=anc["valid_claim"],
                ground_truth_reason=(
                    f"Identifier hijacked: points to '{other['title']}'"
                    if hijacked
                    else "Real paper paired with non-existent DOI/arXiv ID."
                ),
            )
        )

    # Class 3: NUMERIC_MISATTRIBUTION (24 items: swap real metrics across papers or mutate digits)
    for i in range(24):
        anc = VERIFIED_ANCHORS[i % n_anchors]
        donor = VERIFIED_ANCHORS[(i + 1) % n_anchors]
        # First 12 items reproduce cross-artifact misattribution (e.g. AutoSDT 92.2% attributed to ScienceAgentBench!)
        if i % 2 == 0:
            claim = f"In {anc['title'].split(':')[0]}, experiments report: {donor['valid_claim']}"
            reason = f"Cross-artifact misattribution: metric actually comes from '{donor['doc_id']}'."
        else:
            # Digit mutation (e.g. changing 22.9% -> 92.9% or 78.4% -> 98.4%)
            mutated = anc["valid_claim"].replace("22.9%", "92.9%").replace("78.4%", "98.4%").replace("66.0%", "96.0%").replace("88.9%", "99.9%").replace("100%", "12%").replace("87.4%", "37.4%").replace("85.5%", "15.5%").replace("15", "950").replace("3.6%", "43.6%").replace("70.89%", "10.89%").replace("92.2%", "22.2%").replace("146,932", "946,932")
            if mutated == anc["valid_claim"]:
                mutated = anc["valid_claim"] + " achieving 99.7% zero-error guarantee."
            claim = mutated
            reason = "Numeric fabrication / digit mutation inconsistent with cited PDF."
        items.append(
            BenchItem(
                item_id=f"UCAI-BENCH-{len(items)+1:03d}",
                label_class="NUMERIC_MISATTRIBUTION",
                is_corrupted=1,
                cited_doc_id=anc["doc_id"],
                title=anc["title"],
                author=anc["author"],
                year=anc["year"],
                doi=anc["doi"],
                arxiv_id=anc["arxiv_id"],
                claim_text=claim,
                ground_truth_reason=reason,
            )
        )

    # Class 4: POLARITY_REVERSAL (24 items: real source cited with inverted conclusion)
    for i in range(24):
        anc = VERIFIED_ANCHORS[i % n_anchors]
        items.append(
            BenchItem(
                item_id=f"UCAI-BENCH-{len(items)+1:03d}",
                label_class="POLARITY_REVERSAL",
                is_corrupted=1,
                cited_doc_id=anc["doc_id"],
                title=anc["title"],
                author=anc["author"],
                year=anc["year"],
                doi=anc["doi"],
                arxiv_id=anc["arxiv_id"],
                claim_text=anc["reversed_claim"],
                ground_truth_reason="Real citation, but claim reverses the source paper's conclusion.",
            )
        )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([it.to_dict() for it in items], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return items
