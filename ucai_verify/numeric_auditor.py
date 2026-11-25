"""Deterministic PDF Numeric & Cross-Document Claim Matcher (v0.2.0).

Extracts page-indexed text from source PDFs (using PyMuPDF) and checks whether
quantitative metrics cited in a draft appear in the cited paper, appear in a
DIFFERENT paper in the corpus (Cross-Artifact Misattribution), or lack support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_NUM_PATTERN = re.compile(
    r"(?:\$|USD\s*)?\b\d{1,4}(?:,\d{3})*(?:\.\d+)?%?(?:\s*[±]\s*\d+(?:\.\d+)?%?)?"
)

_STOP_KEYWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "were", "have", "been",
    "model", "paper", "using", "across", "results", "benchmark", "rate", "score",
}

_ANTONYM_PAIRS = [
    ("surpasses", "underperforms"),
    ("outperforms", "underperforms"),
    ("superhuman", "underperforms"),
    ("matches", "underperforms"),
    ("beats", "underperforms"),
    ("outperforms", "fails"),
    ("achieves", "fails"),
    ("success", "failure"),
    ("succeed", "fails"),
    ("higher", "lower"),
    ("better", "worse"),
    ("increases", "decreases"),
    ("improves", "degrades"),
    ("valid", "degrades"),
    ("reduces", "increases"),
    ("above", "below"),
    ("exceeds", "falls short"),
    ("persists", "disappears"),
    ("persist", "disappear"),
    ("produced", "eliminates"),
    ("yields", "fails"),
    ("resists", "flip"),
    ("attains", "falls below"),
    ("improves", "drops"),
    ("effective", "ineffective"),
]


def extract_numeric_tokens(text: str) -> List[str]:
    """Extracts salient numeric metrics (percentages, decimals, counts >= 10, currency)."""
    tokens: List[str] = []
    for raw in _NUM_PATTERN.findall(text or ""):
        clean = raw.strip()
        # Filter trivial years 2020-2029 or tiny single digits unless percentage/decimal/currency
        if re.fullmatch(r"202\d", clean):
            continue
        if re.fullmatch(r"\d", clean):
            continue
        if len(clean) >= 2:
            tokens.append(clean)
    return list(dict.fromkeys(tokens))


def _normalize_num(token: str) -> str:
    return token.replace(",", "").replace("USD", "").replace("$", "").replace("%", "").strip()


def _build_number_regex(num_token: str) -> Optional[re.Pattern[str]]:
    """Builds a regex matching a number with thousand separators or table proportion form (87.4% <-> 0.874)."""
    is_pct = "%" in (num_token or "")
    norm = _normalize_num(num_token).split("±")[0].strip()
    if not norm:
        return None
    parts = norm.split(".", 1)
    int_part = parts[0]
    dec_part = f"\\.{re.escape(parts[1])}" if len(parts) > 1 else ""

    alts = [re.escape(int_part) + dec_part]
    if int_part.isdigit() and len(int_part) >= 4:
        # Insert optional comma / thin-space thousand separators from right to left
        groups: List[str] = []
        rem = int_part
        while len(rem) > 3:
            groups.append(rem[-3:])
            rem = rem[:-3]
        groups.append(rem)
        groups.reverse()
        grouped_expr = r"[,\s\u00a0\u2009]?".join(re.escape(g) for g in groups) + dec_part
        alts.append(grouped_expr)

    if is_pct:
        try:
            val = float(norm)
            if 0.0 < val < 100.0:
                prop_trim = f"{val / 100.0:.6f}".rstrip("0").rstrip(".")
                prop_3d = f"{val / 100.0:.3f}"
                alts.append(re.escape(prop_trim))
                alts.append(re.escape(prop_3d))
            elif val == 100.0:
                alts.extend([r"1\.000", r"1\.00"])
        except ValueError:
            pass

    combined = "|".join(dict.fromkeys(alts))
    return re.compile(rf"(?<!\d)(?:{combined})(?!\d)")


def _extract_context_keywords(claim_text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", (claim_text or "").lower())
    return [w for w in words if w not in _STOP_KEYWORDS]


def _clean_pdf_page_text(raw: str) -> str:
    """Rejoins hyphenated linebreaks and normalizes whitespace in extracted PDF text."""
    dehyphen = re.sub(r"-\s*\n\s*", "", raw or "")
    return re.sub(r"\s+", " ", dehyphen).strip()


@dataclass
class NumericAuditResult:
    """Audit outcome for a numeric claim against a corpus of PDFs."""

    claim_text: str
    cited_doc_id: str
    extracted_numbers: List[str]
    status: str  # NUMERIC_VERIFIED | CROSS_ARTIFACT_MISATTRIBUTED | POLARITY_CONFLICT | NUMERIC_UNSUPPORTED | DOC_NOT_INDEXED
    matched_doc_id: Optional[str]
    matched_page: Optional[int]
    evidence_snippet: str
    best_evidence_sentence: str
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NumericAuditor:
    """Page-indexed PDF corpus numeric & claim consistency auditor."""

    def __init__(self) -> None:
        self.corpus_pages: Dict[str, List[Tuple[int, str]]] = {}

    def index_pdf(self, doc_id: str, pdf_path: str | Path) -> int:
        """Indexes all pages of a PDF file using PyMuPDF (fitz)."""
        import fitz  # PyMuPDF

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        pages: List[Tuple[int, str]] = []
        with fitz.open(str(path)) as doc:
            for idx, page in enumerate(doc, start=1):
                pages.append((idx, _clean_pdf_page_text(page.get_text("text") or "")))
        # Merge or keep longest page count if multiple PDF versions exist for same doc_id
        existing = self.corpus_pages.get(doc_id)
        if existing is None or len(pages) >= len(existing):
            self.corpus_pages[doc_id] = pages
        return len(self.corpus_pages[doc_id])

    def index_text_pages(self, doc_id: str, pages: List[Tuple[int, str]]) -> None:
        """Registers pre-extracted (page_num, text) tuples (useful for unit tests & offline runs)."""
        self.corpus_pages[doc_id] = [(p, _clean_pdf_page_text(t)) for p, t in pages]

    def _best_overlap_sentence(
        self, doc_id: str, keywords: List[str], max_pages: Optional[int] = None
    ) -> Tuple[Optional[int], str, int]:
        """Finds the 1-2 sentence window in doc_id with highest lexical keyword overlap."""
        pages = self.corpus_pages.get(doc_id, [])
        if max_pages is not None:
            pages = pages[:max_pages]
        best_page: Optional[int] = None
        best_snip = ""
        best_hits = 0
        for page_num, text in pages:
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            for i, sent in enumerate(sents):
                window = sent if i + 1 >= len(sents) else f"{sent} {sents[i + 1]}"
                w_low = window.lower()
                hits = sum(1 for k in keywords if k in w_low)
                if hits > best_hits:
                    best_hits = hits
                    best_page = page_num
                    best_snip = window[:380]
        return best_page, best_snip, best_hits

    def _top_overlap_windows(
        self, doc_id: str, keywords: List[str], nums: List[str], top_k: int = 5
    ) -> List[str]:
        """Returns top-k deduplicated 1-2 sentence windows ranked by keyword + numeric overlap."""
        pages = self.corpus_pages.get(doc_id, [])
        norm_nums = [_normalize_num(n).split("±")[0].strip() for n in nums]
        scored: List[Tuple[int, str]] = []
        for _, text in pages:
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            for i, sent in enumerate(sents):
                window = sent if i + 1 >= len(sents) else f"{sent} {sents[i + 1]}"
                w_low = window.lower()
                kw_hits = sum(1 for k in keywords if k in w_low)
                num_hits = sum(1 for n in norm_nums if n and n in w_low)
                score = kw_hits * 2 + num_hits
                if score > 0:
                    scored.append((score, " ".join(window.split())[:420]))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected: List[str] = []
        for _, win in scored:
            if all(win[:60] not in s and s[:60] not in win for s in selected):
                selected.append(win)
                if len(selected) >= top_k:
                    break
        return selected

    def _find_number_in_doc(
        self, doc_id: str, num_token: str, keywords: List[str]
    ) -> Optional[Tuple[int, str, int]]:
        """Returns (page_num, snippet, keyword_hits) if num_token appears in doc_id."""
        pages = self.corpus_pages.get(doc_id, [])
        pattern = _build_number_regex(num_token)
        if pattern is None:
            return None
        best: Optional[Tuple[int, str, int]] = None

        for page_num, text in pages:
            for match in pattern.finditer(text):
                start = max(0, match.start() - 160)
                end = min(len(text), match.end() + 160)
                snippet = " ".join(text[start:end].split())
                snip_low = snippet.lower()
                kw_hits = sum(1 for k in keywords if k in snip_low)
                if best is None or kw_hits > best[2]:
                    best = (page_num, snippet, kw_hits)
        return best

    def _check_polarity_conflict(self, claim_text: str, snippet: str) -> bool:
        """Detects exclusive directional contradiction between claim and evidence snippet."""
        c_low, s_low = claim_text.lower(), snippet.lower()
        for pos, neg in _ANTONYM_PAIRS:
            p_re = rf"\b{re.escape(pos)}\b"
            n_re = rf"\b{re.escape(neg)}\b"
            c_has_p = bool(re.search(p_re, c_low))
            c_has_n = bool(re.search(n_re, c_low))
            s_has_p = bool(re.search(p_re, s_low))
            s_has_n = bool(re.search(n_re, s_low))
            if (c_has_p and s_has_n and not s_has_p and not c_has_n) or (
                c_has_n and s_has_p and not s_has_n and not c_has_p
            ):
                return True
        return False

    def audit_claim(self, claim_text: str, cited_doc_id: str) -> NumericAuditResult:
        """Audits numeric tokens and directional polarity against cited_doc_id and full corpus."""
        nums = extract_numeric_tokens(claim_text)
        keywords = _extract_context_keywords(claim_text)

        if cited_doc_id not in self.corpus_pages:
            return NumericAuditResult(
                claim_text=claim_text,
                cited_doc_id=cited_doc_id,
                extracted_numbers=nums,
                status="DOC_NOT_INDEXED",
                matched_doc_id=None,
                matched_page=None,
                evidence_snippet="",
                best_evidence_sentence="",
                notes=f"Cited document '{cited_doc_id}' is not indexed in corpus (abstained).",
            )

        best_page, best_sent, best_hits = self._best_overlap_sentence(cited_doc_id, keywords)

        if not nums:
            # Qualitative directional check against cited document
            if best_sent and self._check_polarity_conflict(claim_text, best_sent):
                return NumericAuditResult(
                    claim_text=claim_text,
                    cited_doc_id=cited_doc_id,
                    extracted_numbers=[],
                    status="POLARITY_CONFLICT",
                    matched_doc_id=cited_doc_id,
                    matched_page=best_page,
                    evidence_snippet=best_sent,
                    best_evidence_sentence=best_sent,
                    notes="Qualitative directional claim contradicts highest-overlap sentence in cited PDF.",
                )
            return NumericAuditResult(
                claim_text=claim_text,
                cited_doc_id=cited_doc_id,
                extracted_numbers=[],
                status="NUMERIC_VERIFIED" if best_hits >= 2 else "NUMERIC_UNSUPPORTED",
                matched_doc_id=cited_doc_id if best_hits >= 2 else None,
                matched_page=best_page if best_hits >= 2 else None,
                evidence_snippet=best_sent,
                best_evidence_sentence=best_sent,
                notes="Qualitative check based on lexical overlap.",
            )

        # Check every extracted numeric token against cited_doc_id
        missing_nums: List[str] = []
        matched_snippets: List[Tuple[int, str, int]] = []
        for num in nums:
            hit = self._find_number_in_doc(cited_doc_id, num, keywords)
            if hit is None:
                missing_nums.append(num)
            else:
                matched_snippets.append(hit)
                if self._check_polarity_conflict(claim_text, hit[1]):
                    return NumericAuditResult(
                        claim_text=claim_text,
                        cited_doc_id=cited_doc_id,
                        extracted_numbers=nums,
                        status="POLARITY_CONFLICT",
                        matched_doc_id=cited_doc_id,
                        matched_page=hit[0],
                        evidence_snippet=hit[1],
                        best_evidence_sentence=best_sent or hit[1],
                        notes=f"Number '{num}' found on p.{hit[0]}, but directional verb contradicts source.",
                    )

        if not missing_nums and matched_snippets:
            # Pick the matched number snippet with highest keyword overlap
            best_num_hit = max(matched_snippets, key=lambda x: x[2])
            top_page, top_sent, _ = self._best_overlap_sentence(cited_doc_id, keywords, max_pages=4)
            check_sent = top_sent or best_sent
            if check_sent and self._check_polarity_conflict(claim_text, check_sent):
                return NumericAuditResult(
                    claim_text=claim_text,
                    cited_doc_id=cited_doc_id,
                    extracted_numbers=nums,
                    status="POLARITY_CONFLICT",
                    matched_doc_id=cited_doc_id,
                    matched_page=top_page or best_num_hit[0],
                    evidence_snippet=check_sent,
                    best_evidence_sentence=check_sent,
                    notes=f"Numbers matched, but directional claim contradicts source text on p.{top_page or best_num_hit[0]}.",
                )
            evidence_parts: List[str] = self._top_overlap_windows(cited_doc_id, keywords, nums, top_k=5)
            if best_sent and all(best_sent[:60] not in ep for ep in evidence_parts):
                evidence_parts.insert(0, best_sent)
            for _, snip, _ in matched_snippets:
                if snip and all(snip[:60] not in ep for ep in evidence_parts):
                    evidence_parts.append(snip)
            combined_evidence = " ... ".join(evidence_parts)[:2200]
            return NumericAuditResult(
                claim_text=claim_text,
                cited_doc_id=cited_doc_id,
                extracted_numbers=nums,
                status="NUMERIC_VERIFIED",
                matched_doc_id=cited_doc_id,
                matched_page=best_num_hit[0],
                evidence_snippet=best_num_hit[1],
                best_evidence_sentence=combined_evidence,
                notes="All numeric tokens verified in cited PDF.",
            )

        # Cross-Artifact Misattribution check across OTHER indexed PDFs
        best_other: Optional[Tuple[str, str, int, str, int]] = None
        for m_num in missing_nums:
            norm_m = _normalize_num(m_num).split("±")[0].strip()
            is_distinctive = ("." in norm_m) or (norm_m.isdigit() and len(norm_m) >= 4)
            min_kw = 0 if is_distinctive else 2
            for other_id in self.corpus_pages:
                if other_id == cited_doc_id:
                    continue
                other_hit = self._find_number_in_doc(other_id, m_num, keywords)
                if other_hit is not None and other_hit[2] >= min_kw:
                    cand = (m_num, other_id, other_hit[0], other_hit[1], other_hit[2])
                    if best_other is None or cand[4] > best_other[4]:
                        best_other = cand

        if best_other is not None:
            m_num, other_id, other_page, other_snip, _ = best_other
            return NumericAuditResult(
                claim_text=claim_text,
                cited_doc_id=cited_doc_id,
                extracted_numbers=nums,
                status="CROSS_ARTIFACT_MISATTRIBUTED",
                matched_doc_id=other_id,
                matched_page=other_page,
                evidence_snippet=other_snip,
                best_evidence_sentence=best_sent,
                notes=(
                    f"Metric '{m_num}' missing from '{cited_doc_id}', "
                    f"but found in '{other_id}' (p.{other_page})."
                ),
            )

        return NumericAuditResult(
            claim_text=claim_text,
            cited_doc_id=cited_doc_id,
            extracted_numbers=nums,
            status="NUMERIC_UNSUPPORTED",
            matched_doc_id=None,
            matched_page=None,
            evidence_snippet="",
            best_evidence_sentence=best_sent,
            notes=f"Missing numeric token(s) {missing_nums} in cited PDF and corpus.",
        )
