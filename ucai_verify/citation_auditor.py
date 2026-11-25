"""Deterministic cascading bibliographic & identifier verifier (v0.2.0).

Queries authoritative open scholarly APIs (arXiv, Crossref, OpenAlex)
without requiring paid LLM API keys, detecting ghost citations, metadata drift,
and identifier hijacking while supporting version-aware preprint title resolution.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional
import requests


_TITLE_STOPWORDS = {"a", "an", "the", "of", "in", "on", "for", "to", "with", "and", "by"}


def _normalize_text(text: str) -> str:
    """Lowercases and strips non-alphanumeric characters for robust matching."""
    return re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower()).strip()


def _title_similarity(a: str, b: str) -> float:
    """Computes max(SequenceRatio, TokenJaccard) between two normalized titles."""
    na, nb = _normalize_text(a), _normalize_text(b)
    if not na or not nb:
        return 0.0
    seq_ratio = SequenceMatcher(None, na, nb).ratio()
    tok_a, tok_b = set(na.split()), set(nb.split())
    union = tok_a | tok_b
    jaccard = (len(tok_a & tok_b) / len(union)) if union else 0.0
    return max(seq_ratio, jaccard)


def _title_containment(claimed: str, matched: str) -> float:
    """Fraction of content tokens in claimed title present in matched title."""
    tok_c = [t for t in _normalize_text(claimed).split() if t not in _TITLE_STOPWORDS]
    tok_m = set(_normalize_text(matched).split())
    if not tok_c:
        return 0.0
    return sum(1 for t in tok_c if t in tok_m) / len(tok_c)


def _extract_surname(author_str: str) -> str:
    """Extracts probable surname token from an author string."""
    cleaned = re.sub(r"[^a-zA-Z\s-]", " ", author_str or "").strip().lower()
    parts = [p for p in cleaned.split() if len(p) > 1]
    return parts[-1] if parts else ""


@dataclass
class CitationAuditResult:
    """Structured audit output for a single bibliographic reference."""

    raw_title: str
    claimed_author: str
    claimed_year: Optional[int]
    claimed_doi: Optional[str]
    claimed_arxiv_id: Optional[str]
    status: str  # VERIFIED_EXACT | METADATA_DRIFT | IDENTIFIER_HIJACKED | SUSPECTED_FAKE
    confidence: float
    matched_source: str
    matched_title: str
    matched_year: Optional[int]
    title_similarity: float
    author_overlap: float
    latency_sec: float
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CitationAuditor:
    """Cascading deterministic bibliographic verifier."""

    def __init__(
        self,
        timeout_sec: int = 12,
        title_threshold: float = 0.78,
        polite_email: str = "ucai-research@example.org",
    ) -> None:
        self.timeout_sec = timeout_sec
        self.title_threshold = title_threshold
        self.headers = {"User-Agent": f"ucai-verify/0.2.0 (mailto:{polite_email})"}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _query_arxiv_by_id(
        self, arxiv_id: str, keep_version: bool = False
    ) -> Optional[Dict[str, Any]]:
        raw_id = arxiv_id.strip()
        clean_id = raw_id if keep_version else re.sub(r"v\d+$", "", raw_id)
        url = f"https://export.arxiv.org/api/query?id_list={clean_id}"
        try:
            resp = self.session.get(url, timeout=self.timeout_sec)
            if resp.status_code != 200:
                return None
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entry = root.find("atom:entry", ns)
            if entry is None:
                return None
            title_el = entry.find("atom:title", ns)
            pub_el = entry.find("atom:published", ns)
            title = " ".join((title_el.text or "").split()) if title_el is not None else ""
            if not title or title.lower() == "error":
                return None
            authors = [
                (a.find("atom:name", ns).text or "").strip()
                for a in entry.findall("atom:author", ns)
                if a.find("atom:name", ns) is not None
            ]
            year = int(pub_el.text[:4]) if pub_el is not None and pub_el.text else None
            src_label = f"arXiv({clean_id})" if keep_version else "arXiv"
            return {"source": src_label, "title": title, "authors": authors, "year": year}
        except Exception:
            return None

    def _query_crossref_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        clean_doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi.strip())
        url = f"https://api.crossref.org/works/{clean_doi}"
        try:
            resp = self.session.get(url, timeout=self.timeout_sec)
            if resp.status_code != 200:
                return None
            msg = resp.json().get("message", {})
            titles = msg.get("title") or []
            title = titles[0] if titles else ""
            authors = [
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in msg.get("author", [])
            ]
            issued = msg.get("issued", {}).get("date-parts", [[None]])
            year = issued[0][0] if issued and issued[0] else None
            return {"source": "Crossref-DOI", "title": title, "authors": authors, "year": year}
        except Exception:
            return None

    def _query_crossref_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        url = "https://api.crossref.org/works"
        try:
            resp = self.session.get(
                url, params={"query.title": title, "rows": 3}, timeout=self.timeout_sec
            )
            if resp.status_code != 200:
                return None
            items = resp.json().get("message", {}).get("items", [])
            best, best_sim = None, 0.0
            for item in items:
                cand_titles = item.get("title") or []
                cand_title = cand_titles[0] if cand_titles else ""
                sim = _title_similarity(title, cand_title)
                if sim > best_sim:
                    best_sim = sim
                    authors = [
                        f"{a.get('given', '')} {a.get('family', '')}".strip()
                        for a in item.get("author", [])
                    ]
                    issued = item.get("issued", {}).get("date-parts", [[None]])
                    year = issued[0][0] if issued and issued[0] else None
                    best = {
                        "source": "Crossref-Search",
                        "title": cand_title,
                        "authors": authors,
                        "year": year,
                    }
            return best
        except Exception:
            return None

    def _query_openalex_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        url = "https://api.openalex.org/works"
        try:
            resp = self.session.get(
                url, params={"search": title, "per-page": 3}, timeout=self.timeout_sec
            )
            if resp.status_code != 200:
                return None
            results = resp.json().get("results", [])
            best, best_sim = None, 0.0
            for item in results:
                cand_title = item.get("title") or ""
                sim = _title_similarity(title, cand_title)
                if sim > best_sim:
                    best_sim = sim
                    authors = [
                        (auth.get("author", {}).get("display_name") or "").strip()
                        for auth in item.get("authorships", [])
                    ]
                    best = {
                        "source": "OpenAlex",
                        "title": cand_title,
                        "authors": authors,
                        "year": item.get("publication_year"),
                    }
            return best
        except Exception:
            return None

    def _compute_author_overlap(self, claimed_author: str, real_authors: List[str]) -> float:
        if not claimed_author or not real_authors:
            return 0.0
        claimed_tokens = [
            _extract_surname(a)
            for a in re.split(r",|;| and |&", claimed_author)
            if _extract_surname(a)
        ]
        if not claimed_tokens:
            return 0.0
        real_surnames = {_extract_surname(a) for a in real_authors if _extract_surname(a)}
        hits = sum(1 for t in claimed_tokens if t in real_surnames)
        return hits / len(claimed_tokens)

    def verify_reference(
        self,
        title: str,
        author: str = "",
        year: Optional[int] = None,
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
    ) -> CitationAuditResult:
        """Verifies a single reference via cascading deterministic checks."""
        t0 = time.perf_counter()
        notes: List[str] = []

        # Step 1: Deterministic ID checks (detect identifier hijacking or dead IDs)
        id_record = None
        if arxiv_id:
            id_record = self._query_arxiv_by_id(arxiv_id, keep_version=False)
            if id_record is None:
                notes.append(f"arXiv ID '{arxiv_id}' not found")
        elif doi:
            id_record = self._query_crossref_by_doi(doi)
            if id_record is None:
                notes.append(f"DOI '{doi}' not found in Crossref")

        if id_record:
            sim = _title_similarity(title, id_record["title"])
            containment = _title_containment(title, id_record["title"])
            auth_ov = self._compute_author_overlap(author, id_record["authors"])
            year_ok = (year is None) or (id_record["year"] is None) or (abs(year - id_record["year"]) <= 1)

            # Version-aware preprint title check: if authors & year match strongly on arXiv,
            # check v1 in case the authors revised the preprint title in v2+ (e.g. 2502.18864).
            if arxiv_id and sim < self.title_threshold and auth_ov >= 0.60 and year_ok:
                base_id = re.sub(r"v\d+$", "", arxiv_id.strip())
                v1_rec = self._query_arxiv_by_id(f"{base_id}v1", keep_version=True)
                if v1_rec:
                    v1_sim = _title_similarity(title, v1_rec["title"])
                    if v1_sim >= self.title_threshold:
                        id_record = v1_rec
                        sim = v1_sim
                        notes.append("Matched v1 preprint title prior to v2+ title revision")
                # Also handle abbreviated/revised preprint titles where same author team & core terms match
                if sim < self.title_threshold and auth_ov >= 0.80 and containment >= 0.50:
                    return CitationAuditResult(
                        raw_title=title,
                        claimed_author=author,
                        claimed_year=year,
                        claimed_doi=doi,
                        claimed_arxiv_id=arxiv_id,
                        status="VERIFIED_EXACT",
                        confidence=0.95,
                        matched_source=id_record["source"],
                        matched_title=id_record["title"],
                        matched_year=id_record["year"],
                        title_similarity=round(sim, 3),
                        author_overlap=round(auth_ov, 3),
                        latency_sec=round(time.perf_counter() - t0, 3),
                        notes="Verified exact ID, author team, and core title tokens across preprint revision.",
                    )

            if sim < 0.55 and auth_ov < 0.50:
                return CitationAuditResult(
                    raw_title=title,
                    claimed_author=author,
                    claimed_year=year,
                    claimed_doi=doi,
                    claimed_arxiv_id=arxiv_id,
                    status="IDENTIFIER_HIJACKED",
                    confidence=0.96,
                    matched_source=id_record["source"],
                    matched_title=id_record["title"],
                    matched_year=id_record["year"],
                    title_similarity=round(sim, 3),
                    author_overlap=round(auth_ov, 3),
                    latency_sec=round(time.perf_counter() - t0, 3),
                    notes="ID resolves to a DIFFERENT paper and author team (Identifier Hijacking).",
                )
            if (sim >= self.title_threshold or (containment >= 0.80 and auth_ov >= 0.60)) and (not author or auth_ov >= 0.5) and year_ok:
                return CitationAuditResult(
                    raw_title=title,
                    claimed_author=author,
                    claimed_year=year,
                    claimed_doi=doi,
                    claimed_arxiv_id=arxiv_id,
                    status="VERIFIED_EXACT",
                    confidence=0.98,
                    matched_source=id_record["source"],
                    matched_title=id_record["title"],
                    matched_year=id_record["year"],
                    title_similarity=round(sim, 3),
                    author_overlap=round(auth_ov, 3),
                    latency_sec=round(time.perf_counter() - t0, 3),
                    notes="; ".join(notes) if notes else "Exact identifier, title, author, and year match.",
                )
            return CitationAuditResult(
                raw_title=title,
                claimed_author=author,
                claimed_year=year,
                claimed_doi=doi,
                claimed_arxiv_id=arxiv_id,
                status="METADATA_DRIFT",
                confidence=0.85,
                matched_source=id_record["source"],
                matched_title=id_record["title"],
                matched_year=id_record["year"],
                title_similarity=round(sim, 3),
                author_overlap=round(auth_ov, 3),
                latency_sec=round(time.perf_counter() - t0, 3),
                notes="ID exists, but title, author, or publication year drifts from primary record.",
            )

        # Step 2: Title search cascade (Crossref -> OpenAlex)
        best_cand = None
        for finder in (self._query_crossref_by_title, self._query_openalex_by_title):
            cand = finder(title)
            if cand and _title_similarity(title, cand["title"]) >= self.title_threshold:
                best_cand = cand
                break

        if best_cand:
            sim = _title_similarity(title, best_cand["title"])
            auth_ov = self._compute_author_overlap(author, best_cand["authors"])
            year_ok = (year is None) or (best_cand["year"] is None) or (abs(year - best_cand["year"]) <= 1)

            # Even if title matches, dead/fake DOI or mismatched author flags drift/fabrication
            if (doi or arxiv_id) or (author and auth_ov < 0.35) or not year_ok:
                notes.append("Title found, but supplied DOI/arXiv ID failed or author/year mismatched.")
                status = "METADATA_DRIFT"
                conf = 0.80
            else:
                status = "VERIFIED_EXACT"
                conf = 0.92
            return CitationAuditResult(
                raw_title=title,
                claimed_author=author,
                claimed_year=year,
                claimed_doi=doi,
                claimed_arxiv_id=arxiv_id,
                status=status,
                confidence=conf,
                matched_source=best_cand["source"],
                matched_title=best_cand["title"],
                matched_year=best_cand["year"],
                title_similarity=round(sim, 3),
                author_overlap=round(auth_ov, 3),
                latency_sec=round(time.perf_counter() - t0, 3),
                notes="; ".join(notes) if notes else "Verified via title cascade.",
            )

        return CitationAuditResult(
            raw_title=title,
            claimed_author=author,
            claimed_year=year,
            claimed_doi=doi,
            claimed_arxiv_id=arxiv_id,
            status="SUSPECTED_FAKE",
            confidence=0.90,
            matched_source="NONE",
            matched_title="",
            matched_year=None,
            title_similarity=0.0,
            author_overlap=0.0,
            latency_sec=round(time.perf_counter() - t0, 3),
            notes="; ".join(notes) if notes else "Not found across arXiv, Crossref, or OpenAlex.",
        )
