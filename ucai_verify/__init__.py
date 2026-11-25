"""UCAI-Verify: Deterministic Citation & Cross-Document Numeric Audit Engine."""

__version__ = "0.2.0"
__license__ = "MIT"

from .citation_auditor import CitationAuditor, CitationAuditResult
from .hybrid_auditor import HybridCascadeAuditor
from .numeric_auditor import NumericAuditor, NumericAuditResult

__all__ = [
    "CitationAuditor",
    "CitationAuditResult",
    "HybridCascadeAuditor",
    "NumericAuditor",
    "NumericAuditResult",
]
