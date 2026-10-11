"""KryptosID Forensics Package.

Includes:
- hashing.py: Canonical evidence chain hashing.
- signing.py: Ed25519 digital signatures and tamper verification.
- report.py: ReportLab A4 forensic PDF report generator.
- store.py: Forensic report database persistence.
"""

from forensics.hashing import hash_raw_bytes, canonical_json_hash
from forensics.signing import (
    generate_keypair,
    sign_payload,
    verify_payload,
    Ed25519Signer,
)
from forensics.report import generate_forensic_report_pdf
from forensics.store import ForensicReportStore

__all__ = [
    "hash_raw_bytes",
    "canonical_json_hash",
    "generate_keypair",
    "sign_payload",
    "verify_payload",
    "Ed25519Signer",
    "generate_forensic_report_pdf",
    "ForensicReportStore",
]
