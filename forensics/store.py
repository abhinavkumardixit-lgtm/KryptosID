"""Forensic Report Database Persistence and Verification Store.

Implements Section 9.4 of DESIGN.md:
- SQLite store for development and forensic audit trails.
- Preserves signed canonical evidence, report SHA-256, PDF paths, and status.
- Re-verifies Ed25519 signatures and evidence hashes on demand.
"""

from typing import Dict, List, Optional, Any
import json
import os
import sqlite3
import time

import contextlib
from forensics.signing import verify_payload, SignatureVerificationError
from forensics.hashing import canonical_json_hash



_DEFAULT_STORE: Optional["ForensicReportStore"] = None


def get_report_store(db_path: str = "kryptosid_reports.db") -> "ForensicReportStore":
    """Singleton getter for ForensicReportStore."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None or _DEFAULT_STORE.db_path != db_path:
        _DEFAULT_STORE = ForensicReportStore(db_path=db_path)
    return _DEFAULT_STORE


class ForensicReportStore:
    """SQLite-backed forensic report store."""

    def __init__(self, db_path: str = "kryptosid_reports.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True) if os.path.dirname(db_path) else None
        self._init_db()

    @contextlib.contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()


    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    report_id TEXT PRIMARY KEY,
                    input_sha256 TEXT NOT NULL,
                    report_sha256 TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    rule_fired TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    pdf_path TEXT,
                    signed_json TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_input_sha ON reports(input_sha256)")
            conn.commit()

    def save_report(
        self,
        report_id: str,
        signed_evidence: Dict[str, Any],
        pdf_path: Optional[str] = None,
    ) -> str:
        """Persist signed report to database."""
        verdict = signed_evidence.get("verdict", {})
        op = signed_evidence.get("operating_point", {})

        signed_json = json.dumps(signed_evidence, sort_keys=True)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reports (
                    report_id, input_sha256, report_sha256, decision,
                    score, threshold, rule_fired, created_at, pdf_path, signed_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    signed_evidence.get("input_sha256", ""),
                    signed_evidence.get("report_sha256", ""),
                    verdict.get("decision", "inconclusive"),
                    float(verdict.get("calibrated_score", 0.0)),
                    float(op.get("decision_threshold", 0.5)),
                    verdict.get("rule_fired", "default"),
                    signed_evidence.get("timestamps", {}).get("finished_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                    pdf_path,
                    signed_json,
                ),
            )
            conn.commit()

        return report_id

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Fetch report record by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM reports WHERE report_id = ?", (report_id,)).fetchone()
            if not row:
                return None
            return {
                "report_id": row["report_id"],
                "input_sha256": row["input_sha256"],
                "report_sha256": row["report_sha256"],
                "decision": row["decision"],
                "score": row["score"],
                "threshold": row["threshold"],
                "rule_fired": row["rule_fired"],
                "created_at": row["created_at"],
                "pdf_path": row["pdf_path"],
                "signed_evidence": json.loads(row["signed_json"]),
            }

    def verify_report(self, report_id: str) -> Dict[str, Any]:
        """Verify report cryptographic signature and evidence hash integrity."""
        rec = self.get_report(report_id)
        if not rec:
            return {"valid": False, "error": f"Report not found: {report_id}"}

        signed_evidence = rec["signed_evidence"]
        expected_report_sha = rec["report_sha256"]

        # 1. Recompute canonical hash of unsigned payload
        unsigned = {k: v for k, v in signed_evidence.items() if k != "signature_block"}
        # If unsigned had 'report_sha256' set, check it
        stored_hash = unsigned.get("report_sha256")
        if stored_hash != expected_report_sha:
            return {"valid": False, "error": "Report SHA-256 header mismatch"}

        # 2. Verify Ed25519 signature
        try:
            verify_payload(signed_evidence)
            return {"valid": True, "report_id": report_id, "report_sha256": expected_report_sha}
        except SignatureVerificationError as e:
            return {"valid": False, "error": str(e)}

    def list_reports(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent reports."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT report_id, decision, score, threshold, created_at, pdf_path FROM reports ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
