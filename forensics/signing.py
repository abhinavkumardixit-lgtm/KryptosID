"""Ed25519 Cryptographic Signing and Tamper Verification.

Implements Section 9.2 of DESIGN.md:
- Signs canonical evidence chain using asymmetric Ed25519 keys.
- Verifies integrity: Any alteration to evidence data or metadata invalidates signature.
- Handles public key identification and key rotation metadata.
"""

from typing import Dict, Optional, Tuple, Any
import hashlib
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

from forensics.hashing import canonical_json_dumps


class SignatureVerificationError(Exception):
    """Raised when an Ed25519 signature fails cryptographic verification."""
    pass


class Ed25519Signer:
    """Ed25519 asymmetric signer and verifier for forensic reports."""

    def __init__(
        self,
        private_key: Optional[ed25519.Ed25519PrivateKey] = None,
        key_id: str = "kryptosid-root-ed25519-v1",
    ) -> None:
        if private_key is None:
            self.private_key = ed25519.Ed25519PrivateKey.generate()
        else:
            self.private_key = private_key

        self.public_key = self.private_key.public_key()
        self.key_id = key_id

    @classmethod
    def from_seed(cls, seed_bytes: bytes, key_id: str = "kryptosid-v1") -> "Ed25519Signer":
        """Deterministic signer from 32-byte seed."""
        if len(seed_bytes) != 32:
            seed_bytes = hashlib.sha256(seed_bytes).digest()
        pk = ed25519.Ed25519PrivateKey.from_private_bytes(seed_bytes)
        return cls(private_key=pk, key_id=key_id)

    @property
    def public_key_hex(self) -> str:
        """Raw 32-byte public key as hex string."""
        from cryptography.hazmat.primitives import serialization
        raw = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return raw.hex()

    def sign(self, payload: Dict[str, Any]) -> Tuple[str, str, str]:
        """Sign canonical JSON payload.
        
        Returns:
            Tuple of (signature_hex, public_key_hex, key_id).
        """
        # Canonical stringification
        canonical_str = canonical_json_dumps(payload)
        signature_bytes = self.private_key.sign(canonical_str.encode("utf-8"))
        return signature_bytes.hex(), self.public_key_hex, self.key_id


def generate_keypair(key_id: str = "kryptosid-v1") -> Ed25519Signer:
    """Generate a new Ed25519 keypair."""
    return Ed25519Signer(key_id=key_id)


def sign_payload(
    signer: Ed25519Signer,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Sign payload dictionary and embed signature block."""
    sig_hex, pub_hex, key_id = signer.sign(payload)
    signed_record = dict(payload)
    signed_record["signature_block"] = {
        "algorithm": "Ed25519",
        "key_id": key_id,
        "public_key": pub_hex,
        "signature": sig_hex,
    }
    return signed_record


def verify_payload(
    signed_record: Dict[str, Any],
    public_key_hex: Optional[str] = None,
) -> bool:
    """Cryptographically verify an Ed25519 signed record.
    
    Args:
        signed_record: Dictionary containing the payload and 'signature_block'.
        public_key_hex: Optional public key override (otherwise reads from signature_block).
        
    Returns:
        True if signature is valid.
        
    Raises:
        SignatureVerificationError: If verification fails or record is tampered.
    """
    if "signature_block" not in signed_record:
        raise SignatureVerificationError("Missing 'signature_block' in record")

    sig_block = signed_record["signature_block"]
    sig_hex = sig_block["signature"]
    pub_hex = public_key_hex or sig_block["public_key"]

    # Reconstruct original unsigned payload (excluding signature_block)
    unsigned_payload = {k: v for k, v in signed_record.items() if k != "signature_block"}
    canonical_str = canonical_json_dumps(unsigned_payload)

    try:
        pub_bytes = bytes.fromhex(pub_hex)
        sig_bytes = bytes.fromhex(sig_hex)
        pk = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
        pk.verify(sig_bytes, canonical_str.encode("utf-8"))
        return True
    except (InvalidSignature, ValueError) as e:
        raise SignatureVerificationError(f"Ed25519 signature verification failed: {e}")
