"""
validate_kb.py
==============
Standalone validation script for knowledge_base.json.

Run this before any embedding work to confirm the knowledge base is
well-formed.  Also prints a semantic diversity preview once the
embedding engine is available (Week 2).

Usage
-----
    python backend/validate_kb.py

Exits with code 0 on success, 1 on any validation failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────

KB_PATH = Path(__file__).parent / "knowledge_base.json"

REQUIRED_FIELDS = {"id", "category", "text", "source", "intervention_type", "semantic_tags"}
VALID_INTERVENTION_TYPES = {"psychoeducation", "substitution", "mindfulness", "behavioural"}
MIN_TEXT_LENGTH = 40   # characters — too short = not useful for RAG
MAX_TEXT_LENGTH = 600  # characters — too long = embedding quality degrades


# ── Validation ───────────────────────────────────────────────────────────────

def validate_knowledge_base(path: Path) -> bool:
    """
    Load and validate the knowledge base JSON.

    Checks
    ------
    1. File exists and is valid JSON.
    2. Top-level structure is a list.
    3. Each entry has all required fields.
    4. IDs are unique.
    5. Text lengths are within bounds.
    6. intervention_type values are from the allowed set.
    7. semantic_tags is a non-empty list of strings.

    Returns
    -------
    bool
        True if all checks pass, False otherwise (also prints failures).
    """
    errors: list[str] = []

    # ── Check 1: file exists ─────────────────────────────────────────────────
    if not path.exists():
        print(f"[FAIL] File not found: {path}")
        return False

    # ── Check 2: valid JSON ──────────────────────────────────────────────────
    try:
        with open(path, encoding="utf-8") as f:
            data: Any = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"[FAIL] Invalid JSON: {exc}")
        return False

    # ── Check 3: top-level is a list ─────────────────────────────────────────
    if not isinstance(data, list):
        print(f"[FAIL] Top-level structure must be a JSON array, got {type(data).__name__}")
        return False

    if len(data) == 0:
        print("[FAIL] Knowledge base is empty.")
        return False

    seen_ids: set[str] = set()

    for i, entry in enumerate(data):
        prefix = f"  Entry {i}"

        # ── Check 4: required fields ─────────────────────────────────────────
        missing = REQUIRED_FIELDS - set(entry.keys())
        if missing:
            errors.append(f"{prefix} — missing fields: {missing}")
            continue  # can't validate further without all fields

        entry_id: str = entry["id"]

        # ── Check 5: unique IDs ───────────────────────────────────────────────
        if entry_id in seen_ids:
            errors.append(f"{prefix} — duplicate id: '{entry_id}'")
        seen_ids.add(entry_id)

        # ── Check 6: text length ──────────────────────────────────────────────
        text_len = len(entry["text"])
        if text_len < MIN_TEXT_LENGTH:
            errors.append(
                f"{prefix} ({entry_id}) — text too short ({text_len} chars, min={MIN_TEXT_LENGTH})"
            )
        if text_len > MAX_TEXT_LENGTH:
            errors.append(
                f"{prefix} ({entry_id}) — text too long ({text_len} chars, max={MAX_TEXT_LENGTH})"
            )

        # ── Check 7: intervention_type ────────────────────────────────────────
        if entry["intervention_type"] not in VALID_INTERVENTION_TYPES:
            errors.append(
                f"{prefix} ({entry_id}) — unknown intervention_type: "
                f"'{entry['intervention_type']}'. "
                f"Valid: {VALID_INTERVENTION_TYPES}"
            )

        # ── Check 8: semantic_tags ────────────────────────────────────────────
        tags = entry.get("semantic_tags", [])
        if not isinstance(tags, list) or len(tags) == 0:
            errors.append(f"{prefix} ({entry_id}) — semantic_tags must be a non-empty list")
        elif not all(isinstance(t, str) for t in tags):
            errors.append(f"{prefix} ({entry_id}) — all semantic_tags must be strings")

    # ── Report ────────────────────────────────────────────────────────────────
    if errors:
        print(f"\n[FAIL] Knowledge base validation failed with {len(errors)} error(s):\n")
        for err in errors:
            print(f"  ✗  {err}")
        print()
        return False

    print(f"\n[PASS] Knowledge base validated successfully.")
    print(f"  ✓  {len(data)} entries found")
    print(f"  ✓  All IDs unique: {sorted(seen_ids)}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    from collections import Counter
    categories = Counter(e["category"] for e in data)
    intervention_types = Counter(e["intervention_type"] for e in data)
    text_lengths = [len(e["text"]) for e in data]

    print(f"\n  Categories:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count}")

    print(f"\n  Intervention types:")
    for itype, count in sorted(intervention_types.items()):
        print(f"    {itype}: {count}")

    print(f"\n  Text length (chars):")
    print(f"    min={min(text_lengths)}, max={max(text_lengths)}, "
          f"mean={sum(text_lengths)/len(text_lengths):.1f}")
    print()
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    success = validate_knowledge_base(KB_PATH)
    sys.exit(0 if success else 1)