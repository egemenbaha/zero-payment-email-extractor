"""Compare extracted amount/currency coverage with the supplied training source."""

import argparse
from decimal import Decimal
import json
from pathlib import Path
import re

import pandas as pd


def audit(source_path: Path, result_path: Path) -> dict:
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    table = pd.read_csv(result_path, dtype=str, keep_default_na=False)
    mismatches = []
    change_candidates = []
    change_misses = []
    amount_pattern = re.compile(r"\b([A-Z]{3})\s+([0-9,]+\.[0-9]{2})(?!\d)")
    change_pattern = re.compile(
        r"new (?:account|IBAN)|old (?:account|IBAN)|(?:previous|former|prior) (?:account|IBAN)|"
        r"\breplace(?:s|d)?\b|instead of|revised account|banking update|updated account", re.I)
    for email in source.to_dict(orient="records"):
        rows = table[(table.email_id == email["email_id"]) & (table.payment_index != "0")]
        text = email["subject"] + "\n" + email["email_body"]
        expected = {(currency, Decimal(amount.replace(",", "")))
                    for currency, amount in amount_pattern.findall(email["email_body"])}
        actual = {(row.currency, Decimal(row.amount)) for row in rows.itertuples()
                  if row.currency and row.amount}
        if expected != actual:
            mismatches.append(email["email_id"])
        if change_pattern.search(text):
            change_candidates.append(email["email_id"])
            if not rows.account_change_requested.str.lower().eq("true").any():
                change_misses.append(email["email_id"])
    return {
        "source_emails": len(source), "represented_emails": int(table.email_id.nunique()),
        "missing_source_ids": sorted(set(source.email_id) - set(table.email_id)),
        "unexpected_source_ids": sorted(set(table.email_id) - set(source.email_id)),
        "duplicate_payment_keys": int(table.duplicated(["email_id", "payment_index"]).sum()),
        "amount_currency_coverage_mismatches": mismatches,
        "account_change_keyword_candidates": change_candidates,
        "account_change_candidates_not_flagged": change_misses,
        "review_rows": int(table.extraction_status.ne("extracted").sum()),
        "limitations": "Independent coverage checks, not proof of all semantic mappings. "
                       "Keyword matching may miss unfamiliar account-change wording. "
                       "Identical repeated amounts are deduplicated for this audit only.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(audit(args.source, args.results), indent=2)
    print(report)
    if args.output:
        args.output.write_text(report + "\n")
