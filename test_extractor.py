"""Offline tests: schema, coverage and edge cases, not a claim of live LLM accuracy."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from extractor import (BatchResult, EmailResult, Payment, SOURCE_COLUMNS, load_source,
                       apply_corrections, request_batch, table_rows, validate_batch, write_results)


def source(email_id="EM-00008", body=None):
    return dict(zip(SOURCE_COLUMNS, [email_id, "2026-09-02T09:02:00", "CL-TEST", "Training client",
        "training@example.test", "Invoice 8841", body or
        "Pay Lynton Paper Mills GBP 24180.40 on 2026-09-04 to account 7731 2208 sort 20-45-11.", "true"]))


def payment(**updates):
    data = dict(payee="Lynton Paper Mills", amount="24180.40", currency="GBP",
                payment_date="2026-09-04", requested_bank_account="7731 2208 sort 20-45-11",
                previous_bank_account=None, reference="8841", account_change_requested=False,
                account_change_evidence=None)
    return Payment(**(data | updates))


class ExtractorTests(unittest.TestCase):
    def test_reviewed_correction_keeps_audit_trail_and_rejects_drift(self):
        result = EmailResult(email_id="EM-00008", payments=[payment()], issues=[])
        rows = table_rows(result, source(), "offline-test")
        correction = {"email_id": "EM-00008", "payment_index": 1, "field": "payment_date",
                      "expected": "2026-09-04", "value": None, "reason": "Test correction"}
        apply_corrections(rows, [correction])
        self.assertIsNone(rows[0]["payment_date"])
        self.assertEqual(rows[0]["extraction_status"], "needs_review")
        self.assertIn("2026-09-04", rows[0]["source_review_corrections"])
        with self.assertRaises(ValueError):
            apply_corrections(rows, [correction])

    def test_missing_reference_uses_single_explicit_source_label(self):
        result = EmailResult(email_id="EM-00008", payments=[payment(reference=None)], issues=[])
        row = table_rows(result, source(), "offline-test")[0]
        self.assertEqual(row["reference"], "8841")
        self.assertEqual(row["reference_source"], "explicit_source_label")

    def test_conflicting_references_are_not_guessed(self):
        email = source(body=source()["email_body"] + " Ref OTHER-22.")
        result = EmailResult(email_id="EM-00008", payments=[payment(reference=None)], issues=[])
        row = table_rows(result, email, "offline-test")[0]
        self.assertIsNone(row["reference"])
        self.assertEqual(row["reference_source"], "missing")

    def test_success_message_is_not_an_issue(self):
        result = EmailResult(email_id="EM-00008", payments=[payment()], issues=["No issues"])
        self.assertEqual(result.issues, [])
        self.assertEqual(table_rows(result, source(), "offline-test")[0]["extraction_status"], "extracted")

    def test_preserves_compound_bank_account(self):
        result = EmailResult(email_id="EM-00008", payments=[payment()], issues=[])
        row = table_rows(result, source(), "offline-test")[0]
        self.assertEqual(row["requested_bank_account"], "7731 2208 sort 20-45-11")
        self.assertEqual(row["extraction_status"], "extracted")

    def test_two_payments_keep_source_and_separate_index(self):
        body = "Pay Alpha USD 100.00 to 001-002 on 2026-11-20. Also pay Beta USD 20.00 to 003-004."
        first = payment(payee="Alpha", amount="100.00", currency="USD", payment_date="2026-11-20",
                        requested_bank_account="001-002", reference=None)
        second = payment(payee="Beta", amount="20.00", currency="USD", payment_date=None,
                         requested_bank_account="003-004", reference=None)
        result = EmailResult(email_id="EM-00021", payments=[first, second], issues=[])
        rows = table_rows(result, source("EM-00021", body), "offline-test")
        self.assertEqual([r["payment_index"] for r in rows], [1, 2])
        self.assertEqual({r["email_id"] for r in rows}, {"EM-00021"})
        self.assertIsNone(rows[1]["payment_date"])
        self.assertEqual(rows[1]["extraction_status"], "needs_review")

    def test_old_and_new_account_evidence_survive(self):
        body = "Pay Alpha USD 100.00 on 2026-11-20 to new account 001-002. Old account 003-004 is closed."
        item = payment(payee="Alpha", amount="100.00", currency="USD", payment_date="2026-11-20",
                       requested_bank_account="001-002", previous_bank_account="003-004",
                       reference=None, account_change_requested=True,
                       account_change_evidence="Old account 003-004 is closed.")
        row = table_rows(EmailResult(email_id="EM-00012", payments=[item], issues=[]),
                         source("EM-00012", body), "offline-test")[0]
        self.assertEqual(row["previous_bank_account"], "003-004")
        self.assertTrue(row["account_change_requested"])
        self.assertEqual(row["extraction_status"], "extracted")

    def test_unsupported_details_are_flagged(self):
        row = table_rows(EmailResult(email_id="EM-00008", payments=[payment(amount="999.00",
            requested_bank_account="invented")], issues=[]), source(), "offline-test")[0]
        self.assertEqual(row["extraction_status"], "needs_review")
        self.assertIn("not supported", row["extraction_issues"])

    def test_repeated_forwarded_payment_is_flagged(self):
        result = EmailResult(email_id="EM-00008", payments=[payment(), payment()], issues=[])
        self.assertIn("duplicate", table_rows(result, source(), "offline-test")[1]["extraction_issues"])

    def test_missing_and_duplicate_source_ids_in_model_response_fail(self):
        raw = {"emails": [{"email_id": "EM-00008", "payments": [], "issues": []}]}
        with self.assertRaises(ValueError):
            validate_batch(raw, [source(), source("EM-00009")])
        raw["emails"] *= 2
        with self.assertRaises(ValueError):
            validate_batch(raw, [source(), source("EM-00009")])

    def test_bad_amount_and_date_fail(self):
        for changes in [{"amount": "NaN"}, {"amount": "-1"}, {"amount": "1e4"},
                        {"payment_date": "2026-02-30"}, {"currency": "US"}]:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                payment(**changes)

    def test_partial_run_keeps_all_emails_and_reports_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = EmailResult(email_id="EM-00008", payments=[payment()], issues=[])
            summary = write_results([source(), source("EM-00009")],
                                    {"EM-00008": (result, "offline-test")}, Path(tmp))
            self.assertTrue(summary["all_source_emails_represented"])
            self.assertFalse(summary["complete_api_run"])
            self.assertEqual(summary["status_counts"]["not_processed"], 1)

    def test_rejects_paid_model_before_network(self):
        with patch("extractor.urlopen") as network:
            with self.assertRaises(ValueError):
                request_batch([source()], "not-a-real-key", "vendor/paid-model")
            network.assert_not_called()

    def test_source_duplicate_id_fails(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.csv"
            pd.DataFrame([source(), source()]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_source(path, 2)

    def test_schema_fields_are_required_but_can_be_null(self):
        schema = BatchResult.model_json_schema()["$defs"]["Payment"]
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
