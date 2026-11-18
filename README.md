# Payment email extractor

Zero Data Engineer scenario, stage 1: turn the supplied payment emails into a
traceable payment table with Python, Pydantic, Pandas and a free OpenRouter model.

## Current status

Completed extraction on 18 November 2026: **300 emails, 302 payment rows**.
All source email IDs are represented, with no duplicate payment keys or
amount/currency coverage mismatches. Seventeen explicit bank-change examples
were flagged and their old/new account pairs checked against the source.

The final table has 297 rows marked `extracted` and five carrying review notes:
the undated EM-00015 payment, both EM-00021 rows (an email-level date ambiguity),
EM-00261 (corrected payee) and EM-00291 payment 2 (removed inferred date).
These are extraction statuses, not payment approval or hold decisions.

The run used `inclusionai/ling-3.2-flash:free`. A provider rate limit interrupted
one batch; the remaining 25 emails were resumed from cache successfully.
Fifteen offline tests pass. The two Zero resource downloads are byte-identical,
so only one source file is used. Sarah accepted the field and row design.

## Setup and run

Use Python 3.11 or newer. From this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
python extractor.py --input data/client_payment_emails.csv --inspect
```

Supply `OPENROUTER_API_KEY` in the process environment, or use `--key-file` with
an existing local file containing only the key. Do not paste the key into chat,
source code, GitHub, a command argument or a screenshot. `.env.example` documents
the variable names; the program does not automatically load `.env` files.

```sh
python extractor.py --input data/client_payment_emails.csv --key-file /path/to/private/key --max-batches 1
python extractor.py --input data/client_payment_emails.csv --key-file /path/to/private/key
```

The first command is an optional ten-email smoke test and exits with code 2
because the dataset is incomplete. The second resumes cached progress. The
default model is `openrouter/free`. Paid model IDs are rejected. Requests also
set a zero token-price ceiling and require support for structured output.

The final reviewed run used the following command. The correction file applies
only to the matching source hash and original values; it refuses silent drift.

```sh
python extractor.py --input data/client_payment_emails.csv --key-file /path/to/private/key --model inclusionai/ling-3.2-flash:free --batch-size 25 --workers 3 --review-corrections review_corrections.json
python audit.py --source data/client_payment_emails.csv --results results/payments.csv --output results/audit_summary.json
```

Each request sends a batch of training emails (ten by default, up to 25) to OpenRouter and its
selected model provider. Use this workflow only for the supplied training data.
No account, key creation, subscription or credit purchase is automated.

## Table contract

- One row per distinct requested payment, linked by `email_id` and a one-based
  `payment_index`. A quoted repetition of the same payment is not a new payment.
- Copy client/source metadata from the CSV, rather than asking the model to guess.
- Extract payee, decimal amount, currency, payment date, requested account,
  previous account, reference, explicit account-change flag and supporting quote.
- Account fields remain strings with leading zeros, separators and sort codes.
- Missing fields remain null/blank. Do not assume that the first payment's date
  or currency applies to another payment without explicit source support.
- Read the whole thread. Keep previous and requested accounts separate.
- If the model omits a reference and the source has one unambiguous `Ref` or
  `invoice` label, recover it directly and mark `reference_source` as
  `explicit_source_label`. This recovered 11 references in the submitted run.

For example, EM-00008 contains an account number plus sort code. EM-00021 and
EM-00291 each request two payments. EM-00012 and EM-00088 contain account changes
inside forwarded messages. These are spot-check cases, not a complete gold set.

## Results and limitations

`results/payments.csv` contains the table. `run_summary.json` reports processed
emails, payment rows and extraction statuses. The cache permits a retry without
resending successful emails when the input, prompt, schema and model match.
Unprocessed or unparsed emails stay visible as explicit placeholder rows with
`payment_index=0`. They never disappear from the report.

Pydantic validates shape and field types. Additional checks flag unsupported
source text, amounts, missing required details and exact duplicates. They cannot
prove that an LLM assigned each supported value to the correct payment. Review
the table against the source, especially multi-payment and bank-change cases.
`extracted` means extraction checks passed; it does not mean verified, safe,
ready, approved or paid. Hold rules, Snowflake and later stages are outside this
project's current scope.

Source review found two model errors and recorded them in
`review_corrections.json`: the added word “Clearing” in EM-00261's payee and an
inferred date on EM-00291's second payment. The table preserves each correction's
original value, replacement and reason. The review was performed with an AI
assistant; it is not an independent assessment of the learner's ability.

The `submission/` folder contains the final table and run/audit summaries.
The source email export and API-response cache remain local. The source audit
checks amount/currency coverage and account-change keywords; it is not a proof
of every semantic mapping or future model accuracy.

The program stops on authentication, rate-limit or response errors and writes
the partial result. Error output excludes API keys and provider response bodies.
Free-model availability and quotas can change. No silent paid fallback is used.

Source CSV, caches, working results and secrets are excluded from Git. The
reviewed submission files are intentionally included. Keep the original
training resource locally at `data/client_payment_emails.csv`.

## References

- [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs)
- [OpenRouter API limits](https://openrouter.ai/docs/api_reference/limits)
- [OpenRouter free router](https://openrouter.ai/openrouter/free)
- Zero stage 1 resources: *Client Payment Emails, Shared Operations Inbox Export
  (Sep–Nov 2026)*, downloaded 18 November 2026. SHA-256:
  `55279909ea1c388192cd57eadd164c848261ba83b3619dba41ba6f8c579fd073`.
