# Payment email extractor

Zero Data Engineer scenario, stage 1: turn the supplied payment emails into a
traceable payment table with Python, Pydantic, Pandas and a free OpenRouter model.

## Current status

Source data inspected: 300 unique email records. The two resource downloads in
Zero are byte-identical, so only one is used. Sarah accepted the field selection
and the rule of one row per distinct payment. Live extraction is pending
OpenRouter authentication. No payment table is claimed to be complete yet.

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

Each request sends a batch of up to ten training emails to OpenRouter and its
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

The program stops on authentication, rate-limit or response errors and writes
the partial result. Error output excludes API keys and provider response bodies.
Free-model availability and quotas can change. No silent paid fallback is used.

Source CSV, caches, outputs and secrets are excluded from Git. Keep the original
training resource locally at `data/client_payment_emails.csv`.

## References

- [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs)
- [OpenRouter API limits](https://openrouter.ai/docs/api_reference/limits)
- [OpenRouter free router](https://openrouter.ai/openrouter/free)
- Zero stage 1 resources: *Client Payment Emails, Shared Operations Inbox Export
  (Sep–Nov 2026)*, downloaded 18 November 2026. SHA-256:
  `55279909ea1c388192cd57eadd164c848261ba83b3619dba41ba6f8c579fd073`.
