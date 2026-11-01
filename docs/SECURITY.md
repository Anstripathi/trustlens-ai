# Security

## Existing application protections

The app treats uploaded files as untrusted data. CSV uploads have extension, size, row/column, binary-content, and strict parsing checks. Serialized model artifacts are rejected and never loaded. `safe_join` blocks path traversal, unexpected exceptions receive generic UI messages, and the Streamlit configuration suppresses detailed errors and usage statistics. Uploaded datasets remain in per-session memory; the app does not log or persist them.

## AI Security screening

The AI Security page in `pages/6_AI_Security.py` runs local checks over a prompt, an optional model output, and optionally selected columns from the dataset already loaded in the current session. It does not execute input, call a model, download a moderation model, or transmit text to an external service. Results and built-in test case inputs remain in the active Streamlit session.

The `src/security/` package is separate from tabular training and contains:

- Prompt-injection pattern checks for instruction overrides, hidden-prompt requests, hierarchy/role manipulation, restriction bypass, disclosure/tool requests, and selected obfuscation signals.
- A deterministic, local jailbreak fixture suite. It measures whether configured rules flag the fixture prompts; it does not send them to a target LLM and does not establish target-model vulnerability.
- PII and credential-like pattern checks for email, phone, US SSN-like values, Luhn-valid card-like numbers, common token shapes, password assignments, private-key markers, IPv4/IPv6 addresses, and user-configured field labels. Detected values are omitted or masked in evidence. Government-ID checks are jurisdiction-limited and are not identity verification.
- Limited lexical unsafe-output rules for selected abusive, harassment, threat, self-harm, sexual exploitation, violence, and hate-related phrases. This is not semantic moderation and is not production-grade content safety.
- A bounded assessment runner that executes selected checks, records skipped/failed status, isolates individual scanner failures, and limits dataset scanning to selected columns, 2,000 cells, and 100,000 characters. Text inputs are capped at 20,000 characters.
- A transparent ordinal risk prioritization score. Per-finding severity points are Informational 0, Low 10, Medium 25, High 50, Critical 75; detector-strength multipliers are High 1.0, Medium 0.75, Low 0.5; category weights are in `src/security/risk_scoring.py`; evidence omissions reduce the contribution; the sum is capped at 100. Tiers are Low below 15, Moderate below 35, High below 60, otherwise Critical. These choices are prioritization weights, not probabilities.
- In-memory JSON and CSV report exports. Reports include check status, findings, bounded masked evidence excerpts, limitations, and scoring method; they exclude full supplied prompts, outputs, and dataset cell contents. CSV values are protected against spreadsheet formula interpretation.

Findings identify their detector type/strength and are marked **Not reviewed**. They require human interpretation. A low score or no matches is not evidence that a model, prompt, output, or dataset is secure. Heuristics can miss attacks and produce false positives. No human review or target-LLM test is claimed by this module.

## Limitations

The project has not been penetration-tested or independently audited. Session isolation relies on Streamlit's per-session state. The local scanners do not cover every language, jurisdiction, encoding, or attack pattern. Dependency ranges are not fully pinned, and supply-chain risk remains. App testing does not establish deployment or browser security.

The AI Firewall adds deterministic policy decisions and optional process-local rate limiting. Its test-console approval queue is session-local and reviewer labels are not authenticated identities; it does not authorize access to a protected system. Callers must enforce `BLOCK`, `REDACT`, and `APPROVAL_REQUIRED` outcomes and avoid logging `output_text`. See [FIREWALL.md](./FIREWALL.md) for the callable contract and limitations.

## Model Auditor privacy

The auditor reuses local scanners but omits all source excerpts from consolidated findings, including otherwise masked prompt/output excerpts. Prompt and output text is not placed in audit results or exported reports. Dataset scan evidence contains source type and detector metadata rather than cell values. JSON/CSV exports are generated in memory and exclude raw dataset rows and class labels; aggregate statistics and selected feature names may still be sensitive in some contexts and should be reviewed before sharing. See `docs/AUDITOR.md` for report fields, risk methodology, and execution bounds.
