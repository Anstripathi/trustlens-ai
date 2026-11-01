# Governance and enterprise controls

## Bounded orchestration

Governance Center runs the existing local auditor once through its fixed stages. It has no shell, arbitrary tool registry, background loop, network call, or autonomous model replacement. Policy gates route cases to human review based on the current audit evidence and configurable risk threshold; eligibility is not approval or certification.

## Decisions and roles

The decision queue stores only decision IDs, run IDs, action, reviewer label, status and timestamps. Only allowlisted actions can be requested. Reviewer/admin role checks are local input validation, not authentication. In the public anonymous Streamlit deployment, a user can claim any role; consequential use requires an authenticated identity layer outside this app.

## Versioned policy and audit log

Governance Control Center stores versioned JSON policy snapshots and metadata-only JSONL events under the ignored local `artifacts/` directory. Policy digests use SHA-256. The event chain links each event to its predecessor and verifies record edits. It does not prevent an administrator, hosting provider or attacker with storage access from rewriting the complete file. The local trail is not a certified, immutable or tamper-proof audit system. Do not treat it as regulatory evidence without managed identity, access controls, durable backups and an external append-only store.

## LLM and RAG evaluation

The evaluator accepts user-supplied CSV traces and invokes no LLM or external API. Lexical overlap is a rough grounding proxy; citation presence checks marker syntax only; repeated-prompt checks detect normalized answer variation, not semantic contradictions. Local injection/PII pattern scanners are incomplete. Raw prompts, answers and contexts are not copied into the returned report or persisted by the evaluator.
