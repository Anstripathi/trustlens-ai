# AI Firewall and Guardrails

The AI Firewall is a local policy-enforcement service over TrustLens' existing deterministic prompt-injection, PII, and unsafe-output scanners. The Streamlit test console is [pages/8_AI_Firewall.py](../pages/8_AI_Firewall.py); callable services are in `src/firewall/`.

## Callable interface

```python
from src.firewall import Action, Direction, Firewall, FirewallPolicy
from src.firewall.rate_limit import RateLimiter

policy = FirewallPolicy()
firewall = Firewall(policy, rate_limiter=RateLimiter())
result = firewall.inspect(untrusted_text, direction=Direction.INPUT, client_id=session_id)

if result.action in (Action.ALLOW, Action.WARN, Action.REDACT):
    # Consume result.output_text only according to the caller's policy.
    ...
elif result.action == Action.APPROVAL_REQUIRED:
    # Keep payload local; request/retrieve a metadata-only approval decision first.
    ...
else:
    # BLOCK and fail-closed scanner errors have no output_text.
    ...
```

`inspect_text` is the functional interface. `Firewall` binds a policy and optional in-memory request limiter. The limiter key must be an opaque session/client identifier, never a prompt, output, email, or other user data. Its state is process-local and resets when the app restarts.

## Decisions and policy

- Input checks run prompt-injection heuristics and sensitive-data detectors. Output checks run unsafe-output rules and sensitive-data detectors.
- A configurable severity threshold selects `warn` or `block`; PII has separate input/output behavior. Output redaction masks common emails, credentials, phone-like values, Luhn-valid payment-card-shaped values, government-ID-like values, and IPv4 addresses. False positives are possible.
- Custom rules use bounded literal phrases rather than regular expressions. Actions are `allow`, `warn`, `block`, `redact`, and `approval_required`.
- Each saved UI policy change increments the session policy version. Policy JSON exports the configured literal phrases; firewall event reports omit them and contain metadata only.
- A `RateLimiter` is optional for callable integrations and is enabled per Streamlit session in the test console.
- Invalid or oversized input, rate-limit failures, and scanner exceptions fail closed with no returned content. Configuration errors are raised to the caller before scanning.

`output_text` is a transient caller payload. It is never included in `FirewallResult` serialization, session event history, or event reports by the provided services. Callers should not log it. `WARN` returns the original text for callers that deliberately allow warned content; `BLOCK` and `APPROVAL_REQUIRED` return no content; `REDACT` returns the transformed text.

## Approvals and history

`ApprovalQueue` accepts only `approval_required` results and retains policy/version, direction, matched rule IDs, timestamps, decision, and a reviewer label. It never stores the payload. It is a session-local workflow, not authentication or access control; a reviewer label is self-asserted. The caller must retain any content locally and gate consequential actions on a separately verified approval state. The sample UI demonstrates a review record only; it does not release held content or invoke an external system.

The UI keeps at most 100 session events. Reports include decisions, policy version, rule IDs, finding counts, categories, and approval metadata, but no raw content, evidence excerpts, or rule phrases. CSV cells are protected against spreadsheet formulas.

## Limits

The scanners are lexical and pattern-based. They do not interpret full context, validate model behavior, prove security, or replace application-specific authorization, moderation, data-loss prevention, or human review. Community Cloud does not provide durable local event history; this UI's policy and event state is session-only. No external AI service is called.
