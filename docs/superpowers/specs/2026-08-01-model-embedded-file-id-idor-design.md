# Model-embedded file ID IDOR: bind caller identity into the ID

Date: 2026-08-01

## Problem

`check_managed_file_id_access` (`enterprise/litellm_enterprise/proxy/hooks/managed_files.py:371-390`)
runs the owner/team check only when the caller-supplied id decodes to the
`litellm_proxy`-prefixed unified format. Every other id shape falls through to
`return False`, a silent pass-through that neither denies nor logs.

The proxy has a second, independent id format that hits exactly that
pass-through. `encode_file_id_with_model`
(`litellm/proxy/openai_files_endpoints/common_utils.py:94`) produces
`<prefix><base64("litellm:<inner_id>;model,<model_name>")>` with prefix `file-`
or `batch_`. Throughout this document that format is called **model-embedded**,
and the unified `litellm_proxy` format is called **unified**.

A caller holding any valid virtual key can hand-craft a model-embedded id and
read, delete, or cancel another tenant's resource:

```
GET /v1/files/file-<base64("litellm:s3://<bucket>/litellm-batch-outputs/<job>/input.jsonl.out;model,<configured-model>")>/content
```

The existing raw-URI guard does not stop this. `is_managed_cloud_storage_uri`
is a plain `startswith` on `("s3://", "gs://")`
(`litellm/litellm_core_utils/cloud_storage_security.py:22-30`) applied to the
outer path parameter at `files_endpoints.py:719`, while the model-embedded
decode happens afterwards at `:730`. The guard therefore fires only on a bare
URI, never on the base64-wrapped form.

### Affected endpoints

| Endpoint | Handler | Impact |
|---|---|---|
| `GET /v1/files/{id}/content` | `files_endpoints.py:581` | read another tenant's object |
| `GET /v1/files/{id}` | `files_endpoints.py:872` | read another tenant's metadata |
| `DELETE /v1/files/{id}` | `files_endpoints.py:1053` | delete another tenant's file |
| `GET /v1/batches/{id}` | `batches_endpoints/endpoints.py:326` | read another tenant's batch |
| `POST /v1/batches/{id}/cancel` | `batches_endpoints/endpoints.py:749` | cancel another tenant's batch |

A second, related defect affects `get_file` (`files_endpoints.py:940`) and
`delete_file` (`:1132`): `if should_route:` is evaluated *before* the unified-id
branch (`:961`, `:1152`). Appending `?model=X` to a legitimate unified id
diverts it onto the model-routing path and away from the ownership check.

### Scope relative to PR #31435

This defect predates PR #31435. On `litellm_internal_staging`,
`BEDROCK_MANAGED_S3_OUTPUT_PREFIX = "litellm-batch-outputs/"` is already in
`BEDROCK_MANAGED_S3_PREFIXES`
(`litellm/litellm_core_utils/cloud_storage_security.py:10-17`), and the
model-embedded path already lacked an ownership check. Bedrock batch create
also falls back to writing output into the input bucket when no separate output
bucket is configured (`litellm/llms/bedrock/batches/transformation.py:108-111`),
so forged ids could already reach batch outputs. PR #31435 widens the trusted
bucket set from `{input}` to `{input, output}`, which extends the reachable
surface to a dedicated output bucket. That is a real but narrow widening of an
already-broken check, which is why the fix ships as its own PR off
`litellm_internal_staging` rather than inside #31435.

## Constraint that rules out the obvious fixes

A model-embedded id whose decoded inner value is a raw `s3://` or `gs://` URI
is a supported contract, not an anomaly. It cannot be rejected.

- Bedrock computes the batch output URI client-side specifically so
  `client.files.content(output_file_id)` works without an extra S3
  `ListObjectsV2` round-trip
  (`litellm/llms/bedrock/batches/handler.py:44-63`, docstring at `:52-53`;
  `output_file_id` set at `:314`).
- `encode_batch_response_ids` (`common_utils.py:138-149`) wraps whatever the
  provider put in `output_file_id` with no shape check, on five call paths.
- `tests/test_litellm/proxy/test_batch_retrieve_bedrock.py:214` asserts the id
  must decode back to the raw S3 URI before reaching `litellm.afile_content`.
- Vertex is equivalent, producing `gs://…/predictions.jsonl`
  (`litellm/llms/vertex_ai/batches/transformation.py:121-143`).

Requiring a database record is equally unavailable. The model-embedded path is
the deliberately DB-free alternative to managed files; the priority list at
`files_endpoints.py:152-159` documents them as co-existing options, and the
unified path is gated on Prisma. Nothing in `enterprise/` mints or decodes the
model-embedded format.

Those two constraints leave one viable remedy: make the id **unforgeable**
rather than unusable, without consulting a database.

## Design

Bind the minting caller's identity into the id and authenticate it with an
HMAC. Because these ids are minted through one function and decoded through one
pair of functions, the change lands at a single chokepoint rather than as five
parallel endpoint patches.

### Payload

Current: `litellm:<inner_id>;model,<model_name>`

New: `litellm:<inner_id>;model,<model_name>;sub,<user_id>;tid,<team_id>;sig,<hmac_hex>`

where the HMAC covers everything preceding `;sig,`:

```
signing_input = "litellm:<inner_id>;model,<model_name>;sub,<user_id>;tid,<team_id>"
key           = HMAC-SHA256(salt_key, b"litellm-file-id")
sig           = HMAC-SHA256(key, signing_input)
```

`salt_key` comes from `LITELLM_SALT_KEY`, falling back to `master_key`, reusing
`_get_salt_key()` (`litellm/proxy/common_utils/encrypt_decrypt_utils.py:8-16`).
The derivation through a fixed label mirrors the established pattern in
`litellm/proxy/plugin_routes.py:126-135`, so the raw salt is never used
directly and a signature for one purpose cannot be replayed for another. Key
material is therefore available on every proxy deployment; `master_key` is
required to serve authenticated traffic at all.

The signature covers `model` as well as `inner_id`. Without that, a caller
could take their own legitimately signed id and swap in a different model to
borrow another deployment's credentials, a privilege escalation adjacent to the
IDOR that costs nothing extra to close.

There is no expiry field. Batch outputs legitimately persist for a long time
and lifetime is already modeled separately through `output_expires_after`
(`batches_endpoints/endpoints.py:146-149`). A second clock would break
long-running batches without adding protection, since the signature binds
identity rather than time.

### Decode and authorize are separate

Two existing callers decode these ids without any caller identity in scope and
must keep working:

- `litellm/proxy/auth/auth_utils.py:1324` calls `decode_model_from_file_id`
  only to derive model-access candidates for a different check.
- `litellm/proxy/hooks/batch_rate_limiter.py:365` calls `get_original_file_id`
  on an internal path.

Therefore:

- **Decoding stays unauthenticated and keeps its current signature.**
  `decode_model_from_file_id` (`common_utils.py:152`) and
  `get_original_file_id` (`:181`) tolerate the appended fields with no change,
  because each already extracts its own field via `re.search` and
  `split(";")[0]`.
- **Authorization is a new, separate function** that the five endpoints call:
  `authorize_model_embedded_file_id(file_id, user_api_key_dict) -> None`.

### Failure modes as values

Verification returns a tagged union rather than raising:

- `NotModelEmbedded` — not this format; **no-op pass-through** so unified ids
  and plain provider ids continue to their own handling.
- `Verified(inner_id, model, sub, tid)` — authorize via `can_access_resource`.
- `Unsigned` — model-embedded but carries no `sig` field.
- `BadSignature` — `sig` present and wrong, or covered fields tampered with.

A single mapping function converts that union to the public HTTP contract with
an exhaustive `match` plus `assert_never`, keeping the raising confined to one
place.

### Authorization semantics

`Verified` delegates to the existing `can_access_resource`
(`litellm/llms/base_llm/managed_resources/isolation.py:72-95`), passing
`created_by=sub` and `resource_team_id=tid`. This reuses one permission model
across both id formats: proxy admins bypass, the creator is allowed, a matching
team is allowed, and the `None == None` bypass is already guarded against
upstream. Defining separate semantics here would fork the permission model
between two paths on the same proxy.

### Error contract

`Unsigned` and `BadSignature` both produce **403**, with wording consistent
with the existing denial at `managed_files.py:385-388`.

Not 404: the unified path already answers 403 for the analogous denial, and
distinguishing "forged" from "not yours" would leak whether an object exists.
Not 400: this is an authorization outcome, not malformed input.

Rejecting unsigned ids is a deliberate breaking change. An id minted by an
older build carries no signature and will be refused; the holder re-fetches via
`GET /v1/batches/{batch_id}` to obtain a signed one. A compatibility switch was
considered and rejected: while enabled it leaves the vulnerability fully open,
which defeats the purpose of the fix. This needs a release note.

### Call placement

The authorization call goes **before** the model-routing branch in all five
handlers, not inside the `should_route` arm. Placing it before is what also
closes the `?model=X` diversion on `get_file` and `delete_file`, since the
check then runs regardless of which branch the request would take.

Existing provider-level confinement stays untouched as defense in depth:
`validate_managed_cloud_file_id`
(`litellm/litellm_core_utils/cloud_storage_security.py:133-166`) continues to
confine ids to server-configured buckets and LiteLLM-managed prefixes. This
design adds the tenancy layer that confinement never provided.

### Mint sites

All four already have `user_api_key_dict` in scope, verified, so no parameter
plumbing through call chains is required:

| Site | Enclosing function |
|---|---|
| `files_endpoints.py:209` | `route_create_file` (`:137`) |
| `batches_endpoints/endpoints.py:183-198`, `:262` | `create_batch` (`:61`) |
| `batches_endpoints/endpoints.py:469` | `retrieve_batch` (`:326`) |
| `batches_endpoints/endpoints.py:664` | `list_batches` (`:574`) |
| `batches_endpoints/endpoints.py:840` | `cancel_batch` (`:749`) |

### `list_batches` must not sign (resolved)

`encode_batch_response_ids` at `endpoints.py:664` re-encodes every batch in a
model-routed `list_batches` response. That path calls `litellm.alist_batches`,
documented as "List your organization's batches"
(`litellm/batches/main.py:654`) and scoped only by the upstream deployment
credentials. It therefore returns every batch created with that provider API
key, across all LiteLLM tenants sharing the deployment.

Signing there would bind the *calling* user to batches they do not own, minting
exactly the credential the rest of this design exists to prevent. It would also
hand an attacker an official signature-issuing oracle: call `list_batches`
once, collect valid signed ids for other tenants' batches, and the fix is void.

Therefore the model-routed `list_batches` path mints **unsigned** ids. Ids
obtained from it are consequently not usable for `retrieve`, `cancel`, or file
download, which will answer 403. Callers needing usable ids must use the
managed path, which is already owner-filtered through `build_owner_filter`
(`isolation.py:33-69`). This is a deliberate behaviour change and needs a
release note alongside the unsigned-id rejection.

The general rule this expresses: sign only where ownership is known at mint
time. `create_file`, `create_batch`, `retrieve_batch`, and `cancel_batch` all
act on a single resource the caller just created or already passed
authorization for; a provider-wide listing does not.

## Testing

Every test below must fail if the corresponding logic is removed or mutated.

1. **Cross-tenant replay.** Tenant A mints an id; tenant B replays it verbatim;
   expect 403. Fails if `can_access_resource` is dropped.
2. **Forgery.** A hand-crafted unsigned id pointing at
   `s3://<bucket>/litellm-batch-outputs/<job>/x.jsonl.out`; expect 403. This is
   the reported attack. Fails if unsigned ids are accepted.
3. **Tampering.** A validly signed id with `model` swapped and the original
   signature retained; expect 403. Fails if the signature stops covering
   `model`.
4. **Team sharing still works.** Same team, different user; expect success.
   Fails if the check is tightened to creator-only, which would silently
   diverge from the unified path.
5. **`?model=X` diversion closed.** A legitimate unified id plus `?model=X` on
   `get_file` and `delete_file`; the ownership check must still run. Fails if
   the call is placed inside the `should_route` arm.
6. **Round-trip preserved.**
   `tests/test_litellm/proxy/test_batch_retrieve_bedrock.py:214` must still
   pass, proving the OSS Bedrock batch flow is intact.
7. **Constant-time comparison.** Verification uses `hmac.compare_digest`, not
   `==`.

Six existing test files construct model-embedded ids and need updating for the
new mint parameters: `test_model_based_routing_files_batches.py`,
`test_batch_retrieve_bedrock.py`, `test_batch_x_litellm_model_encoding.py`,
`proxy/auth/test_auth_utils.py`, `proxy/batches_endpoints/test_endpoints.py`,
`proxy/hooks/test_batch_file_validation.py`.

## Proof of fix

Live proxy against real AWS Bedrock, not mocks, following the repo convention
of curl over test scripts:

1. Create two virtual keys on different teams.
2. Run a real Bedrock batch under key A through to completion.
3. `GET /v1/files/<A's output_file_id>/content` with **key B** -> expect 403.
4. The same call with **key A** -> expect 200 and real output bytes.
5. Repeat step 3 with a hand-forged unsigned id -> expect 403.

## Explicitly out of scope

- The six `vector_store_files_endpoints` handlers have no file-level ownership
  check for *either* id format (only a vector-store-level check). A real gap,
  but a distinct problem.
- `litellm/proxy/pass_through_endpoints/managed_id_rewriter.py` is a third
  managed-id system with its own enforcement
  (`_guard_raw_provider_id`, `:1076-1087`), already handling cross-tenant raw
  ids.
- `create_batch` and `create_fine_tuning_job` perform no ownership check on
  input ids for either format; unchanged here.

Folding these in would make the PR unreviewable. They are recorded for
follow-up.
