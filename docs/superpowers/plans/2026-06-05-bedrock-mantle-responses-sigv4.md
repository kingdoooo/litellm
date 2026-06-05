# Bedrock Mantle Responses SigV4 / IAM Auth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SigV4 / IAM Role auth as an opt-in alongside the existing Bearer token path on the bedrock_mantle Responses route (`/openai/v1/responses`), so IAM-only deployments (EKS IRSA, Pod Identity, EC2 instance role) work without static keys.

**Architecture:** `BedrockMantleResponsesAPIConfig` keeps inheriting `OpenAIResponsesAPIConfig` and gains SigV4 by composition: it holds an injected `BaseAWSLLM` and reuses its `_sign_request` (the same primitive the Mythos chat route uses). A default no-op `sign_request` is added to `BaseResponsesAPIConfig`, and the responses handler (sync + async) calls it after the body is finalized, sending `data=signed_body` when signing occurred and `json=data` otherwise. This mirrors the existing chat/embedding signed-body pattern, so the 15 existing responses providers are unaffected (they inherit the no-op).

**Tech Stack:** Python, pytest, httpx, botocore SigV4, LiteLLM provider-config pattern.

**Spec:** `docs/superpowers/specs/2026-06-05-bedrock-mantle-responses-sigv4-design.md`

**Branch:** `litellm_bedrock_mantle_responses_sigv4` (already cut from `origin/litellm_oss_staging_040626`). PR base = `litellm_oss_staging_040626`.

---

## Key facts verified against the code (read before starting)

- Responses handler call order (`litellm/llms/custom_httpx/llm_http_handler.py`, `response_api_handler` ~2278 and `async_response_api_handler` ~2424): `validate_environment` (no body) -> `get_complete_url` -> `transform_responses_api_request` -> `normalize_responses_api_request_dict` -> `data.update(extra_body)` -> in the stream branch, optional `_prepare_fake_stream_request` (which does `data.pop("stream", None)`) -> `post(json=data)`. **Signing must happen after fake-stream prep so the signed bytes equal the sent bytes.**
- `BaseAWSLLM._sign_request` (`litellm/llms/bedrock/base_aws_llm.py:1464`) returns `Tuple[dict, Optional[bytes]]`. With a bearer (`api_key` arg, else `AWS_BEARER_TOKEN_BEDROCK`) it sets `Authorization: Bearer ...` and returns `json.dumps(request_data).encode()`. Otherwise it reads `aws_access_key_id / aws_secret_access_key / aws_session_token / aws_role_name / aws_session_name / aws_profile_name / aws_web_identity_token / aws_sts_endpoint / aws_external_id / aws_region_name` from `optional_params`, calls `get_credentials(...)` then `SigV4Auth`, and returns `(signed_headers, request.body)`. It does **not** read `BEDROCK_MANTLE_API_KEY` — that resolution stays in the Mantle layer.
- With no bearer and no resolvable AWS credentials, `_sign_request` raises `botocore.exceptions.NoCredentialsError("Unable to locate credentials")` (verified empirically).
- `dict(GenericLiteLLMParams(...))` carries `api_key`, `aws_region_name`, `aws_access_key_id`, `aws_role_name`, etc. (verified), so the handler can pass `optional_params=dict(litellm_params)` and `api_key=litellm_params.api_key` straight through.
- `AmazonInvokeConfig.sign_request` (`litellm/llms/bedrock/chat/invoke_transformations/base_invoke_transformation.py:114`) is the reference: a thin forward to `self._sign_request(service_name="bedrock", ...)`. Mantle's override mirrors it but uses the composed signer and resolves the Mantle bearer first.
- Existing chat/embedding signed-body pattern to mirror: `litellm/llms/custom_httpx/llm_http_handler.py` ~896-956 (`headers, signed_body = provider_config.sign_request(...)` then `if signed_body is not None: post(data=signed_body) else: post(json=data)`).
- Registry already returns the Mantle config (`litellm/utils.py:8920`); **no routing/registry change in this plan.**
- **Region must be a single source of truth (adversarial-review finding, deepened in 2nd review).** Two distinct ways the URL host region and the SigV4 credential-scope region can diverge, both causing a 401 in the IAM-only deployment this PR targets:
  - (a) The original `get_complete_url` resolved region from `BEDROCK_MANTLE_REGION` -> `AWS_REGION` -> default and did **not** read `aws_region_name`/`AWS_REGION_NAME`, while signing's `_get_aws_region_name` reads `aws_region_name` first.
  - (b) **More subtle (2nd-round finding, verified by running `litellm.get_llm_provider`):** before the config runs, `litellm/responses/main.py:688-691` (`_resolve_model_provider_for_responses`) calls `get_llm_provider`, which for `bedrock_mantle` returns `dynamic_api_base = "https://bedrock-mantle.<DEFAULT-region>.api.aws/v1"` (region from `BEDROCK_MANTLE_REGION`/`AWS_REGION` only, ignoring `aws_region_name`) and writes it into `litellm_params.api_base`. So `get_complete_url` receives a **non-None** `api_base` pinned to the default region; a naive "only resolve region when api_base is None" fix is bypassed, and the URL stays default-region while signing uses `aws_region_name`. `dynamic_api_key` is correctly `None` in the IAM case, so SigV4 still fires (the bug is wrong-region signing, not auth selection).
  - The Mythos chat route avoids both because it is routed under `custom_llm_provider="bedrock"` (never hits the `bedrock_mantle` base-injection branch) and its `get_complete_url` rebuilds the URL from `_get_aws_region_name(optional_params)`, ignoring any incoming `api_base`.
  - The fix (Task 3a): one `_resolve_region(params)` helper (precedence: `aws_region_name` -> region embedded in an explicit Mantle `api_base` -> `BEDROCK_MANTLE_REGION`/`AWS_REGION_NAME`/`AWS_REGION` -> default); `get_complete_url` pins standard Mantle hosts to that resolved region (so the injected default-region base cannot win over `aws_region_name`) while preserving genuinely custom proxy hosts; `sign_request` injects the same resolved region into `optional_params` before signing. Verified consistent across all existing URL tests plus the injected-base and aws_region_name-only cases.
- **Caller `Authorization` can clobber SigV4 (adversarial-review finding).** The handler does `headers.update(extra_headers)` after `validate_environment`; once `validate_environment` no longer forces a Bearer header, a caller-supplied `extra_headers["Authorization"]` lands in `headers`. `_sign_request` re-applies original `Authorization` after SigV4 signing (`base_aws_llm.py:1556-1559`, "prevent sigv4 from overwriting the auth header"), so that stale header would override the SigV4 `Authorization`. The fix (Task 3a): in the SigV4 branch (no bearer) strip any incoming `Authorization` before signing.
- By-id responses subroutes (`delete`/`get`/`cancel`/`compact`/`list_input_items`) call `get_provider_responses_api_config` with `model=None`; the Mantle gate (`litellm/utils.py:8915-8920`) returns `None` for `model=None`, so the Mantle SigV4 config never applies to them. They are out of scope here and need no signing (verified; this is why the adversarial-review "unsigned subroutes" concern does not apply).
- **`sign_request` belongs before the handler `try:` (2nd-round finding).** `_handle_error` (`llm_http_handler.py:5203`) wraps whatever it catches into `provider_config.get_error_class(..., status_code=500)`. If the `sign_request` call sat inside the `try:`, the both-auth-missing `ValueError` would be rewrapped as a generic 500 (the message text survives via `str(e)`, but it is no longer a clean `ValueError`). Signing performs no network I/O, so Task 2 places `sign_request` (and the fake-stream prep it depends on) before the `try:`; only the `post` calls — the real source of network/HTTP errors — stay inside it.

## File Structure

- Modify `litellm/llms/base_llm/responses/transformation.py` — add default no-op `sign_request` to `BaseResponsesAPIConfig`.
- Modify `litellm/llms/custom_httpx/llm_http_handler.py` — add the signing hook + `data`/`json` body selection in both `response_api_handler` and `async_response_api_handler`.
- Modify `litellm/llms/bedrock_mantle/responses/transformation.py` — compose a `BaseAWSLLM`, override `sign_request`, relax `validate_environment`, update the module docstring.
- Modify `tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py` — extend with SigV4 tests; update the now-stale missing-key test.
- Modify `tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py` — add the handler-wiring regression test (no-op stays `json`, signer uses `data`).

---

## Task 1: Default no-op `sign_request` on `BaseResponsesAPIConfig`

**Files:**
- Modify: `litellm/llms/base_llm/responses/transformation.py` (class `BaseResponsesAPIConfig`, after `supports_native_file_search` ~line 63)
- Test: `tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py`:

```python
def test_base_responses_config_sign_request_is_noop_by_default():
    """Default responses sign_request must be a no-op: unchanged headers, no signed body.

    Guards the 15 existing responses providers from accidental signing when the
    handler starts calling sign_request.
    """
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

    cfg = OpenAIResponsesAPIConfig()
    headers = {"Authorization": "Bearer sk-existing"}
    out_headers, signed_body = cfg.sign_request(
        headers=headers,
        optional_params={},
        request_data={"input": "hi"},
        api_base="https://api.openai.com/v1/responses",
    )
    assert out_headers == {"Authorization": "Bearer sk-existing"}
    assert signed_body is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py::test_base_responses_config_sign_request_is_noop_by_default -v`
Expected: FAIL with `AttributeError: 'OpenAIResponsesAPIConfig' object has no attribute 'sign_request'`

- [ ] **Step 3: Add the method**

In `litellm/llms/base_llm/responses/transformation.py`, ensure `Tuple` and `Optional` are imported (they are, line 3: `from typing import ... Optional, Tuple, ...`). Add this method to `BaseResponsesAPIConfig` immediately after `supports_native_file_search` (the method ending at ~line 63):

```python
    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        """Sign the request after the body is finalized.

        Default is a no-op (returns headers unchanged, no signed body). Providers
        whose endpoint requires request signing (e.g. Bedrock Mantle SigV4)
        override this and return the signed body bytes so the handler sends those
        exact bytes.
        """
        return headers, None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py::test_base_responses_config_sign_request_is_noop_by_default -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add litellm/llms/base_llm/responses/transformation.py tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py
git commit -m "feat(responses): add default no-op sign_request to BaseResponsesAPIConfig"
```

---

## Task 2: Wire the signing hook into the responses handler (sync + async)

**Files:**
- Modify: `litellm/llms/custom_httpx/llm_http_handler.py` (`response_api_handler` ~2330-2378; `async_response_api_handler` ~2476-2526)
- Test: `tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py`:

```python
def _make_responses_handler_call(signed_body):
    """Drive BaseLLMHTTPHandler.response_api_handler with a fully mocked provider
    config + sync client, returning the kwargs the client.post was called with.

    signed_body=None simulates a no-op (non-signing) provider; bytes simulates a
    signing provider (e.g. Bedrock Mantle).
    """
    from unittest.mock import MagicMock
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    from litellm.types.router import GenericLiteLLMParams

    provider_config = MagicMock()
    provider_config.validate_environment.return_value = {}
    provider_config.get_complete_url.return_value = (
        "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
    )
    provider_config.transform_responses_api_request.return_value = {"input": "hi"}
    provider_config.should_fake_stream.return_value = False
    provider_config.sign_request.return_value = ({"X-Signed": "1"}, signed_body)

    mock_client = MagicMock()
    mock_client.post.return_value = MagicMock()

    handler = BaseLLMHTTPHandler()
    handler.response_api_handler(
        model="openai.gpt-5.5",
        input="hi",
        responses_api_provider_config=provider_config,
        response_api_optional_request_params={},
        custom_llm_provider="bedrock_mantle",
        litellm_params=GenericLiteLLMParams(aws_region_name="us-east-2"),
        logging_obj=MagicMock(),
        client=mock_client,
        _is_async=False,
    )
    return mock_client.post.call_args.kwargs


def test_responses_handler_sends_json_when_not_signed():
    """No-op provider (signed_body is None) -> handler posts json=data, no data= bytes."""
    kwargs = _make_responses_handler_call(signed_body=None)
    assert kwargs.get("json") == {"input": "hi"}
    assert "data" not in kwargs


def test_responses_handler_sends_signed_bytes_when_signed():
    """Signing provider -> handler posts the exact signed bytes via data=, not json=."""
    kwargs = _make_responses_handler_call(signed_body=b'{"input": "hi"}')
    assert kwargs.get("data") == b'{"input": "hi"}'
    assert "json" not in kwargs
    assert kwargs["headers"] == {"X-Signed": "1"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py -k "responses_handler_sends" -v`
Expected: FAIL — `test_responses_handler_sends_signed_bytes_when_signed` fails because the handler currently always posts `json=data` and never calls `sign_request` (so `data`/headers are wrong).

- [ ] **Step 3: Edit the sync handler**

In `litellm/llms/custom_httpx/llm_http_handler.py`, replace the `try:` block of `response_api_handler` that currently reads (starting ~line 2330):

```python
        try:
            if stream:
                # For streaming, use stream=True in the request
                if fake_stream is True:
                    stream, data = self._prepare_fake_stream_request(
                        stream=stream,
                        data=data,
                        fake_stream=fake_stream,
                    )

                response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    timeout=timeout
                    or float(response_api_optional_request_params.get("timeout", 0)),
                    stream=stream,
                )
                if fake_stream is True:
                    return MockResponsesAPIStreamingIterator(
                        response=response,
                        model=model,
                        logging_obj=logging_obj,
                        responses_api_provider_config=responses_api_provider_config,
                        litellm_metadata=litellm_metadata,
                        custom_llm_provider=custom_llm_provider,
                        request_data=request_context,
                        call_type=CallTypes.responses.value,
                    )

                return SyncResponsesAPIStreamingIterator(
                    response=response,
                    model=model,
                    logging_obj=logging_obj,
                    responses_api_provider_config=responses_api_provider_config,
                    litellm_metadata=litellm_metadata,
                    custom_llm_provider=custom_llm_provider,
                    request_data=request_context,
                    call_type=CallTypes.responses.value,
                )
            else:
                # For non-streaming requests
                response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    timeout=timeout
                    or float(response_api_optional_request_params.get("timeout", 0)),
                )
        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=responses_api_provider_config,
            )
```

with (note: fake-stream prep and `sign_request` are deliberately placed **before** the `try:`. Signing does no network I/O, and a `sign_request` failure such as the both-auth-missing `ValueError` must surface to the caller as-is, not be wrapped into a provider HTTP error by `_handle_error`. Only the `post` calls stay inside the `try:`):

```python
        is_stream_request = bool(stream)
        if is_stream_request and fake_stream is True:
            stream, data = self._prepare_fake_stream_request(
                stream=stream,
                data=data,
                fake_stream=fake_stream,
            )

        # Sign after the body is final (post-transform/normalize/extra_body and post
        # fake-stream prep) so signed bytes match what we send. No-op for providers
        # that inherit the default sign_request.
        headers, signed_body = responses_api_provider_config.sign_request(
            headers=headers,
            optional_params=dict(litellm_params),
            request_data=data,
            api_base=api_base,
            api_key=litellm_params.api_key,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )
        body_kwargs: Dict[str, Any] = (
            {"data": signed_body} if signed_body is not None else {"json": data}
        )

        try:
            if is_stream_request:
                response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    timeout=timeout
                    or float(response_api_optional_request_params.get("timeout", 0)),
                    stream=stream,
                    **body_kwargs,
                )
                if fake_stream is True:
                    return MockResponsesAPIStreamingIterator(
                        response=response,
                        model=model,
                        logging_obj=logging_obj,
                        responses_api_provider_config=responses_api_provider_config,
                        litellm_metadata=litellm_metadata,
                        custom_llm_provider=custom_llm_provider,
                        request_data=request_context,
                        call_type=CallTypes.responses.value,
                    )

                return SyncResponsesAPIStreamingIterator(
                    response=response,
                    model=model,
                    logging_obj=logging_obj,
                    responses_api_provider_config=responses_api_provider_config,
                    litellm_metadata=litellm_metadata,
                    custom_llm_provider=custom_llm_provider,
                    request_data=request_context,
                    call_type=CallTypes.responses.value,
                )
            else:
                response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    timeout=timeout
                    or float(response_api_optional_request_params.get("timeout", 0)),
                    **body_kwargs,
                )
        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=responses_api_provider_config,
            )
```

Note this also moves the `logging_obj.pre_call(...)` block: it currently sits between `data` finalization and the `try:`. Keep `pre_call` after the `sign_request` block (so it logs the final signed headers) and immediately before the `try:`.

- [ ] **Step 4: Edit the async handler identically**

In `async_response_api_handler`, apply the same transformation to its `try:` block (~line 2476-2526). It is structurally identical to the sync block except it `await`s `async_httpx_client.post(...)` and returns `ResponsesAPIStreamingIterator` (not `SyncResponsesAPIStreamingIterator`). Replace its body with:

```python
        is_stream_request = bool(stream)
        if is_stream_request and fake_stream is True:
            stream, data = self._prepare_fake_stream_request(
                stream=stream,
                data=data,
                fake_stream=fake_stream,
            )

        headers, signed_body = responses_api_provider_config.sign_request(
            headers=headers,
            optional_params=dict(litellm_params),
            request_data=data,
            api_base=api_base,
            api_key=litellm_params.api_key,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )
        body_kwargs: Dict[str, Any] = (
            {"data": signed_body} if signed_body is not None else {"json": data}
        )

        try:
            if is_stream_request:
                response = await async_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    timeout=timeout
                    or float(response_api_optional_request_params.get("timeout", 0)),
                    stream=stream,
                    **body_kwargs,
                )

                if fake_stream is True:
                    return MockResponsesAPIStreamingIterator(
                        response=response,
                        model=model,
                        logging_obj=logging_obj,
                        responses_api_provider_config=responses_api_provider_config,
                        litellm_metadata=litellm_metadata,
                        custom_llm_provider=custom_llm_provider,
                        request_data=request_context,
                        call_type=CallTypes.responses.value,
                    )

                return ResponsesAPIStreamingIterator(
                    response=response,
                    model=model,
                    logging_obj=logging_obj,
                    responses_api_provider_config=responses_api_provider_config,
                    litellm_metadata=litellm_metadata,
                    custom_llm_provider=custom_llm_provider,
                    request_data=request_context,
                    call_type=CallTypes.responses.value,
                )
            else:
                response = await async_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    timeout=timeout
                    or float(response_api_optional_request_params.get("timeout", 0)),
                    **body_kwargs,
                )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=responses_api_provider_config,
            )
```

Same placement note as the sync handler: keep `logging_obj.pre_call(...)` after the `sign_request` block and just before the `try:`; only the `await ...post(...)` calls stay inside the `try:`.

- [ ] **Step 5: Run the wiring tests + the existing responses handler tests**

Run: `python -m pytest tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py -k "responses" -v`
Expected: PASS for the two new wiring tests and all pre-existing responses tests in that file (regression check that no-op providers are unchanged).

- [ ] **Step 6: Commit**

```bash
git add litellm/llms/custom_httpx/llm_http_handler.py tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py
git commit -m "feat(responses): call sign_request after body is final, send signed bytes when signed"
```

---

## Task 3: Mantle config — compose `BaseAWSLLM`, override `sign_request`, relax `validate_environment`

**Files:**
- Modify: `litellm/llms/bedrock_mantle/responses/transformation.py`
- Test: `tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py`

### 3a. Compose the signer + override `sign_request` (bearer short-circuit path)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py` a new test class. The bearer test injects a signer whose `get_credentials` is a spy, proving the bearer path never touches credential resolution:

```python
class TestBedrockMantleResponsesSigV4:
    def test_bearer_short_circuits_without_credentials(self, monkeypatch):
        from unittest.mock import MagicMock
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)

        signer = BaseAWSLLM()
        signer.get_credentials = MagicMock(
            side_effect=AssertionError("get_credentials must not run for bearer auth")
        )
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=signer)

        headers, signed_body = cfg.sign_request(
            headers={},
            optional_params={},
            request_data={"input": "hi"},
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            api_key="bearer-from-config",
        )
        assert headers["Authorization"] == "Bearer bearer-from-config"
        assert signed_body == b'{"input": "hi"}'
        signer.get_credentials.assert_not_called()

    def test_bearer_resolved_from_mantle_env_key(self, monkeypatch):
        from unittest.mock import MagicMock
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "env-bearer")

        signer = BaseAWSLLM()
        signer.get_credentials = MagicMock(
            side_effect=AssertionError("get_credentials must not run for bearer auth")
        )
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=signer)

        headers, _ = cfg.sign_request(
            headers={},
            optional_params={},
            request_data={"input": "hi"},
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            api_key=None,
        )
        assert headers["Authorization"] == "Bearer env-bearer"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py::TestBedrockMantleResponsesSigV4 -v`
Expected: FAIL — `BedrockMantleResponsesAPIConfig.__init__() got an unexpected keyword argument 'aws_signer'` (and no `sign_request` override).

- [ ] **Step 3: Implement composition + `sign_request`**

Edit `litellm/llms/bedrock_mantle/responses/transformation.py`. Update imports (top of file) to add the signer, error type, `Tuple`, and `re` (used by the region resolver to read an embedded region out of an explicit mantle base):

```python
import re
from typing import Optional, Tuple

from botocore.exceptions import NoCredentialsError

from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
```

Add a module-level regex (near `_BASE_SUFFIXES_TO_STRIP`) that matches the standard Mantle host and captures its region segment:

```python
# Standard Mantle host: https://bedrock-mantle.<region>.api.aws (group 1 = region).
_MANTLE_HOST_RE = re.compile(r"^https?://bedrock-mantle\.([^/.]+)\.api\.aws", re.IGNORECASE)
```

Add an `__init__` (just after the class line `class BedrockMantleResponsesAPIConfig(OpenAIResponsesAPIConfig):`, before the `custom_llm_provider` property) that injects the signer for testability:

```python
    def __init__(self, aws_signer: Optional[BaseAWSLLM] = None):
        super().__init__()
        self._aws_signer = aws_signer or BaseAWSLLM()
```

Add a single region resolver so the URL host and the SigV4 credential scope can never diverge (adversarial self-review fix; see the "Region single source of truth" key fact). Place it as a static method on the class. Precedence: explicit `aws_region_name` first (this is also what the signer's `_get_aws_region_name` reads first), then a region embedded in an explicit Mantle `api_base`, then the region env vars, then the default:

```python
    @staticmethod
    def _resolve_region(params: dict) -> str:
        region = params.get("aws_region_name")
        if region:
            return region
        base = params.get("api_base") or get_secret_str("BEDROCK_MANTLE_API_BASE")
        if base:
            match = _MANTLE_HOST_RE.match(base.rstrip("/"))
            if match:
                return match.group(1)
        return (
            get_secret_str("BEDROCK_MANTLE_REGION")
            or get_secret_str("AWS_REGION_NAME")
            or get_secret_str("AWS_REGION")
            or BEDROCK_MANTLE_DEFAULT_REGION
        )
```

Rewrite `get_complete_url` so the URL host always derives from the single resolved region for standard Mantle hosts (closing the `dynamic_api_base` injection hole, see key fact), while genuinely custom proxy hosts are preserved. Replace the current method body with:

```python
    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        region = self._resolve_region({**litellm_params, "api_base": api_base})
        base = (
            api_base
            or get_secret_str("BEDROCK_MANTLE_API_BASE")
            or f"https://bedrock-mantle.{region}.api.aws"
        )
        base = base.rstrip("/")
        for suffix in _BASE_SUFFIXES_TO_STRIP:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        # For the standard Mantle host (including the default-region base that
        # responses/main.py auto-injects into litellm_params.api_base), pin to the
        # single resolved region so aws_region_name wins; preserve custom proxy hosts.
        if _MANTLE_HOST_RE.match(base):
            base = f"https://bedrock-mantle.{region}.api.aws"
        return f"{base}/openai/v1/responses"
```

Add the `sign_request` override at the end of the class (after `supports_native_websocket`). It (1) resolves the Mantle bearer chain; (2) for the SigV4 path, pins `optional_params["aws_region_name"]` to the same resolved region the URL used and strips any caller-supplied `Authorization` so the signer's "restore original Authorization" step cannot clobber the SigV4 header; (3) forwards to the shared `_sign_request`; (4) converts the unhelpful no-credentials error into a message naming both auth paths:

```python
    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        bearer = (
            api_key
            or get_secret_str("BEDROCK_MANTLE_API_KEY")
            or get_secret_str("AWS_BEARER_TOKEN_BEDROCK")
        )
        if not bearer:
            # SigV4 path. Pin the credential-scope region to the region of the actual
            # signing URL (api_base, already region-resolved by get_complete_url) so the
            # SigV4 scope and the URL host can never disagree. Resolve from api_base first,
            # then fall back to the regular precedence. Also drop any caller Authorization
            # so _sign_request's restore-original-Authorization step cannot override the
            # SigV4 header.
            optional_params = {
                **optional_params,
                "aws_region_name": self._resolve_region(
                    {**optional_params, "api_base": api_base}
                ),
            }
            headers = {
                k: v for k, v in headers.items() if k.lower() != "authorization"
            }
        try:
            return self._aws_signer._sign_request(
                service_name="bedrock",
                headers=headers,
                optional_params=optional_params,
                request_data=request_data,
                api_base=api_base,
                api_key=bearer,
                model=model,
                stream=stream,
                fake_stream=fake_stream,
            )
        except NoCredentialsError as e:
            raise ValueError(
                "Bedrock Mantle auth failed: no Bearer token and no usable AWS "
                "credentials. Set BEDROCK_MANTLE_API_KEY (or AWS_BEARER_TOKEN_BEDROCK) "
                "or pass api_key for Bearer auth, or provide AWS credentials "
                "(IAM role / access key / profile / web identity) for SigV4."
            ) from e
```

Note `_resolve_region` is fed `optional_params` here; in the handler the signer receives `optional_params=dict(litellm_params)`, so `aws_region_name` (and the same env fallbacks) are visible to both the URL builder and the signer. The URL builder is passed `litellm_params` directly, so both call sites resolve identically.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py::TestBedrockMantleResponsesSigV4 -v`
Expected: PASS for both bearer tests.

- [ ] **Step 5: Commit**

```bash
git add litellm/llms/bedrock_mantle/responses/transformation.py tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py
git commit -m "feat(bedrock_mantle): add SigV4 sign_request via composed BaseAWSLLM (bearer path)"
```

### 3b. SigV4 path: access key, AssumeRole, body-byte consistency, region

- [ ] **Step 1: Write the failing tests**

Append to class `TestBedrockMantleResponsesSigV4` in the Mantle test file:

```python
    def test_access_key_produces_sigv4_headers(self, monkeypatch):
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)

        cfg = BedrockMantleResponsesAPIConfig(aws_signer=BaseAWSLLM())
        headers, signed_body = cfg.sign_request(
            headers={},
            optional_params={
                "aws_access_key_id": "AKIAEXAMPLE",
                "aws_secret_access_key": "c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
                "aws_session_token": "session-token-test",
                "aws_region_name": "us-east-2",
            },
            request_data={"input": "hi"},
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            api_key=None,
        )
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "Credential=AKIAEXAMPLE/" in headers["Authorization"]
        assert "/us-east-2/bedrock/aws4_request" in headers["Authorization"]
        assert "X-Amz-Date" in headers
        assert headers["X-Amz-Security-Token"] == "session-token-test"
        assert signed_body == b'{"input": "hi"}'

    def test_assume_role_path_produces_sigv4_headers(self, monkeypatch):
        from unittest.mock import MagicMock
        from botocore.credentials import Credentials
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)

        signer = BaseAWSLLM()
        signer.get_credentials = MagicMock(
            return_value=Credentials(
                access_key="ASIAEXAMPLE",
                secret_key="YXNzdW1lZC1yb2xlLXNlY3JldC1hc3N1bWVk",
                token="assumed-session-token",
            )
        )
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=signer)

        headers, _ = cfg.sign_request(
            headers={},
            optional_params={
                "aws_role_name": "arn:aws:iam::000000000000:role/test-role",
                "aws_session_name": "litellm-test",
                "aws_region_name": "us-east-2",
            },
            request_data={"input": "hi"},
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            api_key=None,
        )
        signer.get_credentials.assert_called_once()
        call = signer.get_credentials.call_args.kwargs
        assert call["aws_role_name"] == "arn:aws:iam::000000000000:role/test-role"
        assert call["aws_session_name"] == "litellm-test"
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "/us-east-2/bedrock/aws4_request" in headers["Authorization"]

    def test_signed_body_matches_final_data_after_normalize(self, monkeypatch):
        """Core regression: the signed bytes must equal the bytes actually sent.

        Sign the *final* data dict and assert the returned signed_body decodes to
        exactly that dict, so a later change to the data would break the SigV4 hash.
        """
        import json
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)

        final_data = {"model": "openai.gpt-5.5", "input": "hi", "max_output_tokens": 16}
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=BaseAWSLLM())
        _, signed_body = cfg.sign_request(
            headers={},
            optional_params={
                "aws_access_key_id": "AKIAEXAMPLE",
                "aws_secret_access_key": "c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
                "aws_region_name": "us-east-2",
            },
            request_data=final_data,
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            api_key=None,
        )
        assert signed_body is not None
        assert json.loads(signed_body) == final_data

    def test_region_comes_from_optional_params(self, monkeypatch):
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_REGION_NAME", raising=False)

        cfg = BedrockMantleResponsesAPIConfig(aws_signer=BaseAWSLLM())
        headers, _ = cfg.sign_request(
            headers={},
            optional_params={
                "aws_access_key_id": "AKIAEXAMPLE",
                "aws_secret_access_key": "c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
                "aws_region_name": "eu-west-1",
            },
            request_data={"input": "hi"},
            api_base="https://bedrock-mantle.eu-west-1.api.aws/openai/v1/responses",
            api_key=None,
        )
        assert "/eu-west-1/bedrock/aws4_request" in headers["Authorization"]

    def test_url_region_and_sigv4_region_agree_from_litellm_params(self, monkeypatch):
        """Adversarial-review regression: a caller-supplied aws_region_name (no region
        env set) must shape BOTH the URL host and the SigV4 credential scope, or the
        request is signed for one region and sent to another -> 401.
        """
        monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_REGION_NAME", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)

        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        params = {
            "aws_region_name": "ap-southeast-2",
            "aws_access_key_id": "AKIAEXAMPLE",
            "aws_secret_access_key": "c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
        }
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=BaseAWSLLM())
        url = cfg.get_complete_url(api_base=None, litellm_params=params)
        assert url == "https://bedrock-mantle.ap-southeast-2.api.aws/openai/v1/responses"

        headers, _ = cfg.sign_request(
            headers={},
            optional_params=params,
            request_data={"input": "hi"},
            api_base=url,
            api_key=None,
        )
        assert "/ap-southeast-2/bedrock/aws4_request" in headers["Authorization"]

    def test_injected_default_region_base_does_not_override_aws_region_name(
        self, monkeypatch
    ):
        """2nd-round adversarial regression: responses/main.py auto-injects
        litellm_params.api_base = https://bedrock-mantle.<DEFAULT>.api.aws/v1 (default
        region, ignoring aws_region_name). The config must still pin BOTH the URL host
        and the SigV4 scope to aws_region_name, or the IAM deployment 401s. A naive
        'resolve region only when api_base is None' fix would fail this test.
        """
        monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_REGION_NAME", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)

        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        injected_base = "https://bedrock-mantle.us-east-1.api.aws/v1"  # default region
        params = {
            "aws_region_name": "us-east-2",  # what the caller actually wants
            "api_base": injected_base,
            "aws_access_key_id": "AKIAEXAMPLE",
            "aws_secret_access_key": "c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
        }
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=BaseAWSLLM())
        url = cfg.get_complete_url(api_base=injected_base, litellm_params=params)
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"

        headers, _ = cfg.sign_request(
            headers={},
            optional_params=params,
            request_data={"input": "hi"},
            api_base=url,
            api_key=None,
        )
        assert "/us-east-2/bedrock/aws4_request" in headers["Authorization"]
        assert "us-east-1" not in headers["Authorization"]

    def test_custom_proxy_host_is_preserved(self, monkeypatch):
        """A genuinely custom (non-Mantle) api_base host must be preserved, not rewritten
        to a bedrock-mantle host. Only standard Mantle hosts are region-pinned.
        """
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="https://mantle-proxy.internal.example/openai/v1",
            litellm_params={"aws_region_name": "us-east-2"},
        )
        assert url == "https://mantle-proxy.internal.example/openai/v1/responses"

    def test_caller_authorization_does_not_override_sigv4(self, monkeypatch):
        """Adversarial-review regression: a caller-supplied Authorization header (e.g.
        from extra_headers, surviving the relaxed validate_environment) must not clobber
        the SigV4 Authorization that _sign_request would otherwise restore.
        """
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)

        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        cfg = BedrockMantleResponsesAPIConfig(aws_signer=BaseAWSLLM())
        headers, _ = cfg.sign_request(
            headers={"Authorization": "Bearer stale-caller-token"},
            optional_params={
                "aws_access_key_id": "AKIAEXAMPLE",
                "aws_secret_access_key": "c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
                "aws_region_name": "us-east-2",
            },
            request_data={"input": "hi"},
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            api_key=None,
        )
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "Bearer stale-caller-token" not in headers["Authorization"]
```

- [ ] **Step 2: Run to verify they pass**

These exercise the code from 3a (the `sign_request` override, the `_resolve_region` helper, and the `get_complete_url` region change). Run:
`python -m pytest tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py::TestBedrockMantleResponsesSigV4 -v`
Expected: PASS. If `test_access_key_produces_sigv4_headers` fails on header casing, inspect the real header keys it produced and adjust the assertions to match botocore's output (do not change product code).

- [ ] **Step 3: Commit**

```bash
git add tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py
git commit -m "test(bedrock_mantle): cover SigV4 access-key, AssumeRole, body bytes, region/auth consistency"
```

### 3c. Relax `validate_environment` + both-auth-missing error message

- [ ] **Step 1: Write/adjust the failing tests**

The existing `test_missing_key_raises` (lines 117-126) asserts `validate_environment` raises when no bearer. Under the new design `validate_environment` must NOT raise (SigV4 may still apply); the both-paths error now surfaces from `sign_request`. Replace that test and add the message test:

Replace `test_missing_key_raises` with:

```python
    def test_missing_bearer_does_not_raise_in_validate_environment(self, monkeypatch):
        # SigV4 may still apply, so validate_environment must defer instead of raising.
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={}, model="openai.gpt-5.5", litellm_params=GenericLiteLLMParams()
        )
        assert "Authorization" not in headers
```

Add to `TestBedrockMantleResponsesSigV4`:

```python
    def test_no_bearer_and_no_credentials_raises_both_paths(self, monkeypatch):
        from unittest.mock import MagicMock
        from botocore.exceptions import NoCredentialsError
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

        signer = BaseAWSLLM()
        signer.get_credentials = MagicMock(side_effect=NoCredentialsError())
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=signer)

        with pytest.raises(ValueError) as exc:
            cfg.sign_request(
                headers={},
                optional_params={"aws_region_name": "us-east-2"},
                request_data={"input": "hi"},
                api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
                api_key=None,
            )
        msg = str(exc.value)
        assert "Bearer" in msg
        assert "SigV4" in msg or "IAM" in msg
```

- [ ] **Step 2: Run to verify the new validate test fails (old behavior still raises)**

Run: `python -m pytest tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py -k "validate_environment or both_paths" -v`
Expected: `test_missing_bearer_does_not_raise_in_validate_environment` FAILS (current code still raises `ValueError`). `test_no_bearer_and_no_credentials_raises_both_paths` already PASSES (covered by 3a's `sign_request`).

- [ ] **Step 3: Relax `validate_environment`**

In `litellm/llms/bedrock_mantle/responses/transformation.py`, replace the current `validate_environment` body (lines ~60-75) so it sets the Bearer header only when a bearer exists and no longer raises when absent:

```python
    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            litellm_params.api_key
            or get_secret_str("BEDROCK_MANTLE_API_KEY")
            or get_secret_str("AWS_BEARER_TOKEN_BEDROCK")
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py -k "validate_environment or both_paths or env_key or bearer_token_fallback or api_key_takes_priority" -v`
Expected: PASS (the three pre-existing bearer-present tests still pass; the relaxed missing-key test passes; the both-paths error test passes).

- [ ] **Step 5: Commit**

```bash
git add litellm/llms/bedrock_mantle/responses/transformation.py tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py
git commit -m "feat(bedrock_mantle): defer auth to sign_request; validate_environment no longer requires bearer"
```

---

## Task 4: Update the Mantle module docstring

**Files:**
- Modify: `litellm/llms/bedrock_mantle/responses/transformation.py` (module docstring, lines 1-11)

- [ ] **Step 1: Replace the docstring**

The current docstring says auth is "NOT SigV4". Update lines 1-11 to reflect both paths:

```python
"""
Amazon Bedrock Mantle - Responses API backend.

gpt-5.5 / gpt-5.4 on Mantle are exposed ONLY on the `/openai/v1/responses`
path (not the standard `/v1/responses`). Payloads and SSE follow the OpenAI
Responses spec, so this config inherits OpenAIResponsesAPIConfig and overrides
only the endpoint URL and authentication.

Auth: Bearer token (BEDROCK_MANTLE_API_KEY or the standard
AWS_BEARER_TOKEN_BEDROCK, or litellm_params.api_key) when present; otherwise
AWS SigV4 (service name "bedrock") using the standard credential chain (IAM
role / access key / profile / web identity), signed via the shared
BaseAWSLLM._sign_request after the request body is finalized.
"""
```

- [ ] **Step 2: Verify no behavior change**

Run: `python -m pytest tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py -v`
Expected: PASS (docstring-only change).

- [ ] **Step 3: Commit**

```bash
git add litellm/llms/bedrock_mantle/responses/transformation.py
git commit -m "docs(bedrock_mantle): document SigV4 + Bearer auth on Responses route"
```

---

## Task 5: Full-suite check, format, lint

**Files:** none (verification + tooling)

- [ ] **Step 1: Run the full affected test set**

```bash
python -m pytest \
  tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py \
  tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py \
  -v
```
Expected: all PASS (new + pre-existing).

- [ ] **Step 2: Guard against regressions in other responses providers**

```bash
python -m pytest tests/test_litellm/llms/ -k "responses" -q
```
Expected: PASS. These exercise OpenAI/Azure/etc. responses configs, confirming the default no-op `sign_request` and the handler change did not alter their behavior.

- [ ] **Step 3: Format and lint (repo standard, per CLAUDE.md)**

```bash
make format
ruff check litellm/llms/bedrock_mantle/responses/transformation.py \
  litellm/llms/base_llm/responses/transformation.py \
  litellm/llms/custom_httpx/llm_http_handler.py
```
Expected: no errors. If `make format` reformats files, re-run Step 1.

- [ ] **Step 4: Commit any formatting changes**

Stage only the files this plan touches (never `git add -A`, which could sweep in unrelated worktree changes):

```bash
git add \
  litellm/llms/base_llm/responses/transformation.py \
  litellm/llms/custom_httpx/llm_http_handler.py \
  litellm/llms/bedrock_mantle/responses/transformation.py \
  tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py \
  tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py
git diff --cached --quiet || git commit -m "chore: format and lint mantle responses sigv4 changes"
```

---

## Task 6: PR (after live Proof of Fix is captured by the human)

This task is performed by the human/operator, not automated, because the Proof of Fix must hit the real Mantle endpoint from an EC2 IAM-role host (per CLAUDE.md: live curl, real provider, no pytest screenshots).

- [ ] **Step 1: Push the branch**

```bash
git push -u fork litellm_bedrock_mantle_responses_sigv4
```

- [ ] **Step 2: Capture Proof of Fix on EC2 (us-east-2, IAM role, no bearer key set)**

Start a proxy with a config containing `bedrock_mantle/openai.gpt-5.5` and no `api_key`/`BEDROCK_MANTLE_API_KEY`/`AWS_BEARER_TOKEN_BEDROCK`:

```bash
python litellm/proxy/proxy_cli.py --config <config>.yaml --detailed_debug --reload --use_v2_migration_resolver 2>&1 | tee litellm.log
```

```bash
curl -s http://localhost:4000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"model": "gpt-5.5", "input": "ping"}'
```

Expected: HTTP 200; in `litellm.log`, outbound goes to `.../openai/v1/responses` and the outbound `Authorization` is `AWS4-HMAC-SHA256 ...` (no outbound `Bearer`). Capture the curl command and output, and the relevant log lines. Repeat with a bearer key set as the control case (expect 200 with an outbound `Bearer` header).

- [ ] **Step 3: Open the PR**

Base = `litellm_oss_staging_040626` (not `main`, not `litellm_internal_staging`; the Responses route prerequisite lives only on the oss staging branch). Use `.github/pull_request_template.md`. Relevant issues: Fixes #29665, #29463. Type: New Feature. In Changes, describe in prose (no emojis, no em-dashes, no "not X but Y", minimal lists per CLAUDE.md) that the Mantle Responses config now signs with SigV4 when no bearer is present by reusing `BaseAWSLLM._sign_request` through composition, that a default no-op `sign_request` was added to the responses base config, and that the responses handler now sends the signed bytes; note that existing responses providers are unaffected. Paste the Step 2 Proof of Fix. Do not include any real keys, secrets, or role ARNs.

---

## Self-Review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §5/§6.1 base no-op `sign_request` | Task 1 |
| §6.2 handler hook, sync + async, streaming covered, signed-body selection | Task 2 |
| §6.3 Mantle override via composed `BaseAWSLLM`, mirrors `AmazonInvokeConfig.sign_request` | Task 3a |
| §7 auth coexistence (bearer-first inside `_sign_request`; `validate_environment` defers) | Task 3a + 3c |
| §7 region resolution from optional_params | Task 3b `test_region_comes_from_optional_params` |
| §9.1 bearer short-circuit, no get_credentials | Task 3a `test_bearer_short_circuits_without_credentials` |
| §9.2 access-key SigV4 headers | Task 3b `test_access_key_produces_sigv4_headers` |
| §9.3 AssumeRole SigV4 | Task 3b `test_assume_role_path_produces_sigv4_headers` |
| §9.4 body-byte consistency | Task 3b `test_signed_body_matches_final_data_after_normalize` |
| §9.5 region + URL unchanged | Task 3b + existing URL tests (unchanged) |
| §9.6 both-missing error names both paths | Task 3c `test_no_bearer_and_no_credentials_raises_both_paths` |
| §9.7 handler regression: no-op stays json | Task 2 `test_responses_handler_sends_json_when_not_signed` |
| Region single-source-of-truth (URL host region == SigV4 scope region) | Task 3a `_resolve_region` + `get_complete_url` rewrite; Task 3b `test_url_region_and_sigv4_region_agree_from_litellm_params` |
| Region: injected default-region `api_base` cannot override `aws_region_name` (2nd-round) | Task 3a `_MANTLE_HOST_RE` pin in `get_complete_url`; Task 3b `test_injected_default_region_base_does_not_override_aws_region_name` |
| Region: custom proxy host preserved | Task 3b `test_custom_proxy_host_is_preserved` |
| Caller `Authorization` cannot clobber SigV4 | Task 3a SigV4-branch strip; Task 3b `test_caller_authorization_does_not_override_sigv4` |
| both-auth `ValueError` surfaces cleanly, not wrapped as 500 (2nd-round) | Task 2 `sign_request` placed before the handler `try:` |
| §8 no impact on other providers | Task 5 Step 2 |
| §10 live Proof of Fix | Task 6 Step 2 |
| §11 branch/PR/base/secrets | Task 6 Step 3 |

No gaps.

**Adversarial-review disposition.**

First round (Codex): two real findings, fixed — region divergence between `get_complete_url` and signing (`_resolve_region`), and caller `Authorization` overriding the SigV4 header (SigV4-branch strip). Two hygiene items applied — fake AWS keys use the repo `AKIAEXAMPLE`/`ASIAEXAMPLE` convention; Task 5 stages explicit paths instead of `git add -A`. Non-applicable findings: the "79-commit/372-file" target inconsistency and "code still Bearer-only" were artifacts of the reviewer diffing against `litellm_internal_staging` (actual branch delta is the docs, and this reviews a plan, not merged code); "unsigned by-id subroutes" does not hold because those routes call the registry with `model=None`, which returns `None` for Mantle; the local-hook/`--no-verify`/secret-scan-on-commit concerns do not apply (no local git hooks; secret scan is CI-only via ggshield, and `test_no_hardcoded_secrets.py` only matches `Basic <base64>` strings).

Second round (self-review, after Codex runtime kept failing on its own upstream Mantle endpoint): two further real findings, fixed. (1) **Deeper region hole, verified by running `litellm.get_llm_provider`:** `responses/main.py:688-691` injects a default-region `dynamic_api_base` into `litellm_params.api_base` before the config runs, which defeated the first-round `_resolve_region` fix (it only resolved region when `api_base` was None). `get_complete_url` now pins standard Mantle hosts to the single resolved region regardless of an incoming base, validated green against every existing URL test plus the injected-base and aws_region_name-only cases. (2) **Error wrapping:** the both-auth-missing `ValueError` would have been rewrapped as a 500 by `_handle_error` had `sign_request` stayed inside the handler `try:`; it now runs before the `try:`.

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to". Every code step shows full code; every test step shows full test bodies.

**3. Type consistency:** `sign_request` signature `(headers, optional_params, request_data, api_base, api_key=None, model=None, stream=None, fake_stream=None) -> Tuple[dict, Optional[bytes]]` is identical across the base no-op (Task 1), the Mantle override (Task 3a), and the handler call site (Task 2). `_aws_signer` is set in `__init__` (3a) and used only in `sign_request` (3a). `body_kwargs` defined and used within each handler branch (Task 2). `aws_signer` constructor kwarg matches across all `BedrockMantleResponsesAPIConfig(aws_signer=...)` test usages. Handler passes `optional_params=dict(litellm_params)` and `api_key=litellm_params.api_key`, both verified to exist on `GenericLiteLLMParams`.
