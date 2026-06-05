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

with:

```python
        try:
            is_stream_request = bool(stream)
            if is_stream_request and fake_stream is True:
                stream, data = self._prepare_fake_stream_request(
                    stream=stream,
                    data=data,
                    fake_stream=fake_stream,
                )

            # Sign after the body is final (post-transform/normalize/extra_body and
            # post fake-stream prep) so signed bytes match what we send. No-op for
            # providers that inherit the default sign_request.
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

- [ ] **Step 4: Edit the async handler identically**

In `async_response_api_handler`, apply the same transformation to its `try:` block (~line 2476-2526). It is structurally identical to the sync block except it `await`s `async_httpx_client.post(...)` and returns `ResponsesAPIStreamingIterator` (not `SyncResponsesAPIStreamingIterator`). Replace its body with:

```python
        try:
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

Edit `litellm/llms/bedrock_mantle/responses/transformation.py`. Update imports (top of file) to add the signer, error type, and `Tuple`:

```python
from typing import Optional, Tuple

from botocore.exceptions import NoCredentialsError

from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
```

Add an `__init__` (just after the class line `class BedrockMantleResponsesAPIConfig(OpenAIResponsesAPIConfig):`, before the `custom_llm_provider` property) that injects the signer for testability:

```python
    def __init__(self, aws_signer: Optional[BaseAWSLLM] = None):
        super().__init__()
        self._aws_signer = aws_signer or BaseAWSLLM()
```

Add the `sign_request` override at the end of the class (after `supports_native_websocket`). It resolves the Mantle bearer chain, forwards to the shared `_sign_request`, and converts the unhelpful no-credentials error into a message naming both auth paths:

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
                "aws_access_key_id": "AKIATESTTESTTESTTEST",
                "aws_secret_access_key": "c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
                "aws_session_token": "session-token-test",
                "aws_region_name": "us-east-2",
            },
            request_data={"input": "hi"},
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            api_key=None,
        )
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "Credential=AKIATESTTESTTESTTEST/" in headers["Authorization"]
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
                access_key="ASIAASSUMEDROLEKEY00",
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
                "aws_access_key_id": "AKIATESTTESTTESTTEST",
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
                "aws_access_key_id": "AKIATESTTESTTESTTEST",
                "aws_secret_access_key": "c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
                "aws_region_name": "eu-west-1",
            },
            request_data={"input": "hi"},
            api_base="https://bedrock-mantle.eu-west-1.api.aws/openai/v1/responses",
            api_key=None,
        )
        assert "/eu-west-1/bedrock/aws4_request" in headers["Authorization"]
```

- [ ] **Step 2: Run to verify they pass**

These exercise the code from 3a (no new product code needed). Run:
`python -m pytest tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py::TestBedrockMantleResponsesSigV4 -v`
Expected: PASS. If `test_access_key_produces_sigv4_headers` fails on header casing, inspect the real header keys it produced and adjust the assertions to match botocore's output (do not change product code).

- [ ] **Step 3: Commit**

```bash
git add tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py
git commit -m "test(bedrock_mantle): cover SigV4 access-key, AssumeRole, body-byte consistency, region"
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

```bash
git add -A
git commit -m "chore: format and lint mantle responses sigv4 changes" || echo "nothing to format"
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
| §8 no impact on other providers | Task 5 Step 2 |
| §10 live Proof of Fix | Task 6 Step 2 |
| §11 branch/PR/base/secrets | Task 6 Step 3 |

No gaps.

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to". Every code step shows full code; every test step shows full test bodies.

**3. Type consistency:** `sign_request` signature `(headers, optional_params, request_data, api_base, api_key=None, model=None, stream=None, fake_stream=None) -> Tuple[dict, Optional[bytes]]` is identical across the base no-op (Task 1), the Mantle override (Task 3a), and the handler call site (Task 2). `_aws_signer` is set in `__init__` (3a) and used only in `sign_request` (3a). `body_kwargs` defined and used within each handler branch (Task 2). `aws_signer` constructor kwarg matches across all `BedrockMantleResponsesAPIConfig(aws_signer=...)` test usages. Handler passes `optional_params=dict(litellm_params)` and `api_key=litellm_params.api_key`, both verified to exist on `GenericLiteLLMParams`.
