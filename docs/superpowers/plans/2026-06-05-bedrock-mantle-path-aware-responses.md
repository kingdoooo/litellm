# bedrock_mantle Path-Aware Responses Routing + `/v1/responses` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LiteLLM route Bedrock Mantle Responses requests to the correct upstream path per model (`/openai/v1/responses` for OpenAI gpt-frontier models, `/v1/responses` for any model declared `mode: responses`), adding support for the previously-unsupported `/v1/responses` path.

**Architecture:** The path decision lives in the gate (`ProviderConfigManager._get_python_responses_api_config`), which has the `model` argument; it constructs `BedrockMantleResponsesAPIConfig` with a `use_openai_path` flag. The config's `get_complete_url` (which does not receive `model`) picks its URL prefix from that injected flag. This mirrors how Azure's gate returns different responses configs per model and how Azure's config builds its URL from injected params rather than hardcoding.

**Tech Stack:** Python, pytest. Files in `litellm/utils.py`, `litellm/llms/bedrock_mantle/responses/transformation.py`, and the existing test file `tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py`.

**Spec:** `docs/superpowers/specs/2026-06-05-bedrock-mantle-path-aware-responses-design.md`

**Branch:** `litellm_bedrock_mantle_v1_responses` (already cut from `litellm_oss_staging_040626`; PR base will be `litellm_oss_staging_040626`).

---

## Background an implementer needs

Bedrock Mantle exposes Responses on two distinct upstream paths (confirmed against AWS docs):

- `/openai/v1/responses` — only OpenAI closed frontier gpt models (`openai.gpt-5.5`, `openai.gpt-5.4`, future `gpt-6`). base URL form `.../openai/v1`.
- `/v1/responses` — the standard path; today only `openai.gpt-oss-120b` / `openai.gpt-oss-20b`. base URL form `.../v1`.
- Everything else OpenAI-compatible (nvidia, qwen, mistral, google, zai, ...) is chat-only and 400s on either responses path; those return `None` from the gate and fall back to chat-completions emulation.

`get_model_info(model).get("mode") == "responses"` tells you a model supports Responses, but it does NOT tell you which path. Path is decided by "is this a gpt-frontier model": gpt-frontier → `/openai/v1/responses`, otherwise (any other declared-responses model) → `/v1/responses`.

`gpt-oss-120b/20b` are `mode: chat` in the price map, so by default they are NOT caught by the declared-responses branch and keep emulation. A user opts them into native `/v1/responses` by setting `model_info: {mode: responses}` in proxy config (loaded via `register_model` into the global `litellm.model_cost`).

Key constraint: the responses-flavor `get_complete_url(self, api_base, litellm_params)` does NOT receive `model`, so the config cannot pick its path from the model name. The gate, which has `model`, must inject the decision.

Current gate branch lives at `litellm/utils.py:8912-8922`. Current config `get_complete_url` is at `litellm/llms/bedrock_mantle/responses/transformation.py:38-58` and hardcodes the trailing `/openai/v1/responses`.

---

## File Structure

- **Modify** `litellm/llms/bedrock_mantle/responses/transformation.py` — add `__init__(self, use_openai_path: bool = True)` and make `get_complete_url` pick the trailing path from `self.use_openai_path`. (Task 1, 2)
- **Modify** `litellm/utils.py` — rewrite the BEDROCK_MANTLE branch in `_get_python_responses_api_config` to the three-branch path-aware gate. (Task 3)
- **Modify** `tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py` — extend existing `TestBedrockMantleResponsesURL`, `TestBedrockMantleResponsesAuth`, and `TestBedrockMantleResponsesRegistry` classes; add a `TestBedrockMantleResponsesRequestBody` class; add a `restore_model_cost` fixture (deepcopy snapshot, distinct from the shallow `local_cost_map` fixture which only works because it reassigns the whole map). (Tasks 1-4)

No new files. No new instantiation sites beyond the one at `utils.py:8920` (the only place the class is constructed). The constructor param defaults to `True`, so existing call sites and the lazy-import surface (`litellm/__init__.py:1743`, `litellm/_lazy_imports_registry.py`) are unaffected.

---

### Task 1: Config grows a `use_openai_path` flag and standard-path URL (config layer)

**Files:**
- Modify: `litellm/llms/bedrock_mantle/responses/transformation.py:33-58`
- Test: `tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py` (extend `TestBedrockMantleResponsesURL`)

- [ ] **Step 1: Write the failing tests for the standard `/v1/responses` path**

Add these methods inside the existing `class TestBedrockMantleResponsesURL:` in the test file. They construct the config with `use_openai_path=False` and assert the standard path, across three base inputs (default env region, user-supplied `.../v1`, user-supplied full endpoint URL). Also lock the default-construction behavior.

```python
    def test_standard_path_uses_region_from_env(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_REGION", "us-east-2")
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig(use_openai_path=False)
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://bedrock-mantle.us-east-2.api.aws/v1/responses"
        assert "/openai/v1/responses" not in url

    def test_standard_path_normalizes_v1_base(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig(use_openai_path=False)
        url = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-2.api.aws/v1",
            litellm_params={},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/v1/responses"
        assert url.count("/responses") == 1
        assert "/v1/v1/responses" not in url

    def test_standard_path_full_endpoint_base_not_doubled(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig(use_openai_path=False)
        url = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-2.api.aws/v1/responses",
            litellm_params={},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/v1/responses"
        assert url.count("/responses") == 1

    def test_default_construction_keeps_openai_path(self, monkeypatch):
        # Default (no arg) must remain the frontier /openai/v1/responses path so
        # gpt-5.x behavior is unchanged.
        monkeypatch.setenv("BEDROCK_MANTLE_REGION", "us-east-2")
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest "tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py::TestBedrockMantleResponsesURL" -v`
Expected: the three `test_standard_path_*` tests FAIL — `get_complete_url` currently always returns `/openai/v1/responses`, so they get the openai path instead of `/v1/responses`. `test_default_construction_keeps_openai_path` PASSES already (it documents current behavior). `BedrockMantleResponsesAPIConfig(use_openai_path=False)` also raises `TypeError` until the constructor exists — that counts as a fail.

- [ ] **Step 3: Implement the constructor and path-aware URL**

Edit `litellm/llms/bedrock_mantle/responses/transformation.py`. Add an `__init__` and replace the final return of `get_complete_url`. The full class head becomes:

```python
class BedrockMantleResponsesAPIConfig(OpenAIResponsesAPIConfig):
    def __init__(self, use_openai_path: bool = True):
        super().__init__()
        self.use_openai_path = use_openai_path

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.BEDROCK_MANTLE

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        region = (
            get_secret_str("BEDROCK_MANTLE_REGION")
            or get_secret_str("AWS_REGION")
            or BEDROCK_MANTLE_DEFAULT_REGION
        )
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
        path = "/openai/v1/responses" if self.use_openai_path else "/v1/responses"
        return f"{base}{path}"
```

Leave `validate_environment`, `supports_native_file_search`, `supports_native_websocket` unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest "tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py::TestBedrockMantleResponsesURL" -v`
Expected: all PASS (the 6 pre-existing openai-path URL tests plus the 4 new ones).

- [ ] **Step 5: Commit**

```bash
git add litellm/llms/bedrock_mantle/responses/transformation.py tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py
git commit -m "feat(bedrock_mantle): path-aware Responses URL via use_openai_path flag"
```

---

### Task 2: Shared behavior + outbound request body hold on the standard path

**Files:**
- Test: `tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py` (extend `TestBedrockMantleResponsesAuth`; add a new `TestBedrockMantleResponsesRequestBody` class)

This task adds no production code. It locks two things for the `use_openai_path=False` instance: (a) it shares the same Bearer auth and native-feature opt-outs as the default instance, so a future refactor cannot silently diverge the two paths; (b) the outbound request body carries the bare model id. (b) closes the gap between "the URL is `/v1/responses`" (Task 1) and "the request sent to that URL is correct" — the whole point of the feature is the right model reaching the right path.

- [ ] **Step 1: Write the shared-behavior tests**

Add inside the existing `class TestBedrockMantleResponsesAuth:`:

```python
    def test_standard_path_still_uses_bearer_auth(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "env-key")
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        cfg = BedrockMantleResponsesAPIConfig(use_openai_path=False)
        headers = cfg.validate_environment(
            headers={},
            model="openai.gpt-oss-120b",
            litellm_params=GenericLiteLLMParams(),
        )
        assert headers["Authorization"] == "Bearer env-key"

    def test_standard_path_opts_out_of_native_features(self):
        cfg = BedrockMantleResponsesAPIConfig(use_openai_path=False)
        assert cfg.supports_native_file_search() is False
        assert cfg.supports_native_websocket() is False
```

- [ ] **Step 2: Write the outbound-body test**

Add a new class to the test file (after `TestBedrockMantleResponsesAuth`). `GenericLiteLLMParams` is already imported at the top of the file (used by the auth tests):

```python
class TestBedrockMantleResponsesRequestBody:
    def test_standard_path_outbound_body_carries_bare_model(self):
        # The whole feature is "the right model reaches /v1/responses". The URL
        # tests prove the path; this proves the request body sent to it has the
        # bare model id and the input. transform is inherited and path-agnostic,
        # so this also guards against a future regression in transform.
        cfg = BedrockMantleResponsesAPIConfig(use_openai_path=False)
        body = cfg.transform_responses_api_request(
            model="openai.gpt-oss-120b",
            input="hello",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body["model"] == "openai.gpt-oss-120b"
        assert "input" in body
```

- [ ] **Step 3: Run the tests**

Run: `python3 -m pytest "tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py::TestBedrockMantleResponsesAuth" "tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py::TestBedrockMantleResponsesRequestBody" -v`
Expected: PASS (these behaviors are inherited and unchanged; the tests confirm the flag affects only the URL, not auth/features/body). Verified empirically that the default config already returns `model="openai.gpt-oss-120b"` with an `input` field, so this passes once `use_openai_path` exists from Task 1.

- [ ] **Step 4: Commit**

```bash
git add tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py
git commit -m "test(bedrock_mantle): lock shared auth/feature/body invariants on standard responses path"
```

---

### Task 3: Path-aware gate (the core routing change)

**Files:**
- Modify: `litellm/utils.py:8912-8922` (the `elif litellm.LlmProviders.BEDROCK_MANTLE == provider:` branch inside `_get_python_responses_api_config`)
- Test: `tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py` (extend `TestBedrockMantleResponsesRegistry`)

- [ ] **Step 1: Write the failing test for the new-feature core (declared-responses → standard path)**

Add a register_model isolation fixture and the core test to the test file. Put the fixture at module level near the existing `local_cost_map` fixture, and the test inside `class TestBedrockMantleResponsesRegistry:`.

Add `import copy` to the test file's imports if it is not already present (it is needed for the deepcopy snapshot below).

```python
@pytest.fixture
def restore_model_cost():
    """Snapshot litellm.model_cost so register_model edits don't leak across tests.

    register_model mutates the global litellm.model_cost, and get_model_info is
    lru_cached, so without restore + cache_clear a registered model would bleed
    into sibling tests in the same process.

    The snapshot MUST be a deepcopy, not a shallow dict() copy. register_model
    overwrites an existing key via
    `litellm.model_cost.setdefault(key, {}).update(...)`, mutating the nested
    dict in place. A shallow copy shares those nested dicts, so restoring the
    outer dict cannot undo an overwrite of an existing entry (e.g. gpt-oss-120b);
    teardown would leave mode=responses and poison TestBedrockMantleResponsesPricing.
    Verified empirically: shallow copy fails to restore, deepcopy restores to chat.
    """
    original_model_cost = copy.deepcopy(litellm.model_cost)
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()
```

```python
    def test_declared_responses_non_openai_routes_to_standard_path(
        self, restore_model_cost
    ):
        # New feature: a non-OpenAI model declared mode=responses (e.g. via a
        # user's proxy model_info block) must route to the STANDARD /v1/responses
        # path, not the frontier /openai/v1/responses path. Fails before the
        # path-aware gate exists (old gate returned None for non-gpt models).
        from litellm.utils import ProviderConfigManager, register_model

        register_model(
            {
                "bedrock_mantle/somelab.future-model": {
                    "litellm_provider": "bedrock_mantle",
                    "mode": "responses",
                }
            }
        )
        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="somelab.future-model",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        assert cfg.use_openai_path is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest "tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py::TestBedrockMantleResponsesRegistry::test_declared_responses_non_openai_routes_to_standard_path" -v`
Expected: FAIL — the current gate returns `None` for a non-gpt model, so `isinstance(cfg, BedrockMantleResponsesAPIConfig)` fails.

- [ ] **Step 3: Rewrite the gate branch**

In `litellm/utils.py`, replace the existing BEDROCK_MANTLE branch (currently lines 8912-8922):

```python
        elif litellm.LlmProviders.BEDROCK_MANTLE == provider:
            # Only OpenAI gpt frontier models (gpt-5.x, and future gpt-6 etc.) are
            # served on the /openai/v1/responses path. gpt-oss and every non-OpenAI
            # model on Mantle (nvidia, mistral, google, zai, ...) are chat-completions
            # only and 400 on that path, so they fall through to None to keep the
            # chat-completions emulation (see litellm/responses/main.py "config is None").
            model_lower = model.lower() if model else ""
            if "openai.gpt-" in model_lower and "gpt-oss" not in model_lower:
                return litellm.BedrockMantleResponsesAPIConfig()
            return None
        return None
```

with the path-aware version:

```python
        elif litellm.LlmProviders.BEDROCK_MANTLE == provider:
            # gpt frontier models (gpt-5.x, future gpt-6) live on the
            # /openai/v1/responses path; any other model declared mode=responses
            # (price-map entry or a user model_info block) is served on the
            # standard /v1/responses path. Everything else returns None and keeps
            # the chat-completions emulation (see responses/main.py "config is None").
            model_lower = model.lower() if model else ""
            if "openai.gpt-" in model_lower and "gpt-oss" not in model_lower:
                return litellm.BedrockMantleResponsesAPIConfig(use_openai_path=True)
            if model:
                try:
                    if (
                        get_model_info(model, "bedrock_mantle").get("mode")
                        == "responses"
                    ):
                        return litellm.BedrockMantleResponsesAPIConfig(
                            use_openai_path=False
                        )
                except Exception:
                    pass
            return None
        return None
```

Note: `get_model_info` is defined at module level in `litellm/utils.py`, so it is callable directly inside this static method without import.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest "tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py::TestBedrockMantleResponsesRegistry::test_declared_responses_non_openai_routes_to_standard_path" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add litellm/utils.py tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py
git commit -m "feat(bedrock_mantle): route declared mode=responses models to /v1/responses"
```

---

### Task 4: Lock the full routing matrix (regression + gpt-oss opt-in + flag assertions + graceful fallback)

**Files:**
- Test: `tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py` (extend `TestBedrockMantleResponsesRegistry`)

This task adds no production code. It strengthens the existing registry tests so the frontier branch asserts the flag (not just isinstance), and adds the gpt-oss opt-in, graceful-fallback, and a non-OpenAI-default-None case. These guard the routing matrix against mutation.

- [ ] **Step 1: Add flag assertions to the existing frontier tests**

In `class TestBedrockMantleResponsesRegistry`, the existing `test_registry_returns_config_for_gpt_5_5`, `test_registry_returns_config_for_gpt_5_4_enum`, and `test_registry_returns_config_for_future_frontier_model` currently only assert `isinstance(...)`. Add a flag assertion line to each so a path mutation is caught. For example, `test_registry_returns_config_for_gpt_5_5` becomes:

```python
    def test_registry_returns_config_for_gpt_5_5(self):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-5.5",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        assert cfg.use_openai_path is True
```

Apply the same `assert cfg.use_openai_path is True` addition to `test_registry_returns_config_for_gpt_5_4_enum` and `test_registry_returns_config_for_future_frontier_model`.

- [ ] **Step 2: Add the gpt-oss opt-in, graceful-fallback, and non-openai-default tests**

Add these methods to `class TestBedrockMantleResponsesRegistry`:

```python
    def test_gpt_oss_default_returns_none(self):
        # gpt-oss is mode=chat in the price map, so by default it keeps the
        # chat-completions emulation (returns None). Locks the opt-in semantics.
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-oss-120b",
        )
        assert cfg is None

    def test_gpt_oss_opt_in_routes_to_standard_path(self, restore_model_cost):
        # When a user opts gpt-oss into native Responses via model_info mode,
        # it must take the STANDARD /v1/responses path (gpt-oss Responses is on
        # /v1/responses, NOT the frontier /openai/v1/responses path).
        from litellm.utils import ProviderConfigManager, register_model

        register_model(
            {
                "bedrock_mantle/openai.gpt-oss-120b": {
                    "litellm_provider": "bedrock_mantle",
                    "mode": "responses",
                }
            }
        )
        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-oss-120b",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        assert cfg.use_openai_path is False

    def test_unmapped_model_degrades_to_none_without_crashing(
        self, restore_model_cost
    ):
        # A non-frontier model that is not in model_cost makes get_model_info
        # raise; the gate must swallow it and return None rather than crash.
        from litellm.utils import ProviderConfigManager

        litellm.model_cost.pop("bedrock_mantle/somelab.unmapped-model", None)
        litellm.get_model_info.cache_clear()
        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="somelab.unmapped-model",
        )
        assert cfg is None
```

- [ ] **Step 2b: Add the fixture-isolation guard test**

This test directly catches a shallow-copy regression in `restore_model_cost`: it opts gpt-oss in inside a nested fixture-style block, then asserts the gate returns None again once the snapshot is restored. With a shallow `dict()` snapshot this assertion fails (the in-place `setdefault().update()` leaks); with the deepcopy snapshot it passes. Add to `class TestBedrockMantleResponsesRegistry`:

```python
    def test_opt_in_does_not_leak_after_restore(self):
        # Guards the restore_model_cost fixture: a deepcopy snapshot must fully
        # undo a register_model overwrite of an existing key. With a shallow copy
        # this fails because register_model mutates the nested dict in place.
        import copy

        from litellm.utils import ProviderConfigManager, register_model

        snapshot = copy.deepcopy(litellm.model_cost)
        litellm.get_model_info.cache_clear()
        try:
            register_model(
                {
                    "bedrock_mantle/openai.gpt-oss-120b": {
                        "litellm_provider": "bedrock_mantle",
                        "mode": "responses",
                    }
                }
            )
            during = ProviderConfigManager.get_provider_responses_api_config(
                provider="bedrock_mantle", model="openai.gpt-oss-120b"
            )
            assert isinstance(during, BedrockMantleResponsesAPIConfig)
        finally:
            litellm.model_cost = snapshot
            litellm.get_model_info.cache_clear()
        after = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle", model="openai.gpt-oss-120b"
        )
        assert after is None
```

The pre-existing `test_registry_returns_none_for_non_openai_models` (parametrized over nvidia/mistral/google/zai) and `test_registry_returns_none_when_model_is_none` already cover the non-OpenAI-default-None and model=None cases, so no new test is needed for those.

- [ ] **Step 3: Run the full registry + URL + auth test classes**

Run: `python3 -m pytest "tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py" -v`
Expected: all PASS, including the strengthened frontier tests and the new opt-in/fallback tests.

- [ ] **Step 4: Run the broader bedrock_mantle + a price-map sanity check to confirm no global pollution**

Run: `python3 -m pytest tests/test_litellm/llms/bedrock_mantle/ -q`
Expected: all PASS. This confirms the `restore_model_cost` fixture teardown left `litellm.model_cost` clean for the sibling `TestBedrockMantleResponsesPricing` tests (which read gpt-5.5/5.4 entries from the local cost map).

- [ ] **Step 5: Commit**

```bash
git add tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py
git commit -m "test(bedrock_mantle): lock responses routing matrix (frontier/standard/opt-in/fallback)"
```

---

### Task 5: Format, lint, full-file test sweep

**Files:** none changed beyond formatting.

- [ ] **Step 1: Format the changed files**

Run:
```bash
python3 -m black litellm/utils.py litellm/llms/bedrock_mantle/responses/transformation.py tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py
```
Expected: files reformatted or already-formatted.

- [ ] **Step 2: Lint the changed files**

Run:
```bash
python3 -m ruff check litellm/llms/bedrock_mantle/responses/transformation.py tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py
```
Expected: no errors. (If `litellm/utils.py` reports pre-existing warnings unrelated to the small diff, leave them; do not expand scope.)

- [ ] **Step 3: Run the full responses + bedrock_mantle test sweep**

Run:
```bash
python3 -m pytest tests/test_litellm/llms/bedrock_mantle/ -q
```
Expected: all PASS.

- [ ] **Step 4: Commit any formatting changes**

```bash
git add -A
git commit -m "style(bedrock_mantle): format + lint responses path-aware changes" || echo "nothing to commit"
```

---

## Manual verification (Proof of Fix) — for the PR, run by the user on EC2

Not a plan step; this is the proof-of-fix the PR description will carry. Real proxy, real Bedrock Mantle, curl only (no python script, no mock).

1. In the proxy config, add `bedrock_mantle/openai.gpt-oss-120b` with `model_info: {mode: responses}`.
2. Start: `python litellm/proxy/proxy_cli.py --config <cfg> --detailed_debug --reload --use_v2_migration_resolver 2>&1 | tee litellm.log`.
3. `curl` the `/v1/responses` endpoint for this model, then a second call passing `previous_response_id` from the first; confirm the multi-turn state works (emulation cannot do this) and that `litellm.log` shows the outbound hitting `bedrock-mantle.<region>.api.aws/v1/responses`.
4. Control: remove `model_info` for the same model; confirm it falls back to emulation and the outbound hits `/v1/chat/completions`.
5. Frontier regression (if quota allows): `curl` `bedrock_mantle/openai.gpt-5.5`; confirm the outbound still hits `/openai/v1/responses`.

---

## Self-review notes

- **Spec coverage:** gate three-branch (Task 3) ✓; config flag + standard-path URL (Task 1) ✓; default unchanged / gpt-5.x regression (Task 1 default test + Task 4 flag assertions) ✓; non-OpenAI default None (pre-existing parametrized test, noted in Task 4) ✓; declared-responses core (Task 3) ✓; gpt-oss opt-in (Task 4) ✓; graceful fallback on unmapped (Task 4) ✓; model=None (pre-existing test, noted in Task 4) ✓; shared auth/feature opt-out on standard path (Task 2) ✓; outbound-body carries bare model on standard path / F5 (Task 2 `TestBedrockMantleResponsesRequestBody`) ✓; test isolation via deepcopy restore fixture + cache_clear / F1 (Task 3 fixture + Task 4 no-leak guard test) ✓; format/lint (Task 5) ✓; proof-of-fix runbook (manual section) ✓.
- **Adversarial-review fixes folded in:** F1 (fixture must deepcopy, not shallow dict — register_model mutates nested dicts in place; verified empirically) addressed in the Task 3 fixture and the Task 4 `test_opt_in_does_not_leak_after_restore` guard. F2 (mode vs supported_endpoints routing-signal tradeoff) documented in the spec. F5 (outbound-body test) added as Task 2 `TestBedrockMantleResponsesRequestBody`. F3/F4 (demand justification and the opt-in footgun) are PR-description decisions, not code changes.
- **Placeholder scan:** no TBD/TODO; every code step shows full code; every run step shows exact command + expected outcome.
- **Type/name consistency:** `use_openai_path` flag name, `BedrockMantleResponsesAPIConfig` class name, `restore_model_cost` fixture name, `TestBedrockMantleResponsesRequestBody` class name, and `get_model_info(model, "bedrock_mantle")` signature are used identically across Tasks 1-4. `import copy` is required by both the fixture and the Task 4 guard test.
