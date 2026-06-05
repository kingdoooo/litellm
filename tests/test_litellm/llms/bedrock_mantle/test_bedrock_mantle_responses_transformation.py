"""
Unit tests for Amazon Bedrock Mantle Responses API configuration.

Mantle's gpt-5.5 / gpt-5.4 are served ONLY on the non-standard
`/openai/v1/responses` path. These tests lock the URL construction and
Bearer auth that make that routing work.
"""

import copy
import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

import pytest

import litellm
from litellm.llms.bedrock_mantle.responses.transformation import (
    BedrockMantleResponsesAPIConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class TestBedrockMantleResponsesURL:
    def test_url_uses_region_from_env(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_REGION", "us-east-2")
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"

    def test_url_normalizes_v1_suffix(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-2.api.aws/v1",
            litellm_params={},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
        assert "/v1/openai/v1/responses" not in url
        url_trailing = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-2.api.aws/v1/",
            litellm_params={},
        )
        assert (
            url_trailing
            == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
        )

    def test_url_does_not_double_openai_v1(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
            litellm_params={},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"

    def test_url_full_endpoint_base_not_doubled(self, monkeypatch):
        # AWS model card tells users to set OPENAI_BASE_URL to the full endpoint.
        # If copied into api_base, it must not be doubled.
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            litellm_params={},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
        assert url.count("/responses") == 1

    def test_url_region_fallback_to_aws_region(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://bedrock-mantle.us-west-2.api.aws/openai/v1/responses"

    def test_url_region_default_us_east_1(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses"

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
        monkeypatch.setenv("BEDROCK_MANTLE_REGION", "us-east-2")
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"


class TestBedrockMantleResponsesAuth:
    def test_config_api_key_takes_priority(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "env-key")
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={},
            model="openai.gpt-5.5",
            litellm_params=GenericLiteLLMParams(api_key="config-key"),
        )
        assert headers["Authorization"] == "Bearer config-key"

    def test_env_key_fallback(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "env-key")
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={}, model="openai.gpt-5.5", litellm_params=GenericLiteLLMParams()
        )
        assert headers["Authorization"] == "Bearer env-key"

    def test_bedrock_bearer_token_fallback(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bearer-key")
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={}, model="openai.gpt-5.5", litellm_params=GenericLiteLLMParams()
        )
        assert headers["Authorization"] == "Bearer bearer-key"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        with pytest.raises(ValueError, match="Bedrock Mantle API key"):
            cfg.validate_environment(
                headers={},
                model="openai.gpt-5.5",
                litellm_params=GenericLiteLLMParams(),
            )

    def test_custom_llm_provider(self):
        cfg = BedrockMantleResponsesAPIConfig()
        assert cfg.custom_llm_provider == LlmProviders.BEDROCK_MANTLE

    def test_native_websocket_disabled(self):
        # Mantle Responses has no realtime/websocket transport, so the config
        # must opt out; otherwise realtime routing would try a socket Mantle
        # does not serve.
        cfg = BedrockMantleResponsesAPIConfig()
        assert cfg.supports_native_websocket() is False

    def test_file_search_routes_to_emulation(self):
        # Mantle cannot reach OpenAI's vector stores, so a native file_search
        # tool forwarded as-is gets a 400. The config must opt out of native
        # file_search so LiteLLM's emulation handles it instead of forwarding.
        from litellm.responses.file_search.emulated_handler import (
            should_use_emulated_file_search,
        )

        cfg = BedrockMantleResponsesAPIConfig()
        assert cfg.supports_native_file_search() is False
        assert (
            should_use_emulated_file_search(
                tools=[{"type": "file_search", "vector_store_ids": ["vs_1"]}],
                provider_config=cfg,
            )
            is True
        )

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


class TestBedrockMantleResponsesRequestBody:
    def test_standard_path_outbound_body_carries_bare_model(self):
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


class TestBedrockMantleResponsesRegistry:
    def test_registry_returns_config_for_gpt_5_5(self):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-5.5",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        assert cfg.use_openai_path is True

    def test_registry_returns_config_for_gpt_5_4_enum(self):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.BEDROCK_MANTLE,
            model="openai.gpt-5.4",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        assert cfg.use_openai_path is True

    def test_registry_returns_none_for_gpt_oss(self):
        # Regression guard: gpt-oss must NOT get the native Responses config; it
        # keeps the chat-completions emulation path (responses/main.py ~line 1109).
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-oss-120b",
        )
        assert cfg is None

    def test_registry_returns_none_for_gpt_oss_safeguard(self):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-oss-safeguard-20b",
        )
        assert cfg is None

    def test_registry_returns_config_for_future_frontier_model(self):
        # Forward-compatibility: an unseen OpenAI gpt frontier model (e.g. gpt-6) must
        # get the native Responses config without a code change. The gate allow-lists
        # the openai.gpt- family (minus gpt-oss), so gpt-6 matches automatically.
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-6",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        assert cfg.use_openai_path is True

    @pytest.mark.parametrize(
        "model",
        [
            "nvidia.nemotron-nano-9b-v2",
            "mistral.ministral-3-3b-instruct",
            "google.gemma-3-27b-it",
            "zai.glm-4.6",
        ],
    )
    def test_registry_returns_none_for_non_openai_models(self, model):
        # Regression for the chat-only families on Mantle. These models 400 on
        # /openai/v1/responses and are served on /v1/chat/completions, so the
        # registry must NOT hand them the Responses config; they fall through to
        # None and keep the chat-completions emulation.
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model=model,
        )
        assert cfg is None

    def test_registry_returns_none_when_model_is_none(self):
        # By-id operations (delete/get/cancel) call with model=None; keep returning
        # None so those paths are unchanged.
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model=None,
        )
        assert cfg is None

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

    def test_unmapped_model_degrades_to_none_without_crashing(self, restore_model_cost):
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

    def test_register_model_restore_undoes_existing_key_overwrite(self):
        # Self-contained guard for the deepcopy requirement of restore_model_cost.
        # register_model overwrites an existing key by mutating its nested dict in
        # place, so the snapshot must be a deepcopy: a shallow dict() copy would
        # share that nested dict and leave mode=responses after restore, making
        # the final assertion fail. The in-place clear+update mirrors the fixture.
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
            litellm.model_cost.clear()
            litellm.model_cost.update(snapshot)
            litellm.get_model_info.cache_clear()
        after = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle", model="openai.gpt-oss-120b"
        )
        assert after is None


@pytest.fixture
def restore_model_cost():
    """Snapshot litellm.model_cost so register_model edits don't leak across tests.

    register_model mutates the global litellm.model_cost, and get_model_info is
    lru_cached, so without restore + cache_clear a registered model would bleed
    into sibling tests in the same process.

    Two subtleties make this fixture non-obvious:

    1. The snapshot must be a deepcopy. register_model overwrites an existing key
       via `litellm.model_cost.setdefault(key, {}).update(...)`, mutating the
       nested dict in place; a shallow copy would share those nested dicts and
       could not capture the pre-mutation values of an existing entry.
    2. The restore must be in place (clear + update the SAME dict object), not a
       reassignment. The conftest autouse `isolate_litellm_state` fixture
       snapshots `litellm.model_cost` by reference and restores that reference on
       its teardown, which runs after this one. Reassigning `litellm.model_cost`
       to a fresh dict here is undone when conftest reinstalls its (in-place
       mutated) reference, so the registered mode would leak and poison
       TestBedrockMantleResponsesPricing. Mutating the original object in place
       restores the contents conftest's reference points at.
    """
    original_model_cost = copy.deepcopy(litellm.model_cost)
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost.clear()
        litellm.model_cost.update(original_model_cost)
        litellm.get_model_info.cache_clear()


@pytest.fixture
def local_cost_map(monkeypatch):
    """Force the bundled backup cost map and re-derive the provider model sets.

    ``litellm.model_cost`` is populated once at import time (here, from the
    network-fetched ``main`` copy, which lags this branch). ``add_known_models``
    only re-buckets whatever is already in ``model_cost``, so the cost map must
    first be reloaded from the local backup before the new keys appear.
    """
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    litellm.add_known_models()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


class TestBedrockMantleResponsesPricing:
    def test_gpt_5_5_pricing_and_mode(self, local_cost_map):
        info = litellm.get_model_info("bedrock_mantle/openai.gpt-5.5")
        assert info["mode"] == "responses"
        assert info["input_cost_per_token"] == pytest.approx(5.5e-06)
        assert info["output_cost_per_token"] == pytest.approx(3.3e-05)
        assert info["cache_read_input_token_cost"] == pytest.approx(5.5e-07)
        assert info["max_input_tokens"] == 272000

    def test_gpt_5_4_pricing_and_mode(self, local_cost_map):
        info = litellm.get_model_info("bedrock_mantle/openai.gpt-5.4")
        assert info["mode"] == "responses"
        assert info["input_cost_per_token"] == pytest.approx(2.75e-06)
        assert info["output_cost_per_token"] == pytest.approx(1.65e-05)
        assert info["cache_read_input_token_cost"] == pytest.approx(2.75e-07)
        assert info["max_input_tokens"] == 272000

    def test_models_registered(self, local_cost_map):
        assert "bedrock_mantle/openai.gpt-5.5" in litellm.bedrock_mantle_models
        assert "bedrock_mantle/openai.gpt-5.4" in litellm.bedrock_mantle_models
