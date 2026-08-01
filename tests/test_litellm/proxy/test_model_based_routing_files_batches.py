"""
Unit tests for encode_file_id_with_model, decode_model_from_file_id,
and get_original_file_id in common_utils.py.

Tests the model-based routing ID encoding/decoding used by the batch
and file proxy endpoints.
"""

from types import MappingProxyType, SimpleNamespace

from litellm.proxy.openai_files_endpoints.common_utils import (
    decode_model_from_file_id,
    encode_batch_response_ids,
    encode_file_id_with_model,
    get_original_file_id,
    prepare_data_with_credentials,
)


class TestEncodeFileIdWithModel:
    """Tests for encode_file_id_with_model."""

    def test_openai_file_id_gets_file_prefix(self):
        """OpenAI file IDs (file-xxx) should produce file- prefix."""
        result = encode_file_id_with_model("file-abc123", "gpt-4o")
        assert result.startswith("file-")

    def test_openai_batch_id_gets_batch_prefix(self):
        """OpenAI batch IDs (batch_xxx) should produce batch_ prefix."""
        result = encode_file_id_with_model("batch_abc123", "gpt-4o")
        assert result.startswith("batch_")

    def test_vertex_numeric_batch_id_gets_batch_prefix_with_id_type(self):
        """Vertex AI numeric batch IDs should produce batch_ prefix when id_type='batch'."""
        result = encode_file_id_with_model(
            "3814889423749775360", "gemini-2.5-pro", id_type="batch"
        )
        assert result.startswith(
            "batch_"
        ), f"Expected batch_ prefix for Vertex numeric batch ID, got: {result[:10]}"

    def test_vertex_numeric_id_defaults_to_file_prefix(self):
        """Vertex AI numeric IDs should default to file- prefix when id_type is not specified."""
        result = encode_file_id_with_model("3814889423749775360", "gemini-2.5-pro")
        assert result.startswith(
            "file-"
        ), "Default id_type should produce file- prefix for backward compatibility"

    def test_gcs_uri_gets_file_prefix(self):
        """GCS URIs (output_file_id) should produce file- prefix."""
        result = encode_file_id_with_model(
            "gs://bucket/path/to/file.jsonl", "gemini-2.5-pro"
        )
        assert result.startswith("file-")


class TestPrepareDataWithCredentials:
    def test_preserves_trusted_internal_credentials_snapshot(self):
        data = {"file_id": "file-abc"}
        credentials = {
            "custom_llm_provider": "bedrock",
            "s3_bucket_name": "safe-bucket",
        }

        prepare_data_with_credentials(
            data=data,
            credentials=credentials,
            include_internal_credentials=True,
        )

        assert data["s3_bucket_name"] == "safe-bucket"
        assert "custom_llm_provider" not in data
        assert isinstance(
            data["_litellm_internal_model_credentials"], type(MappingProxyType({}))
        )
        assert (
            data["_litellm_internal_model_credentials"]["s3_bucket_name"]
            == "safe-bucket"
        )

    def test_does_not_add_internal_credentials_by_default(self):
        data = {"file_id": "file-abc"}
        credentials = {
            "custom_llm_provider": "bedrock",
            "s3_bucket_name": "safe-bucket",
        }

        prepare_data_with_credentials(data=data, credentials=credentials)

        assert "_litellm_internal_model_credentials" not in data


class TestRoundTrip:
    """Tests for encode -> decode round-trip integrity."""

    def test_roundtrip_openai_file_id(self):
        """Encode then decode an OpenAI file ID — model and original ID should be recovered."""
        original = "file-abc123"
        model = "gpt-4o-litellm"
        encoded = encode_file_id_with_model(original, model)

        assert decode_model_from_file_id(encoded) == model
        assert get_original_file_id(encoded) == original

    def test_roundtrip_openai_batch_id(self):
        """Encode then decode an OpenAI batch ID — model and original ID should be recovered."""
        original = "batch_abc123"
        model = "gpt-4o-test"
        encoded = encode_file_id_with_model(original, model)

        assert decode_model_from_file_id(encoded) == model
        assert get_original_file_id(encoded) == original

    def test_roundtrip_vertex_numeric_batch_id(self):
        """Encode then decode a Vertex AI numeric batch ID with id_type='batch'."""
        original = "3814889423749775360"
        model = "gemini-2.5-pro"
        encoded = encode_file_id_with_model(original, model, id_type="batch")

        assert encoded.startswith("batch_")
        assert decode_model_from_file_id(encoded) == model
        assert get_original_file_id(encoded) == original

    def test_roundtrip_vertex_gcs_uri_file_id(self):
        """Encode then decode a Vertex AI GCS URI (output file)."""
        original = "gs://vertex-bucket/litellm-files/output.jsonl"
        model = "gemini-2.5-pro"
        encoded = encode_file_id_with_model(original, model)

        assert encoded.startswith("file-")
        assert decode_model_from_file_id(encoded) == model
        assert get_original_file_id(encoded) == original


class TestMintingBindsIdentity:
    """Minting must bind the caller so verification can authorize later."""

    def test_encoded_id_carries_signed_identity(self, monkeypatch):
        monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-for-signing")
        from litellm.proxy.openai_files_endpoints.model_embedded_id_auth import (
            Verified,
            verify_model_embedded_file_id,
        )

        encoded = encode_file_id_with_model(
            "file-abc123", "gpt-4o", user_id="u1", team_id="t1"
        )
        result = verify_model_embedded_file_id(encoded)
        assert isinstance(result, Verified)
        assert (result.created_by, result.team_id) == ("u1", "t1")
        assert result.inner_id == "file-abc123"

    def test_prefix_is_still_derived_from_the_inner_id(self, monkeypatch):
        monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-for-signing")
        assert encode_file_id_with_model(
            "batch_abc", "gpt-4o", user_id="u1", team_id="t1"
        ).startswith("batch_")
        assert encode_file_id_with_model(
            "file-abc", "gpt-4o", user_id="u1", team_id="t1"
        ).startswith("file-")

    def test_sign_false_mints_an_unsigned_id(self, monkeypatch):
        """list_batches cannot prove ownership, so it must mint unsigned."""
        monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-for-signing")
        from litellm.proxy.openai_files_endpoints.model_embedded_id_auth import (
            Unsigned,
            verify_model_embedded_file_id,
        )

        encoded = encode_file_id_with_model("batch_abc", "gpt-4o", sign=False)
        assert isinstance(verify_model_embedded_file_id(encoded), Unsigned)

    def test_decoders_still_work_on_signed_ids(self, monkeypatch):
        """auth_utils and batch_rate_limiter decode without identity in scope."""
        monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-for-signing")
        encoded = encode_file_id_with_model(
            "s3://bucket/litellm-batch-outputs/j/in.jsonl.out",
            "bedrock-claude",
            user_id="u1",
            team_id="t1",
        )
        assert decode_model_from_file_id(encoded) == "bedrock-claude"
        assert (
            get_original_file_id(encoded)
            == "s3://bucket/litellm-batch-outputs/j/in.jsonl.out"
        )


class TestBatchResponseIdsBindIdentity:
    """encode_batch_response_ids must thread identity to every id it rewrites."""

    def test_every_rewritten_id_carries_the_caller(self, monkeypatch):
        monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-for-signing")
        from litellm.proxy.openai_files_endpoints.model_embedded_id_auth import (
            Verified,
            verify_model_embedded_file_id,
        )

        response = SimpleNamespace(
            id="batch_abc",
            output_file_id="file-out",
            error_file_id="file-err",
            input_file_id="file-in",
        )

        encode_batch_response_ids(response, model="gpt-4o", user_id="u1", team_id="t1")

        for attr in ("id", "output_file_id", "error_file_id", "input_file_id"):
            result = verify_model_embedded_file_id(getattr(response, attr))
            assert isinstance(result, Verified), attr
            assert (result.created_by, result.team_id) == ("u1", "t1"), attr

    def test_sign_false_leaves_every_id_unsigned(self, monkeypatch):
        monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-for-signing")
        from litellm.proxy.openai_files_endpoints.model_embedded_id_auth import (
            Unsigned,
            verify_model_embedded_file_id,
        )

        response = SimpleNamespace(
            id="batch_abc",
            output_file_id="file-out",
            error_file_id="file-err",
            input_file_id="file-in",
        )

        encode_batch_response_ids(response, model="gpt-4o", sign=False)

        for attr in ("id", "output_file_id", "error_file_id", "input_file_id"):
            assert isinstance(
                verify_model_embedded_file_id(getattr(response, attr)), Unsigned
            ), attr


class TestDecodeEdgeCases:
    """Tests for decode functions with non-encoded inputs."""

    def test_decode_model_returns_none_for_plain_id(self):
        """Plain (non-encoded) IDs should return None from decode_model_from_file_id."""
        assert decode_model_from_file_id("batch_abc123") is None
        assert decode_model_from_file_id("file-abc123") is None
        assert decode_model_from_file_id("3814889423749775360") is None

    def test_get_original_file_id_returns_input_for_plain_id(self):
        """Plain (non-encoded) IDs should be returned as-is from get_original_file_id."""
        assert get_original_file_id("batch_abc123") == "batch_abc123"
        assert get_original_file_id("file-abc123") == "file-abc123"

    def test_decode_model_handles_non_string(self):
        """Non-string inputs should return None without raising."""
        assert decode_model_from_file_id(None) is None  # type: ignore[arg-type]
        assert decode_model_from_file_id(12345) is None  # type: ignore[arg-type]
