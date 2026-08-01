"""Unit tests for HMAC-bound identity on model-embedded file ids.

These ids are the DB-free alternative to unified managed-file ids. Because no
record exists to check ownership against, the caller's identity is signed into
the id itself; these tests pin the signing contract and every rejection path.
"""

import base64

import pytest

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.openai_files_endpoints.model_embedded_id_auth import (
    BadSignature,
    NotModelEmbedded,
    Unsigned,
    Verified,
    authorize_model_embedded_file_id,
    sign_model_embedded_payload,
    verify_model_embedded_file_id,
)

S3_URI = "s3://out-bucket/litellm-batch-outputs/litellm-batch-abc12345/input.jsonl.out"


def _encode(payload: str, prefix: str = "file-") -> str:
    return prefix + base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


@pytest.fixture(autouse=True)
def _salt(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-for-signing")


def test_signed_payload_appends_identity_and_signature():
    signed = sign_model_embedded_payload(f"litellm:{S3_URI};model,bedrock-claude", user_id="u1", team_id="t1")
    assert ";sub,u1;tid,t1;sig," in signed


def test_verify_returns_identity_for_signed_id():
    signed = sign_model_embedded_payload(f"litellm:{S3_URI};model,bedrock-claude", user_id="u1", team_id="t1")
    result = verify_model_embedded_file_id(_encode(signed))
    assert isinstance(result, Verified)
    assert result.inner_id == S3_URI
    assert result.model == "bedrock-claude"
    assert result.created_by == "u1"
    assert result.team_id == "t1"


def test_unsigned_model_embedded_id_is_rejected():
    """The reported attack: a hand-crafted id carrying no signature."""
    forged = _encode(f"litellm:{S3_URI};model,bedrock-claude")
    assert isinstance(verify_model_embedded_file_id(forged), Unsigned)


def test_tampered_signature_is_rejected():
    signed = sign_model_embedded_payload(f"litellm:{S3_URI};model,bedrock-claude", user_id="u1", team_id="t1")
    assert isinstance(verify_model_embedded_file_id(_encode(signed[:-4] + "0000")), BadSignature)


def test_swapping_the_model_invalidates_the_signature():
    """The signature covers `model`, so a caller cannot reuse their own id
    against a different deployment's credentials."""
    signed = sign_model_embedded_payload(f"litellm:{S3_URI};model,cheap-model", user_id="u1", team_id="t1")
    assert isinstance(
        verify_model_embedded_file_id(_encode(signed.replace("model,cheap-model", "model,expensive-model"))),
        BadSignature,
    )


def test_swapping_the_inner_id_invalidates_the_signature():
    signed = sign_model_embedded_payload(f"litellm:{S3_URI};model,bedrock-claude", user_id="u1", team_id="t1")
    other = "s3://out-bucket/litellm-batch-outputs/litellm-batch-victim01/input.jsonl.out"
    assert isinstance(verify_model_embedded_file_id(_encode(signed.replace(S3_URI, other))), BadSignature)


def test_escalating_identity_in_the_payload_is_rejected():
    signed = sign_model_embedded_payload(f"litellm:{S3_URI};model,bedrock-claude", user_id="u1", team_id="t1")
    assert isinstance(
        verify_model_embedded_file_id(_encode(signed.replace(";sub,u1;", ";sub,victim;"))),
        BadSignature,
    )


@pytest.mark.parametrize(
    "file_id",
    [
        "file-abc123",
        "batch_xyz",
        "s3://out-bucket/litellm-batch-outputs/j/input.jsonl.out",
        "",
    ],
)
def test_non_model_embedded_ids_pass_through(file_id):
    """Unified ids and plain provider ids must reach their own handling."""
    assert isinstance(verify_model_embedded_file_id(file_id), NotModelEmbedded)


def test_creator_is_authorized():
    signed = sign_model_embedded_payload(f"litellm:{S3_URI};model,bedrock-claude", user_id="u1", team_id="t1")
    authorize_model_embedded_file_id(_encode(signed), UserAPIKeyAuth(api_key="k", user_id="u1", team_id="t1"))


def test_same_team_different_user_is_authorized():
    """Must match the unified path's team-sharing behaviour."""
    signed = sign_model_embedded_payload(f"litellm:{S3_URI};model,bedrock-claude", user_id="u1", team_id="shared-team")
    authorize_model_embedded_file_id(
        _encode(signed),
        UserAPIKeyAuth(api_key="k", user_id="u2", team_id="shared-team"),
    )


def test_other_tenant_replaying_a_valid_id_is_denied():
    """The core IDOR: a genuinely signed id replayed by a different tenant."""
    from fastapi import HTTPException

    signed = sign_model_embedded_payload(
        f"litellm:{S3_URI};model,bedrock-claude", user_id="victim", team_id="victim-team"
    )
    with pytest.raises(HTTPException) as exc:
        authorize_model_embedded_file_id(
            _encode(signed),
            UserAPIKeyAuth(api_key="k", user_id="attacker", team_id="attacker-team"),
        )
    assert exc.value.status_code == 403


def test_unsigned_id_is_denied_with_403():
    from fastapi import HTTPException

    forged = _encode(f"litellm:{S3_URI};model,bedrock-claude")
    with pytest.raises(HTTPException) as exc:
        authorize_model_embedded_file_id(forged, UserAPIKeyAuth(api_key="k", user_id="attacker", team_id="t"))
    assert exc.value.status_code == 403


def test_proxy_admin_bypasses():
    signed = sign_model_embedded_payload(
        f"litellm:{S3_URI};model,bedrock-claude", user_id="someone", team_id="their-team"
    )
    authorize_model_embedded_file_id(
        _encode(signed),
        UserAPIKeyAuth(api_key="k", user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
    )


def test_a_sig_field_inside_the_inner_id_still_verifies():
    """`;sig,` is legal inside a caller-chosen inner id, so the real signature is
    the last one; splitting on the first would reject a legitimately minted id."""
    signed = sign_model_embedded_payload("litellm:s3://b/k;sig,deadbeef;model,m", user_id="u1", team_id="t1")
    result = verify_model_embedded_file_id(_encode(signed))
    assert isinstance(result, Verified)
    assert result.inner_id == "s3://b/k;sig,deadbeef"
    assert result.created_by == "u1"


def test_non_ascii_signature_is_rejected_not_crashed():
    """`hmac.compare_digest` raises TypeError on non-ASCII str, and the signature
    field is attacker-controlled, so comparing must happen over bytes."""
    forged = _encode(f"litellm:{S3_URI};model,m;sub,u1;tid,t1;sig,ábcdef")
    assert isinstance(verify_model_embedded_file_id(forged), BadSignature)


def test_identity_fields_injected_into_the_inner_id_cannot_spoof_the_signer():
    """A caller may choose the inner id, so `;sub,`/`;tid,` inside it must not be
    parsed as the signed identity; the trailing fields are the authoritative ones."""
    signed = sign_model_embedded_payload(
        "litellm:s3://b/k;sub,victim;tid,victim-team;model,m",
        user_id="attacker",
        team_id="attacker-team",
    )
    result = verify_model_embedded_file_id(_encode(signed))
    assert isinstance(result, Verified)
    assert result.created_by == "attacker"
    assert result.team_id == "attacker-team"


def test_identity_injected_in_the_inner_id_is_denied_to_the_victims_team():
    from fastapi import HTTPException

    signed = sign_model_embedded_payload(
        "litellm:s3://b/k;sub,victim;tid,victim-team;model,m",
        user_id="attacker",
        team_id="attacker-team",
    )
    with pytest.raises(HTTPException) as exc:
        authorize_model_embedded_file_id(
            _encode(signed),
            UserAPIKeyAuth(api_key="k", user_id="victim", team_id="victim-team"),
        )
    assert exc.value.status_code == 403


def test_signature_is_compared_in_constant_time():
    """Guards against reintroducing `==`, which leaks via timing."""
    import inspect

    from litellm.proxy.openai_files_endpoints import model_embedded_id_auth

    source = inspect.getsource(model_embedded_id_auth)
    assert "compare_digest" in source


def test_rotating_the_salt_invalidates_existing_signatures(monkeypatch):
    signed = sign_model_embedded_payload(f"litellm:{S3_URI};model,bedrock-claude", user_id="u1", team_id="t1")
    monkeypatch.setenv("LITELLM_SALT_KEY", "a-different-salt")
    assert isinstance(verify_model_embedded_file_id(_encode(signed)), BadSignature)
