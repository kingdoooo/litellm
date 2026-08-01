"""Caller-identity binding for model-embedded file/batch ids.

Model-embedded ids are the database-free alternative to unified managed-file
ids, so no stored record exists to check ownership against. The minting
caller's identity is instead signed into the id, and verified on the way back
in. Key material is derived through a fixed label so a signature minted here
cannot be replayed against another consumer of the same salt.
"""

import base64
import hashlib
import hmac
from binascii import Error as BinasciiError
from dataclasses import dataclass

from typing_extensions import assert_never

from litellm.llms.base_llm.managed_resources.isolation import can_access_resource
from litellm.proxy._types import UserAPIKeyAuth

_KEY_LABEL = b"litellm-file-id"
_PAYLOAD_PREFIX = "litellm:"
_MODEL_FIELD = ";model,"
_SUB_FIELD = ";sub,"
_TID_FIELD = ";tid,"
_SIG_FIELD = ";sig,"


@dataclass(frozen=True, slots=True)
class NotModelEmbedded:
    """Not a model-embedded id; the caller must fall through to other handling."""


@dataclass(frozen=True, slots=True)
class Unsigned:
    """Model-embedded but carries no signature: minted by an older build or forged."""


@dataclass(frozen=True, slots=True)
class BadSignature:
    """Signature present but does not cover the payload as presented."""


@dataclass(frozen=True, slots=True)
class Verified:
    inner_id: str
    model: str
    created_by: str | None
    team_id: str | None


IdentityVerification = NotModelEmbedded | Unsigned | BadSignature | Verified


def _signing_key() -> bytes:
    from litellm.proxy.common_utils.encrypt_decrypt_utils import _get_salt_key

    salt = _get_salt_key()
    if not salt:
        raise ValueError(
            "Signing model-embedded file ids requires key material: set LITELLM_SALT_KEY "
            "(or a master key). Without it the signature is forgeable."
        )
    return hmac.new(salt.encode(), _KEY_LABEL, hashlib.sha256).digest()


def _signature_for(signing_input: str) -> str:
    return hmac.new(_signing_key(), signing_input.encode(), hashlib.sha256).hexdigest()


def sign_model_embedded_payload(payload: str, user_id: str | None, team_id: str | None) -> str:
    signing_input = f"{payload}{_SUB_FIELD}{user_id or ''}{_TID_FIELD}{team_id or ''}"
    return f"{signing_input}{_SIG_FIELD}{_signature_for(signing_input)}"


def _decode_payload(file_id: str) -> str | None:
    if not isinstance(file_id, str) or not file_id:
        return None

    if file_id.startswith("file-"):
        b64_part = file_id[len("file-") :]
    elif file_id.startswith("batch_"):
        b64_part = file_id[len("batch_") :]
    else:
        b64_part = file_id

    try:
        decoded = base64.urlsafe_b64decode(b64_part + "=" * (-len(b64_part) % 4)).decode()
    except (BinasciiError, ValueError, UnicodeDecodeError):
        return None

    if not decoded.startswith(_PAYLOAD_PREFIX) or _MODEL_FIELD not in decoded:
        return None
    return decoded


def verify_model_embedded_file_id(file_id: str) -> IdentityVerification:
    payload = _decode_payload(file_id)
    if payload is None:
        return NotModelEmbedded()

    if _SIG_FIELD not in payload:
        return Unsigned()

    signing_input, _, signature = payload.rpartition(_SIG_FIELD)
    try:
        expected = _signature_for(signing_input)
    except ValueError:
        return BadSignature()
    if not hmac.compare_digest(signature.encode(), expected.encode()):
        return BadSignature()

    # The inner id is caller-chosen and may itself contain these delimiters, so the
    # identity fields are read right-to-left: only the trailing ones are signed as
    # the minter's identity.
    before_tid, tid_field, team_id = signing_input.rpartition(_TID_FIELD)
    before_sub, sub_field, user_id = before_tid.rpartition(_SUB_FIELD)
    inner_id, model_field, model = before_sub.rpartition(_MODEL_FIELD)
    if not (tid_field and sub_field and model_field):
        return BadSignature()

    return Verified(
        inner_id=inner_id.removeprefix(_PAYLOAD_PREFIX),
        model=model,
        created_by=user_id or None,
        team_id=team_id or None,
    )


def raise_if_unauthorized(result: IdentityVerification, user_api_key_dict: UserAPIKeyAuth) -> None:
    from fastapi import HTTPException

    denial = HTTPException(
        status_code=403,
        detail=f"User {user_api_key_dict.user_id} does not have access to this file id",
    )

    match result:
        case NotModelEmbedded():
            return
        case Unsigned() | BadSignature():
            raise denial
        case Verified(created_by=created_by, team_id=team_id):
            if can_access_resource(
                user_api_key_dict=user_api_key_dict,
                created_by=created_by,
                resource_team_id=team_id,
            ):
                return
            raise denial
        case _:
            assert_never(result)


def authorize_model_embedded_file_id(file_id: str, user_api_key_dict: UserAPIKeyAuth) -> None:
    raise_if_unauthorized(verify_model_embedded_file_id(file_id), user_api_key_dict)
