# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import hmac
import json
import warnings
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from tensorrt_llm.serve.openai_protocol import UCompletionRequest

INTERNAL_DISAGG_AUTH_HEADER = "x-trtllm-disagg-auth"
_SIGNATURE_PREFIX = "sha256="
_INTERNAL_DISAGG_AUTH_FIELDS = ("encoded_opaque_state", "ctx_info_endpoint")
_MISSING_AUTH_KEY_WARNING = (
    "Internal disaggregated authentication key is required for protected "
    "disaggregated request fields. In a future release the requirement to "
    "use internal_request_auth_key will be enforced. Please update workflow "
    "accordingly."
)


def get_internal_disagg_auth_fields() -> tuple[str, ...]:
    return _INTERNAL_DISAGG_AUTH_FIELDS


def _warn_missing_auth_key() -> None:
    warnings.warn(_MISSING_AUTH_KEY_WARNING, FutureWarning, stacklevel=2)


def request_requires_internal_disagg_auth(request: UCompletionRequest) -> bool:
    disaggregated_params = getattr(request, "disaggregated_params", None)
    return disaggregated_params is not None and any(
        getattr(disaggregated_params, field_name) is not None
        for field_name in get_internal_disagg_auth_fields()
    )


def _canonical_ctx_info_endpoint(endpoint: Any) -> Any:
    if isinstance(endpoint, list):
        return endpoint[0] if endpoint else None
    return endpoint


def _auth_payload(encoded_opaque_state: Optional[str], ctx_info_endpoint: Any) -> bytes:
    payload = {
        "ctx_info_endpoint": _canonical_ctx_info_endpoint(ctx_info_endpoint),
        "encoded_opaque_state": encoded_opaque_state,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sign_fields(
    internal_disagg_auth_key: str,
    encoded_opaque_state: Optional[str],
    ctx_info_endpoint: Any,
) -> str:
    signature = hmac.new(
        internal_disagg_auth_key.encode("utf-8"),
        _auth_payload(encoded_opaque_state, ctx_info_endpoint),
        hashlib.sha256,
    ).hexdigest()
    return f"{_SIGNATURE_PREFIX}{signature}"


def build_internal_disagg_auth_signature(
    internal_disagg_auth_key: Optional[str],
    encoded_opaque_state: Optional[str],
    ctx_info_endpoint: Any,
) -> Optional[str]:
    """Sign protected disaggregated fields for any transport."""
    if encoded_opaque_state is None and ctx_info_endpoint is None:
        return None
    if not internal_disagg_auth_key:
        _warn_missing_auth_key()
        return None
    return _sign_fields(internal_disagg_auth_key, encoded_opaque_state, ctx_info_endpoint)


def validate_internal_disagg_auth_signature(
    internal_disagg_auth_key: Optional[str],
    encoded_opaque_state: Optional[str],
    ctx_info_endpoint: Any,
    provided: Optional[str],
) -> None:
    """Validate protected disaggregated fields for any transport."""
    if encoded_opaque_state is None and ctx_info_endpoint is None:
        return
    if not internal_disagg_auth_key:
        _warn_missing_auth_key()
        return

    expected = _sign_fields(internal_disagg_auth_key, encoded_opaque_state, ctx_info_endpoint)
    try:
        valid = provided is not None and hmac.compare_digest(
            provided.encode("utf-8"), expected.encode("utf-8")
        )
    except (AttributeError, UnicodeEncodeError):
        valid = False
    if not valid:
        raise ValueError("Invalid internal disaggregated request authentication")


def build_internal_disagg_auth_headers(
    internal_disagg_auth_key: Optional[str],
    request: UCompletionRequest,
) -> dict[str, str]:
    disaggregated_params = request.disaggregated_params
    signature = build_internal_disagg_auth_signature(
        internal_disagg_auth_key,
        None if disaggregated_params is None else disaggregated_params.encoded_opaque_state,
        None if disaggregated_params is None else disaggregated_params.ctx_info_endpoint,
    )
    if signature is None:
        return {}
    return {INTERNAL_DISAGG_AUTH_HEADER: signature}


def validate_internal_disagg_request(
    internal_disagg_auth_key: Optional[str],
    request: UCompletionRequest,
    headers: Optional[Mapping[str, str]],
) -> None:
    disaggregated_params = request.disaggregated_params
    provided = None if headers is None else headers.get(INTERNAL_DISAGG_AUTH_HEADER)
    validate_internal_disagg_auth_signature(
        internal_disagg_auth_key,
        None if disaggregated_params is None else disaggregated_params.encoded_opaque_state,
        None if disaggregated_params is None else disaggregated_params.ctx_info_endpoint,
        provided,
    )
