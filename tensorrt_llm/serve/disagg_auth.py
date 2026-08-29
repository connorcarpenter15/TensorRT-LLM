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

import warnings
from typing import TYPE_CHECKING, Any, Mapping, Optional

from tensorrt_llm.disagg_auth import sign_disaggregated_payload, validate_disaggregated_payload

if TYPE_CHECKING:
    from tensorrt_llm.serve.openai_protocol import UCompletionRequest

INTERNAL_DISAGG_AUTH_HEADER = "x-trtllm-disagg-auth"
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


def _auth_payload(request: UCompletionRequest) -> dict[str, Any]:
    disaggregated_params = request.disaggregated_params
    return {
        "ctx_info_endpoint": _canonical_ctx_info_endpoint(disaggregated_params.ctx_info_endpoint),
        "encoded_opaque_state": disaggregated_params.encoded_opaque_state,
    }


def build_internal_disagg_auth_headers(
    internal_disagg_auth_key: Optional[str],
    request: UCompletionRequest,
) -> dict[str, str]:
    if not request_requires_internal_disagg_auth(request):
        return {}
    if not internal_disagg_auth_key:
        _warn_missing_auth_key()
        return {}
    signature = sign_disaggregated_payload(internal_disagg_auth_key, _auth_payload(request))
    return {INTERNAL_DISAGG_AUTH_HEADER: signature}


def validate_internal_disagg_request(
    internal_disagg_auth_key: Optional[str],
    request: UCompletionRequest,
    headers: Optional[Mapping[str, str]],
) -> None:
    if not request_requires_internal_disagg_auth(request):
        return
    if not internal_disagg_auth_key:
        _warn_missing_auth_key()
        return
    provided = None if headers is None else headers.get(INTERNAL_DISAGG_AUTH_HEADER)
    validate_disaggregated_payload(
        internal_disagg_auth_key,
        _auth_payload(request),
        provided,
    )
