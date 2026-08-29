# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-neutral authentication for disaggregated-serving payloads."""

import hashlib
import hmac
import json
from collections.abc import Mapping
from typing import Any

_SIGNATURE_PREFIX = "sha256="


def sign_disaggregated_payload(
    internal_disagg_auth_key: str | None,
    payload: Mapping[str, Any],
) -> str:
    """Return an HMAC signature over a canonical JSON payload."""
    if not internal_disagg_auth_key:
        raise ValueError("internal_request_auth_key must be a non-empty string")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(
        internal_disagg_auth_key.encode("utf-8"),
        encoded,
        hashlib.sha256,
    ).hexdigest()
    return f"{_SIGNATURE_PREFIX}{signature}"


def validate_disaggregated_payload(
    internal_disagg_auth_key: str | None,
    payload: Mapping[str, Any],
    provided_signature: str | None,
) -> None:
    """Reject a missing or invalid HMAC signature for a payload."""
    expected = sign_disaggregated_payload(internal_disagg_auth_key, payload)
    try:
        valid = provided_signature is not None and hmac.compare_digest(
            provided_signature.encode("utf-8"),
            expected.encode("utf-8"),
        )
    except (AttributeError, UnicodeEncodeError):
        valid = False
    if not valid:
        raise ValueError("Invalid internal disaggregated request authentication")


__all__ = ["sign_disaggregated_payload", "validate_disaggregated_payload"]
