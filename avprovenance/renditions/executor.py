"""Execute a normalized plan and convert media failures to domain errors."""

from __future__ import annotations

from typing import Any

from ..errors import RenderFailed, UnsupportedMedia
from ..media import MediaOperationFailure, RenderOutcome, apply_media_operations


def execute_plan(
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
) -> RenderOutcome:
    """Apply one plan to immutable source bytes.

    Rendering is a pure function of the supplied bytes and the plan. It owns no
    shared mutable state, so independent renditions and integrity checks are
    free to compute concurrently. Serializing this work would let one long
    derivative starve every other request, and a single failure must never
    close the gate for later work.
    """

    try:
        return apply_media_operations(primary_bytes, operations, secondary_bytes)
    except MediaOperationFailure as exc:
        raise RenderFailed(
            "The rendition could not be produced.",
            details={"reason": exc.code, "detail": exc.message},
        ) from exc
    except UnsupportedMedia as exc:
        raise RenderFailed(
            "The rendition produced unsupported media bytes.",
            details={"reason": exc.code},
        ) from exc
