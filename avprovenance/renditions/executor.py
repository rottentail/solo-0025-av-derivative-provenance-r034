"""Execute a normalized plan and convert media failures to domain errors."""

from __future__ import annotations

from typing import Any
from typing import Iterator
import threading
from contextlib import contextmanager

from ..errors import RenderFailed, UnsupportedMedia
from ..media import MediaOperationFailure, RenderOutcome, apply_media_operations


_execution_condition = threading.Condition()
_execution_active = False


@contextmanager
def execution_window() -> Iterator[None]:
    global _execution_active
    with _execution_condition:
        while _execution_active:
            _execution_condition.wait()
        _execution_active = True
    yield
    with _execution_condition:
        _execution_active = False
        _execution_condition.notify_all()


def execute_plan(
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
) -> RenderOutcome:
    with execution_window():
        return _execute_media(primary_bytes, operations, secondary_bytes)


def _execute_media(
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
) -> RenderOutcome:
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
