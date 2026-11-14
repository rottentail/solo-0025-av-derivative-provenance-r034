"""Content-addressed media files with path checks and atomic writes."""

from __future__ import annotations

import os
import tempfile
import threading
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

from ..errors import PersistenceFailure
from ..serialization import sha256_hex


class _KeyLock:
    """Reader/writer lock for one content key.

    Blobs are immutable once written, so any number of readers may proceed at
    the same time; only a write or removal excludes them. The lock is not
    acquired in any multi-key order, because callers never hold more than one
    key at a time, so requests that touch the same sources in opposite roles
    cannot deadlock. Writes are re-entrant for the owning thread so a caller can
    keep one key exclusive across "write blob, then commit metadata".
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._waiting_writers = 0
        self._writer: int | None = None
        self._writer_depth = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        owner = threading.get_ident()
        acquired = False
        with self._condition:
            if self._writer == owner:
                # A writer may also read the bytes it is publishing.
                pass
            else:
                while self._writer is not None or self._waiting_writers:
                    self._condition.wait()
                self._readers += 1
                acquired = True
        try:
            yield
        finally:
            if acquired:
                with self._condition:
                    self._readers -= 1
                    if self._readers == 0:
                        self._condition.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        owner = threading.get_ident()
        nested = False
        with self._condition:
            if self._writer == owner:
                self._writer_depth += 1
                nested = True
            else:
                self._waiting_writers += 1
                while self._writer is not None or self._readers:
                    self._condition.wait()
                self._waiting_writers -= 1
                self._writer = owner
                self._writer_depth = 1
        try:
            yield
        finally:
            with self._condition:
                if nested:
                    self._writer_depth -= 1
                else:
                    self._writer_depth = 0
                    self._writer = None
                    self._condition.notify_all()


class ContentFileRepository:
    """Immutable, content-addressed blob store.

    Readers of a key run concurrently; writers are serialized per key and
    excluded from readers. No operation acquires two key locks, so independent
    media work is never serialized or deadlocked against another request.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_catalog = threading.Lock()
        self._key_locks: dict[str, _KeyLock] = {}

    def _key_lock(self, key: str) -> _KeyLock:
        with self._lock_catalog:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = _KeyLock()
                self._key_locks[key] = lock
            return lock

    def read_lease(self, key: str) -> Iterator[None]:
        return self._key_lock(key).read()

    def write_lease(self, key: str) -> Iterator[None]:
        return self._key_lock(key).write()

    @contextmanager
    def source_lease(self, keys: list[str]) -> Iterator[None]:
        """Hold reader leases for several keys in a global order.

        Acquiring keys in sorted order makes overlapping multi-key leases
        deadlock-free. Callers that need only one key should prefer a plain
        read, which never waits on another key.
        """

        with ExitStack() as stack:
            for key in sorted(set(keys)):
                stack.enter_context(self.read_lease(key))
            yield

    @staticmethod
    def key_for_digest(digest: str) -> str:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise PersistenceFailure("A content digest has an invalid shape.")
        return f"{digest[:2]}/{digest[2:4]}/{digest}"

    def _path_for_key(self, key: str) -> Path:
        parts = Path(key).parts
        if len(parts) != 3:
            raise PersistenceFailure("A content key has an invalid shape.")
        root = self.root.resolve()
        candidate = (self.root / key).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PersistenceFailure("A content key escapes the data root.") from exc
        return candidate

    def write(self, data: bytes, digest: str) -> str:
        if sha256_hex(data) != digest:
            raise PersistenceFailure("Content digest does not match the supplied bytes.")
        key = self.key_for_digest(digest)
        target = self._path_for_key(key)
        with self.write_lease(key):
            if target.exists():
                return key
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".content-",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            )
            temporary_path = Path(temporary.name)
            try:
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary.close()
                os.replace(temporary_path, target)
            except OSError as exc:
                temporary.close()
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise PersistenceFailure("Media bytes could not be saved.") from exc
        return key

    def read(self, key: str) -> bytes:
        path = self._path_for_key(key)
        with self.read_lease(key):
            try:
                return path.read_bytes()
            except OSError as exc:
                raise PersistenceFailure("A referenced media file is missing.") from exc

    def remove(self, key: str) -> None:
        path = self._path_for_key(key)
        with self.write_lease(key):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                return

    def remove_unreferenced(self, key: str, is_referenced) -> None:
        """Remove a key only when the caller's truth test says nothing uses it.

        The write lease is held across the caller's reference test, so a
        concurrent rendition that publishes this exact key either finished
        committing metadata before the test or waits until after the removal and
        re-creates the blob. Delivered media is therefore never deleted.
        """

        with self.write_lease(key):
            if is_referenced():
                return
            try:
                self._path_for_key(key).unlink(missing_ok=True)
            except OSError:
                return

    def exists(self, key: str) -> bool:
        with self.read_lease(key):
            return self._path_for_key(key).is_file()
