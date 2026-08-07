"""SQLite persistence for versioned Ramsey structural metric snapshots.

Stores :class:`ramsey.RMetric.RMetricSnapshot` values keyed by the
coloring they describe and the metric schema version they were computed
under, without altering the schema of any existing ``colorings`` table.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

from .RMetric import METRIC_VERSION, RMetricSnapshot


@dataclass(frozen=True, slots=True)
class RStoredMetric:
    """One metric snapshot restored from persistent storage.

    Attributes:
        coloring_id (int): Identifier of the coloring this snapshot
            describes, matching a row in the ``colorings`` table.
        metric_version (int): Metric schema version the snapshot was
            computed under.
        computed_at (str): Timestamp string (SQLite ``CURRENT_TIMESTAMP``
            format) of when the row was last inserted or updated.
        snapshot (RMetricSnapshot): The restored structural fingerprint.
    """

    coloring_id: int
    metric_version: int
    computed_at: str
    snapshot: RMetricSnapshot


class RMetricStore:
    """Store metric snapshots beside colorings without changing their table.

    Snapshots are stored in a dedicated ``coloring_metrics`` table keyed
    by ``(coloring_id, metric_version)``, with a foreign key back to
    ``colorings(coloring_id)`` (cascading on delete) so metrics never
    outlive the coloring they describe. Multiple metric versions can
    coexist for the same coloring, which lets callers keep old snapshots
    while migrating to a new :data:`ramsey.RMetric.METRIC_VERSION`.
    """

    def __init__(
        self,
        database_path: str | Path,
    ) -> None:
        """Open (creating if necessary) a metric store at a SQLite database path.

        Configures the connection with foreign keys enabled, WAL journal
        mode, and normal synchronous durability, then ensures the
        ``coloring_metrics`` table and its version index exist.

        Args:
            database_path (str | Path): Filesystem path to the SQLite
                database file.
        """
        self.database_path = Path(database_path)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self._closed = False
        self._create_schema()

    def _create_schema(self) -> None:
        """Create the ``coloring_metrics`` table and version index if absent."""
        with self.connection:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS coloring_metrics (
                    coloring_id INTEGER NOT NULL,
                    metric_version INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    computed_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    PRIMARY KEY (
                        coloring_id,
                        metric_version
                    ),

                    FOREIGN KEY (coloring_id)
                        REFERENCES colorings(coloring_id)
                        ON DELETE CASCADE
                )
            """)

            self.connection.execute("""
                CREATE INDEX IF NOT EXISTS
                    coloring_metrics_version_index
                ON coloring_metrics (
                    metric_version,
                    coloring_id
                )
            """)

    def save(
        self,
        coloring_id: int,
        snapshot: RMetricSnapshot,
    ) -> RStoredMetric:
        """Insert or replace one versioned metric snapshot.

        Upserts on the ``(coloring_id, metric_version)`` primary key: an
        existing row for the same coloring and version is overwritten and
        its ``computed_at`` timestamp refreshed.

        Args:
            coloring_id (int): Identifier of the coloring the snapshot
                describes.
            snapshot (RMetricSnapshot): Snapshot to persist. Its
                ``metric_version`` must match the software's current
                :data:`ramsey.RMetric.METRIC_VERSION`.

        Returns:
            RStoredMetric: The freshly saved row, reloaded from storage.

        Raises:
            ValueError: If ``snapshot.metric_version`` does not match
                :data:`ramsey.RMetric.METRIC_VERSION`.
            RuntimeError: If the store has been closed.
        """
        self._require_open()

        if snapshot.metric_version != METRIC_VERSION:
            raise ValueError(
                "Snapshot metric version does not match this software."
            )

        payload = json.dumps(
            snapshot.to_dict(),
            separators=(",", ":"),
            sort_keys=True,
        )

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO coloring_metrics (
                    coloring_id,
                    metric_version,
                    metrics_json
                )
                VALUES (?, ?, ?)
                ON CONFLICT(coloring_id, metric_version)
                DO UPDATE SET
                    metrics_json = excluded.metrics_json,
                    computed_at = CURRENT_TIMESTAMP
                """,
                (
                    int(coloring_id),
                    snapshot.metric_version,
                    payload,
                ),
            )

        return self.load(
            coloring_id,
            metric_version=snapshot.metric_version,
        )

    def load(
        self,
        coloring_id: int,
        *,
        metric_version: int = METRIC_VERSION,
    ) -> RStoredMetric:
        """Restore one stored snapshot.

        Args:
            coloring_id (int): Identifier of the coloring to look up.
            metric_version (int): Metric schema version to look up.
                Defaults to the current :data:`ramsey.RMetric.METRIC_VERSION`.

        Returns:
            RStoredMetric: The stored row for ``(coloring_id,
            metric_version)``.

        Raises:
            KeyError: If no snapshot is stored for that coloring and
                metric version.
            RuntimeError: If the store has been closed.
        """
        self._require_open()

        row = self.connection.execute(
            """
            SELECT
                coloring_id,
                metric_version,
                metrics_json,
                computed_at
            FROM coloring_metrics
            WHERE coloring_id = ?
              AND metric_version = ?
            """,
            (
                int(coloring_id),
                int(metric_version),
            ),
        ).fetchone()

        if row is None:
            raise KeyError(
                "No stored metrics for coloring "
                f"{coloring_id} at version {metric_version}."
            )

        return RStoredMetric(
            coloring_id=int(row["coloring_id"]),
            metric_version=int(row["metric_version"]),
            computed_at=str(row["computed_at"]),
            snapshot=RMetricSnapshot.from_dict(
                json.loads(row["metrics_json"])
            ),
        )

    def contains(
        self,
        coloring_id: int,
        *,
        metric_version: int = METRIC_VERSION,
    ) -> bool:
        """Return whether one coloring already has stored metrics.

        Args:
            coloring_id (int): Identifier of the coloring to check.
            metric_version (int): Metric schema version to check.
                Defaults to the current :data:`ramsey.RMetric.METRIC_VERSION`.

        Returns:
            bool: ``True`` if a snapshot is stored for that coloring and
            metric version.

        Raises:
            RuntimeError: If the store has been closed.
        """
        self._require_open()

        row = self.connection.execute(
            """
            SELECT 1
            FROM coloring_metrics
            WHERE coloring_id = ?
              AND metric_version = ?
            """,
            (
                int(coloring_id),
                int(metric_version),
            ),
        ).fetchone()

        return row is not None

    def count(
        self,
        *,
        metric_version: int = METRIC_VERSION,
    ) -> int:
        """Return the number of snapshots stored for one metric version.

        Args:
            metric_version (int): Metric schema version to count.
                Defaults to the current :data:`ramsey.RMetric.METRIC_VERSION`.

        Returns:
            int: Number of stored rows for that metric version.

        Raises:
            RuntimeError: If the store has been closed.
        """
        self._require_open()

        row = self.connection.execute(
            """
            SELECT COUNT(*) AS metric_count
            FROM coloring_metrics
            WHERE metric_version = ?
            """,
            (int(metric_version),),
        ).fetchone()

        return int(row["metric_count"])

    def close(self) -> None:
        """Commit any pending transaction and close the database connection.

        Idempotent: calling this more than once has no additional
        effect.
        """
        if not self._closed:
            self.connection.commit()
            self.connection.close()
            self._closed = True

    def __enter__(self) -> "RMetricStore":
        """Enter a context block with this already-open store.

        Returns:
            RMetricStore: This store instance.

        Raises:
            RuntimeError: If the store has already been closed.
        """
        self._require_open()
        return self

    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback,
    ) -> None:
        """Close the store on exit from a context block, regardless of exception."""
        self.close()

    def _require_open(self) -> None:
        """Guard a method against use after :meth:`close`.

        Raises:
            RuntimeError: If the store has already been closed.
        """
        if self._closed:
            raise RuntimeError("Metric store is closed.")