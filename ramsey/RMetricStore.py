"""SQLite persistence for versioned Ramsey structural metric snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

from .RMetric import METRIC_VERSION, RMetricSnapshot


@dataclass(frozen=True, slots=True)
class RStoredMetric:
    """One metric snapshot restored from persistent storage."""

    coloring_id: int
    metric_version: int
    computed_at: str
    snapshot: RMetricSnapshot


class RMetricStore:
    """Store metric snapshots beside colorings without changing their table."""

    def __init__(
        self,
        database_path: str | Path,
    ) -> None:
        self.database_path = Path(database_path)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self._closed = False
        self._create_schema()

    def _create_schema(self) -> None:
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
        """Insert or replace one versioned metric snapshot."""
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
        """Restore one stored snapshot."""
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
        """Return whether one coloring already has stored metrics."""
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
        """Return the number of snapshots stored for one metric version."""
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
        if not self._closed:
            self.connection.commit()
            self.connection.close()
            self._closed = True

    def __enter__(self) -> "RMetricStore":
        self._require_open()
        return self

    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback,
    ) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Metric store is closed.")