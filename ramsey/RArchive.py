"""Persistent coloring, provenance, and leaderboard storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import sqlite3

import numpy as np
from numpy.typing import NDArray

from .RColoring import RColoring
from .RGraph import RGraph
from .RScoring import binary_histogram


@dataclass(frozen=True, slots=True)
class RArchiveRecord:
    """
    Metadata for one unique archived coloring.
    """

    coloring_id: int
    state_hash: str

    score: int
    color_zero_count: int
    color_one_count: int

    run_name: str
    iteration: int
    times_seen: int


@dataclass(
    frozen=True,
    slots=True,
    eq=False,
)
class RArchivedColoring:
    """
    One restored coloring with its stored metadata and histogram.
    """

    record: RArchiveRecord
    coloring: RColoring
    histogram: NDArray[np.int64]

    def __post_init__(self) -> None:
        histogram = np.asarray(
            self.histogram,
            dtype=np.int64,
        ).copy()

        histogram.flags.writeable = False

        object.__setattr__(
            self,
            "histogram",
            histogram,
        )


class RArchive(ABC):
    """
    Persistent storage interface for exact Ramsey colorings.
    """

    @abstractmethod
    def save_coloring(
        self,
        coloring: RColoring,
        *,
        run_name: str,
        iteration: int,
    ) -> RArchiveRecord:
        """
        Save or observe one exact coloring.
        """
        ...

    @abstractmethod
    def load_coloring(
        self,
        coloring_id: int,
        graph: RGraph,
    ) -> RArchivedColoring:
        """
        Restore one coloring using a compatible graph.
        """
        ...

    @abstractmethod
    def best_score(
        self,
        graph: RGraph | None = None,
    ) -> int | None:
        """
        Return the lowest stored score.
        """
        ...

    @abstractmethod
    def best_colorings(
        self,
        limit: int = 10,
        graph: RGraph | None = None,
    ) -> list[RArchiveRecord]:
        """
        Return metadata for the best stored colorings.
        """
        ...

    @abstractmethod
    def colorings_in_score_range(
        self,
        *,
        minimum_score: int | None = None,
        maximum_score: int | None = None,
        limit: int | None = None,
        graph: RGraph | None = None,
    ) -> list[RArchiveRecord]:
        """
        Return records in an inclusive score range.
        """
        ...

    @abstractmethod
    def coloring_count_in_score_range(
        self,
        *,
        minimum_score: int | None = None,
        maximum_score: int | None = None,
        graph: RGraph | None = None,
    ) -> int:
        """
        Count unique colorings in an inclusive score range.
        """
        ...

    @abstractmethod
    def coloring_count(
        self,
        graph: RGraph | None = None,
    ) -> int:
        """
        Return the number of unique stored colorings.
        """
        ...


class RSQLiteArchive(RArchive):
    """
    SQLite archive compatible with the original coloring database.

    Binary colorings are packed into bits. A K43 coloring therefore
    occupies 113 bytes rather than 903 bytes.
    """

    def __init__(
        self,
        database_path: str | Path,
    ) -> None:
        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(self.database_path)

        self.connection.row_factory = sqlite3.Row

        self.connection.execute("PRAGMA foreign_keys = ON")

        self.connection.execute("PRAGMA journal_mode = WAL")

        self.connection.execute("PRAGMA synchronous = NORMAL")

        self._closed = False

        self._create_schema()

    def _create_schema(self) -> None:
        """
        Create the compatible coloring table and indexes.
        """
        with self.connection:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS colorings (
                    coloring_id INTEGER PRIMARY KEY,
                    state_hash TEXT NOT NULL UNIQUE,

                    n_vertices INTEGER NOT NULL,
                    k_size INTEGER NOT NULL,
                    edge_count INTEGER NOT NULL,

                    packed_coloring BLOB NOT NULL,
                    histogram BLOB NOT NULL,

                    score INTEGER NOT NULL,
                    red_kn_count INTEGER NOT NULL,
                    blue_kn_count INTEGER NOT NULL,

                    first_run TEXT NOT NULL,
                    first_iteration INTEGER NOT NULL,

                    last_run TEXT NOT NULL,
                    last_iteration INTEGER NOT NULL,

                    times_seen INTEGER NOT NULL DEFAULT 1,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    updated_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """)

            self.connection.execute("""
                CREATE INDEX IF NOT EXISTS
                    colorings_score_index
                ON colorings (
                    score,
                    coloring_id
                )
                """)

            self.connection.execute("""
                CREATE INDEX IF NOT EXISTS
                    colorings_problem_score_index
                ON colorings (
                    n_vertices,
                    k_size,
                    score,
                    coloring_id
                )
                """)

    def save_coloring(
        self,
        coloring: RColoring,
        *,
        run_name: str,
        iteration: int,
    ) -> RArchiveRecord:
        """
        Save or observe one exact coloring.

        The histogram and score are calculated from the coloring
        inside the archive boundary. Callers cannot accidentally
        supply metadata belonging to a different coloring.

        If the coloring already exists, its observation count and
        most recent provenance are updated.
        """
        self._require_open()

        (
            n_vertices,
            k_size,
        ) = self._problem_key(coloring.graph)

        self._validate_provenance(
            run_name,
            iteration,
        )

        histogram = binary_histogram(coloring).astype(
            "<i8",
            copy=False,
        )

        color_zero_count = int(histogram[0])

        color_one_count = int(histogram[-1])

        score = color_zero_count + color_one_count

        packed_coloring = coloring.packed()

        state_hash = self._compatible_state_hash(
            packed_coloring,
            n_vertices,
            k_size,
            coloring.graph.number_of_edges,
        )

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO colorings (
                    state_hash,
                    n_vertices,
                    k_size,
                    edge_count,

                    packed_coloring,
                    histogram,

                    score,
                    red_kn_count,
                    blue_kn_count,

                    first_run,
                    first_iteration,
                    last_run,
                    last_iteration
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(state_hash)
                DO UPDATE SET
                    last_run = excluded.last_run,
                    last_iteration = excluded.last_iteration,
                    times_seen =
                        colorings.times_seen + 1,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    state_hash,
                    n_vertices,
                    k_size,
                    coloring.graph.number_of_edges,
                    packed_coloring,
                    histogram.tobytes(),
                    score,
                    color_zero_count,
                    color_one_count,
                    run_name,
                    iteration,
                    run_name,
                    iteration,
                ),
            )

        row = self.connection.execute(
            """
            SELECT *
            FROM colorings
            WHERE state_hash = ?
            """,
            (state_hash,),
        ).fetchone()

        if row is None:
            raise RuntimeError("Coloring was not found after insertion.")

        return self._row_to_record(row)

    def load_coloring(
        self,
        coloring_id: int,
        graph: RGraph,
    ) -> RArchivedColoring:
        """
        Restore one coloring using a compatible graph.
        """
        self._require_open()

        (
            n_vertices,
            k_size,
        ) = self._problem_key(graph)

        row = self.connection.execute(
            """
            SELECT *
            FROM colorings
            WHERE coloring_id = ?
            """,
            (coloring_id,),
        ).fetchone()

        if row is None:
            raise KeyError(f"Unknown coloring ID: " f"{coloring_id}")

        if int(row["n_vertices"]) != n_vertices or int(row["k_size"]) != k_size:
            raise ValueError("Archived coloring problem does " "not match graph.")

        if int(row["edge_count"]) != graph.number_of_edges:
            raise ValueError("Archived coloring edge count does " "not match graph.")

        packed = np.frombuffer(
            row["packed_coloring"],
            dtype=np.uint8,
        )

        colors = np.unpackbits(
            packed,
            bitorder="little",
        )[
            : graph.number_of_edges
        ].astype(np.uint8)

        expected_bins = graph.problem.edges_per_clique(k_size) + 1

        histogram = np.frombuffer(
            row["histogram"],
            dtype="<i8",
            count=expected_bins,
        ).copy()

        return RArchivedColoring(
            record=self._row_to_record(row),
            coloring=RColoring(
                graph,
                colors,
            ),
            histogram=histogram,
        )

    def best_score(
        self,
        graph: RGraph | None = None,
    ) -> int | None:
        """
        Return the lowest stored score.

        If graph is supplied, only colorings belonging to that
        problem are considered.
        """
        self._require_open()

        (
            where_clause,
            parameters,
        ) = self._problem_filter(graph)

        row = self.connection.execute(
            f"""
            SELECT MIN(score) AS best_score
            FROM colorings
            {where_clause}
            """,
            parameters,
        ).fetchone()

        if row is None or row["best_score"] is None:
            return None

        return int(row["best_score"])

    def best_colorings(
        self,
        limit: int = 10,
        graph: RGraph | None = None,
    ) -> list[RArchiveRecord]:
        """
        Return metadata for the best stored colorings.

        Results are ordered by score and then coloring ID.
        """
        self._require_open()

        if isinstance(limit, bool) or not isinstance(
            limit,
            (int, np.integer),
        ):
            raise TypeError("limit must be an integer.")

        if limit <= 0:
            raise ValueError("limit must be positive.")

        (
            where_clause,
            parameters,
        ) = self._problem_filter(graph)

        rows = self.connection.execute(
            f"""
            SELECT *
            FROM colorings
            {where_clause}
            ORDER BY
                score ASC,
                coloring_id ASC
            LIMIT ?
            """,
            (
                *parameters,
                int(limit),
            ),
        ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def coloring_count(
        self,
        graph: RGraph | None = None,
    ) -> int:
        """
        Return the number of unique stored colorings.

        If graph is supplied, only colorings belonging to that
        problem are counted.
        """
        self._require_open()

        (
            where_clause,
            parameters,
        ) = self._problem_filter(graph)

        row = self.connection.execute(
            f"""
            SELECT COUNT(*) AS coloring_count
            FROM colorings
            {where_clause}
            """,
            parameters,
        ).fetchone()

        return int(row["coloring_count"])

    def colorings_in_score_range(
        self,
        *,
        minimum_score: int | None = None,
        maximum_score: int | None = None,
        limit: int | None = None,
        graph: RGraph | None = None,
    ) -> list[RArchiveRecord]:
        """
        Return records ordered by score within inclusive bounds.
        """
        self._require_open()

        (
            where_clause,
            parameters,
        ) = self._score_range_filter(
            graph=graph,
            minimum_score=minimum_score,
            maximum_score=maximum_score,
        )

        limit = self._validate_optional_positive_integer(
            "limit",
            limit,
        )

        limit_clause = ""

        if limit is not None:
            limit_clause = "LIMIT ?"
            parameters = (
                *parameters,
                limit,
            )

        rows = self.connection.execute(
            f"""
            SELECT *
            FROM colorings
            {where_clause}
            ORDER BY
                score ASC,
                coloring_id ASC
            {limit_clause}
            """,
            parameters,
        ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def coloring_count_in_score_range(
        self,
        *,
        minimum_score: int | None = None,
        maximum_score: int | None = None,
        graph: RGraph | None = None,
    ) -> int:
        """
        Count bounded records without loading their metadata.
        """
        self._require_open()

        (
            where_clause,
            parameters,
        ) = self._score_range_filter(
            graph=graph,
            minimum_score=minimum_score,
            maximum_score=maximum_score,
        )

        row = self.connection.execute(
            f"""
            SELECT COUNT(*) AS coloring_count
            FROM colorings
            {where_clause}
            """,
            parameters,
        ).fetchone()

        return int(row["coloring_count"])

    def close(self) -> None:
        """
        Commit outstanding work and close the archive.
        """
        if not self._closed:
            self.connection.commit()
            self.connection.close()
            self._closed = True

    def __enter__(
        self,
    ) -> "RSQLiteArchive":
        self._require_open()
        return self

    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback,
    ) -> None:
        self.close()

    @staticmethod
    def _compatible_state_hash(
        packed_coloring: bytes,
        n_vertices: int,
        k_size: int,
        edge_count: int,
    ) -> str:
        """
        Create the exact identifier used by ColoringDatabase.

        Retaining this format allows existing database files to
        recognize identical colorings saved through the new API.
        """
        header = (f"{n_vertices}:" f"{k_size}:" f"{edge_count}:").encode("ascii")

        return sha256(header + packed_coloring).hexdigest()

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row,
    ) -> RArchiveRecord:
        """
        Convert one SQLite row into immutable metadata.
        """
        return RArchiveRecord(
            coloring_id=int(row["coloring_id"]),
            state_hash=str(row["state_hash"]),
            score=int(row["score"]),
            color_zero_count=int(row["red_kn_count"]),
            color_one_count=int(row["blue_kn_count"]),
            run_name=str(row["last_run"]),
            iteration=int(row["last_iteration"]),
            times_seen=int(row["times_seen"]),
        )

    @staticmethod
    def _problem_key(
        graph: RGraph,
    ) -> tuple[int, int]:
        """
        Return the problem fields supported by the compatible schema.
        """
        problem = graph.problem

        if problem.n_colors != 2 or not problem.is_symmetric:
            raise ValueError(
                "RSQLiteArchive currently supports " "symmetric two-color problems."
            )

        return (
            problem.n_vertices,
            problem.required_clique_sizes[0],
        )

    def _problem_filter(
        self,
        graph: RGraph | None,
    ) -> tuple[
        str,
        tuple[int, ...],
    ]:
        """
        Build the optional SQL problem filter.
        """
        if graph is None:
            return "", ()

        (
            n_vertices,
            k_size,
        ) = self._problem_key(graph)

        return (
            ("WHERE n_vertices = ? " "AND k_size = ?"),
            (
                n_vertices,
                k_size,
            ),
        )

    def _score_range_filter(
        self,
        *,
        graph: RGraph | None,
        minimum_score: int | None,
        maximum_score: int | None,
    ) -> tuple[
        str,
        tuple[int, ...],
    ]:
        """
        Build a validated SQL problem-and-score filter.
        """
        minimum_score = self._validate_optional_nonnegative_integer(
            "minimum_score",
            minimum_score,
        )

        maximum_score = self._validate_optional_nonnegative_integer(
            "maximum_score",
            maximum_score,
        )

        if (
            minimum_score is not None
            and maximum_score is not None
            and minimum_score > maximum_score
        ):
            raise ValueError(
                "minimum_score cannot be greater than "
                "maximum_score."
            )

        clauses: list[str] = []
        parameters: list[int] = []

        if graph is not None:
            (
                n_vertices,
                k_size,
            ) = self._problem_key(graph)

            clauses.extend(
                (
                    "n_vertices = ?",
                    "k_size = ?",
                )
            )

            parameters.extend(
                (
                    n_vertices,
                    k_size,
                )
            )

        if minimum_score is not None:
            clauses.append("score >= ?")
            parameters.append(minimum_score)

        if maximum_score is not None:
            clauses.append("score <= ?")
            parameters.append(maximum_score)

        where_clause = ""

        if clauses:
            where_clause = "WHERE " + " AND ".join(clauses)

        return (
            where_clause,
            tuple(parameters),
        )

    @staticmethod
    def _validate_provenance(
        run_name: str,
        iteration: int,
    ) -> None:
        """
        Validate run metadata stored with a coloring.
        """
        if not isinstance(run_name, str) or not run_name.strip():
            raise ValueError("run_name must be a nonempty string.")

        if isinstance(iteration, bool) or not isinstance(
            iteration,
            (int, np.integer),
        ):
            raise TypeError("iteration must be an integer.")

        if iteration < 0:
            raise ValueError("iteration cannot be negative.")

    @staticmethod
    def _validate_optional_nonnegative_integer(
        name: str,
        value: int | None,
    ) -> int | None:
        """
        Validate an optional nonnegative integer.
        """
        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError(f"{name} must be an integer or None.")

        value = int(value)

        if value < 0:
            raise ValueError(f"{name} cannot be negative.")

        return value

    @staticmethod
    def _validate_optional_positive_integer(
        name: str,
        value: int | None,
    ) -> int | None:
        """
        Validate an optional positive integer.
        """
        value = RSQLiteArchive._validate_optional_nonnegative_integer(
            name,
            value,
        )

        if value == 0:
            raise ValueError(
                f"{name} must be positive when supplied."
            )

        return value

    def _require_open(self) -> None:
        """
        Reject operations after the archive has been closed.
        """
        if self._closed:
            raise RuntimeError("Archive is closed.")