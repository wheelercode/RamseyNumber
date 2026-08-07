"""Persistent archive interface and SQLite-backed store for exact colorings.

Defines :class:`RArchive`, the abstract persistence contract used to save
and query exact colorings discovered during search, and
:class:`RSQLiteArchive`, a concrete implementation backed by a SQLite
database that deduplicates colorings by an exact content hash and tracks
provenance (which run/iteration produced each observation) and a
leaderboard ordered by score.
"""

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
    """Metadata for one unique archived coloring.

    Represents a single deduplicated row of the archive: the exact score
    and provenance of a coloring, without the coloring data itself. Two
    colorings that hash identically are merged into one record, with
    ``times_seen`` counting every observation and ``run_name``/
    ``iteration`` reflecting the most recent one.

    Attributes:
        coloring_id (int): Archive-assigned primary key identifying the
            stored coloring, used to retrieve the full coloring via
            :meth:`RArchive.load_coloring`.
        state_hash (str): Exact content hash used to deduplicate
            colorings; identical colorings for the same problem share a
            record.
        score (int): Exact score of the coloring, i.e. the total count of
            forbidden monochromatic cliques (lower is better).
        color_zero_count (int): Count of forbidden cliques monochromatic
            in color zero.
        color_one_count (int): Count of forbidden cliques monochromatic
            in color one.
        run_name (str): Name of the most recent run that observed this
            coloring.
        iteration (int): Iteration index, within ``run_name``, of the
            most recent observation.
        times_seen (int): Total number of times this exact coloring has
            been saved to the archive, across all runs.
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
    """One restored coloring with its stored metadata and histogram.

    Returned by :meth:`RArchive.load_coloring` when reconstructing a
    previously archived coloring against a caller-supplied, compatible
    host graph.

    Attributes:
        record (RArchiveRecord): Archive metadata for this coloring
            (score, provenance, and deduplication information).
        coloring (RColoring): Restored coloring on the caller's graph.
        histogram (numpy.ndarray): Read-only ``int64`` array of length
            ``edges_per_clique + 1`` giving, for each possible per-clique
            same-color edge count, the number of indexed cliques with
            that count for the archived color (the "legacy" binary
            histogram, see :func:`ramsey.RScoring.binary_histogram`).
    """

    record: RArchiveRecord
    coloring: RColoring
    histogram: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Copy the supplied histogram and freeze it against mutation.

        A defensive copy is taken so that the stored histogram is
        independent of any array the caller continues to hold, and the
        copy's ``writeable`` flag is cleared so callers cannot corrupt
        archived metadata in place.
        """
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
    """Persistent storage interface for exact Ramsey colorings.

    Implementations save colorings discovered during search or
    construction, deduplicate them by exact content, and expose
    leaderboard-style queries (best score, best colorings, colorings
    within a score range) that batch drivers such as
    :class:`ramsey.RArchiveBatch.RArchiveBatch` use to decide when a
    score band has been sufficiently populated.
    """

    @abstractmethod
    def save_coloring(
        self,
        coloring: RColoring,
        *,
        run_name: str,
        iteration: int,
    ) -> RArchiveRecord:
        """Save or observe one exact coloring.

        Args:
            coloring (RColoring): Coloring to persist. Implementations
                compute score and deduplication data from the coloring
                itself rather than trusting caller-supplied metadata.
            run_name (str): Name of the run performing the save, used as
                provenance.
            iteration (int): Iteration index, within ``run_name``, at
                which the coloring was produced.

        Returns:
            RArchiveRecord: Metadata for the archived coloring. If an
            identical coloring already exists, this reflects the merged
            record (incremented ``times_seen``, updated provenance)
            rather than a newly created one.
        """
        ...

    @abstractmethod
    def load_coloring(
        self,
        coloring_id: int,
        graph: RGraph,
    ) -> RArchivedColoring:
        """Restore one coloring using a compatible graph.

        Args:
            coloring_id (int): Archive-assigned identifier of the
                coloring to restore.
            graph (RGraph): Host graph the restored coloring will be
                attached to. Must describe the same problem (vertex
                count, clique size, edge count) as the one the coloring
                was originally archived under.

        Returns:
            RArchivedColoring: The restored coloring together with its
            archive metadata and histogram.
        """
        ...

    @abstractmethod
    def best_score(
        self,
        graph: RGraph | None = None,
    ) -> int | None:
        """Return the lowest stored score.

        Args:
            graph (RGraph | None): If supplied, restrict the search to
                colorings belonging to this graph's problem.

        Returns:
            int | None: The minimum score among matching colorings, or
            ``None`` if the archive (or the matching subset) is empty.
        """
        ...

    @abstractmethod
    def best_colorings(
        self,
        limit: int = 10,
        graph: RGraph | None = None,
    ) -> list[RArchiveRecord]:
        """Return metadata for the best stored colorings.

        Args:
            limit (int): Maximum number of records to return.
            graph (RGraph | None): If supplied, restrict results to
                colorings belonging to this graph's problem.

        Returns:
            list[RArchiveRecord]: Records ordered by ascending score
            (and, for ties, ascending coloring ID), truncated to
            ``limit`` entries.
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
        """Return records in an inclusive score range.

        Args:
            minimum_score (int | None): Inclusive lower bound on score,
                or ``None`` for no lower bound.
            maximum_score (int | None): Inclusive upper bound on score,
                or ``None`` for no upper bound.
            limit (int | None): Maximum number of records to return, or
                ``None`` for no limit.
            graph (RGraph | None): If supplied, restrict results to
                colorings belonging to this graph's problem.

        Returns:
            list[RArchiveRecord]: Matching records ordered by ascending
            score (and, for ties, ascending coloring ID).
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
        """Count unique colorings in an inclusive score range.

        Args:
            minimum_score (int | None): Inclusive lower bound on score,
                or ``None`` for no lower bound.
            maximum_score (int | None): Inclusive upper bound on score,
                or ``None`` for no upper bound.
            graph (RGraph | None): If supplied, restrict the count to
                colorings belonging to this graph's problem.

        Returns:
            int: Number of unique stored colorings whose score falls
            within the inclusive bounds.
        """
        ...

    @abstractmethod
    def coloring_count(
        self,
        graph: RGraph | None = None,
    ) -> int:
        """Return the number of unique stored colorings.

        Args:
            graph (RGraph | None): If supplied, restrict the count to
                colorings belonging to this graph's problem.

        Returns:
            int: Number of unique stored colorings.
        """
        ...


class RSQLiteArchive(RArchive):
    """SQLite archive compatible with the original coloring database.

    Colorings are stored in a single ``colorings`` table keyed by an
    exact ``state_hash`` (see :meth:`_compatible_state_hash`) so that
    identical colorings collapse to one row (``ON CONFLICT`` updates
    ``times_seen`` and the most recent provenance instead of inserting a
    duplicate). Only symmetric two-color problems are supported, since
    the schema (and the legacy ``red_kn_count``/``blue_kn_count``
    columns) predates general multi-color/asymmetric problems.

    Binary colorings are packed into bits. A K43 coloring therefore
    occupies 113 bytes rather than 903 bytes.

    The connection uses WAL journaling and ``synchronous = NORMAL`` so
    that concurrent readers (e.g. other processes in
    :class:`ramsey.RArchiveBatchParallel.RArchiveBatchParallel`) are not
    blocked by writers, while writes that change the ``colorings`` table
    are wrapped in a transaction via ``with self.connection:``.
    """

    def __init__(
        self,
        database_path: str | Path,
    ) -> None:
        """Open (creating if necessary) a SQLite archive database.

        Creates the parent directory of ``database_path`` if it does not
        exist, opens the SQLite connection with row access by column
        name, enables foreign keys, switches to WAL journaling with
        ``synchronous = NORMAL`` for better concurrent read/write
        behavior, and ensures the ``colorings`` schema and indexes exist.

        Args:
            database_path (str | Path): Filesystem path to the SQLite
                database file. Created if it does not already exist.
        """
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
        """Create the compatible coloring table and indexes.

        Creates, if not already present, the ``colorings`` table (keyed
        by ``coloring_id`` with a unique ``state_hash``) and two indexes
        supporting the leaderboard queries: one on ``(score,
        coloring_id)`` for problem-agnostic score queries, and one on
        ``(n_vertices, k_size, score, coloring_id)`` for queries scoped
        to a specific problem. Idempotent: safe to call on an existing
        database.
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
        """Save or observe one exact coloring.

        The histogram and score are calculated from the coloring
        inside the archive boundary. Callers cannot accidentally
        supply metadata belonging to a different coloring.

        If the coloring already exists (same ``state_hash``), its
        ``times_seen`` counter is incremented and its ``last_run``/
        ``last_iteration`` provenance is updated to this call's values,
        rather than inserting a new row.

        Args:
            coloring (RColoring): Coloring to persist. Must belong to a
                symmetric two-color problem.
            run_name (str): Nonempty name of the run performing the
                save, used as provenance.
            iteration (int): Nonnegative iteration index, within
                ``run_name``, at which the coloring was produced.

        Returns:
            RArchiveRecord: Metadata for the archived (or re-observed)
            coloring, read back from the database after the write.

        Raises:
            ValueError: If ``coloring``'s problem is not a symmetric
                two-color problem, ``run_name`` is empty, or
                ``iteration`` is negative.
            TypeError: If ``iteration`` is not an integer.
            RuntimeError: If the archive has been closed, or if the row
                cannot be re-read immediately after being written (this
                should not occur under normal operation).
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
        """Restore one coloring using a compatible graph.

        Unpacks the stored bit-packed coloring back into a per-edge
        color array (matching ``graph``'s edge ordering) and decodes the
        stored histogram bytes back into an ``int64`` array.

        Args:
            coloring_id (int): Archive-assigned identifier of the
                coloring to restore.
            graph (RGraph): Host graph the restored coloring will be
                attached to.

        Returns:
            RArchivedColoring: The restored coloring, its archive
            metadata, and its histogram.

        Raises:
            RuntimeError: If the archive has been closed.
            KeyError: If no coloring with ``coloring_id`` exists.
            ValueError: If ``graph``'s problem (vertex count, clique
                size, or edge count) does not match the problem the
                coloring was archived under.
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
        """Return the lowest stored score.

        If graph is supplied, only colorings belonging to that
        problem are considered.

        Args:
            graph (RGraph | None): If supplied, restrict the search to
                colorings belonging to this graph's problem.

        Returns:
            int | None: The minimum score among matching colorings, or
            ``None`` if no matching coloring is stored.

        Raises:
            RuntimeError: If the archive has been closed.
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
        """Return metadata for the best stored colorings.

        Results are ordered by score and then coloring ID.

        Args:
            limit (int): Maximum number of records to return. Must be a
                positive integer.
            graph (RGraph | None): If supplied, restrict results to
                colorings belonging to this graph's problem.

        Returns:
            list[RArchiveRecord]: Up to ``limit`` records ordered by
            ascending score, then ascending coloring ID.

        Raises:
            RuntimeError: If the archive has been closed.
            TypeError: If ``limit`` is not an integer.
            ValueError: If ``limit`` is not positive.
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
        """Return the number of unique stored colorings.

        If graph is supplied, only colorings belonging to that
        problem are counted.

        Args:
            graph (RGraph | None): If supplied, restrict the count to
                colorings belonging to this graph's problem.

        Returns:
            int: Number of unique stored colorings.

        Raises:
            RuntimeError: If the archive has been closed.
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
        """Return records ordered by score within inclusive bounds.

        Args:
            minimum_score (int | None): Inclusive lower bound on score,
                or ``None`` for no lower bound.
            maximum_score (int | None): Inclusive upper bound on score,
                or ``None`` for no upper bound.
            limit (int | None): Maximum number of records to return, or
                ``None`` for no limit.
            graph (RGraph | None): If supplied, restrict results to
                colorings belonging to this graph's problem.

        Returns:
            list[RArchiveRecord]: Matching records ordered by ascending
            score, then ascending coloring ID.

        Raises:
            RuntimeError: If the archive has been closed.
            TypeError: If ``minimum_score``, ``maximum_score``, or
                ``limit`` is not an integer or ``None``.
            ValueError: If ``minimum_score`` exceeds ``maximum_score``,
                if either score bound is negative, or if ``limit`` is
                not positive.
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
        """Count bounded records without loading their metadata.

        Args:
            minimum_score (int | None): Inclusive lower bound on score,
                or ``None`` for no lower bound.
            maximum_score (int | None): Inclusive upper bound on score,
                or ``None`` for no upper bound.
            graph (RGraph | None): If supplied, restrict the count to
                colorings belonging to this graph's problem.

        Returns:
            int: Number of unique stored colorings whose score falls
            within the inclusive bounds.

        Raises:
            RuntimeError: If the archive has been closed.
            TypeError: If ``minimum_score`` or ``maximum_score`` is not
                an integer or ``None``.
            ValueError: If ``minimum_score`` exceeds ``maximum_score``
                or either bound is negative.
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
        """Commit outstanding work and close the archive.

        Idempotent: calling this more than once has no effect after the
        first call.
        """
        if not self._closed:
            self.connection.commit()
            self.connection.close()
            self._closed = True

    def __enter__(
        self,
    ) -> "RSQLiteArchive":
        """Enter a context block, verifying the archive is open.

        Returns:
            RSQLiteArchive: This archive instance.

        Raises:
            RuntimeError: If the archive has already been closed.
        """
        self._require_open()
        return self

    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback,
    ) -> None:
        """Close the archive on exit from a context block."""
        self.close()

    @staticmethod
    def _compatible_state_hash(
        packed_coloring: bytes,
        n_vertices: int,
        k_size: int,
        edge_count: int,
    ) -> str:
        """Create the exact identifier used by ColoringDatabase.

        Retaining this format allows existing database files to
        recognize identical colorings saved through the new API.

        Args:
            packed_coloring (bytes): Bit-packed coloring bytes, as
                returned by :meth:`ramsey.RColoring.RColoring.packed`.
            n_vertices (int): Number of vertices in the host graph.
            k_size (int): Required clique size for the (symmetric,
                two-color) problem.
            edge_count (int): Number of edges in the host graph.

        Returns:
            str: SHA-256 hex digest of the problem header concatenated
            with the packed coloring bytes, used as the ``state_hash``
            deduplication key.
        """
        header = (f"{n_vertices}:" f"{k_size}:" f"{edge_count}:").encode("ascii")

        return sha256(header + packed_coloring).hexdigest()

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row,
    ) -> RArchiveRecord:
        """Convert one SQLite row into immutable metadata.

        Args:
            row (sqlite3.Row): Row from the ``colorings`` table.

        Returns:
            RArchiveRecord: Immutable metadata built from the row's
            columns, using the row's ``last_run``/``last_iteration``
            values as ``run_name``/``iteration``.
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
        """Return the problem fields supported by the compatible schema.

        Args:
            graph (RGraph): Host graph whose problem is being
                identified.

        Returns:
            tuple[int, int]: The graph's vertex count and required
            clique size, i.e. the ``(n_vertices, k_size)`` pair used to
            scope archive queries to a specific problem.

        Raises:
            ValueError: If the graph's problem is not a symmetric
                two-color problem.
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
        """Build the optional SQL problem filter.

        Args:
            graph (RGraph | None): If supplied, the resulting clause
                restricts to this graph's problem; otherwise no
                filtering is applied.

        Returns:
            tuple[str, tuple[int, ...]]: A ``WHERE`` clause (or an empty
            string if ``graph`` is ``None``) and its bound parameters,
            ready to be interpolated into a query and passed to
            ``execute``.
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
        """Build a validated SQL problem-and-score filter.

        Args:
            graph (RGraph | None): If supplied, the resulting clause
                restricts to this graph's problem.
            minimum_score (int | None): Inclusive lower bound on score,
                or ``None`` for no lower bound.
            maximum_score (int | None): Inclusive upper bound on score,
                or ``None`` for no upper bound.

        Returns:
            tuple[str, tuple[int, ...]]: A ``WHERE`` clause (empty if no
            filters apply) and its bound parameters, ready to be
            interpolated into a query and passed to ``execute``.

        Raises:
            TypeError: If ``minimum_score`` or ``maximum_score`` is not
                an integer or ``None``.
            ValueError: If ``minimum_score`` exceeds ``maximum_score``,
                or if either score bound is negative, or if ``graph``'s
                problem is not a symmetric two-color problem.
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
        """Validate run metadata stored with a coloring.

        Args:
            run_name (str): Candidate run name; must be a nonempty
                string.
            iteration (int): Candidate iteration index; must be a
                nonnegative integer.

        Raises:
            ValueError: If ``run_name`` is empty (or whitespace-only) or
                ``iteration`` is negative.
            TypeError: If ``iteration`` is not an integer.
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
        """Validate an optional nonnegative integer.

        Args:
            name (str): Parameter name, used in error messages.
            value (int | None): Candidate value, or ``None``.

        Returns:
            int | None: ``None`` if ``value`` is ``None``, otherwise
            ``value`` coerced to ``int``.

        Raises:
            TypeError: If ``value`` is not ``None`` and not an integer.
            ValueError: If ``value`` is negative.
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
        """Validate an optional positive integer.

        Args:
            name (str): Parameter name, used in error messages.
            value (int | None): Candidate value, or ``None``.

        Returns:
            int | None: ``None`` if ``value`` is ``None``, otherwise
            ``value`` coerced to ``int``.

        Raises:
            TypeError: If ``value`` is not ``None`` and not an integer.
            ValueError: If ``value`` is negative or zero.
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
        """Reject operations after the archive has been closed.

        Raises:
            RuntimeError: If the archive has been closed.
        """
        if self._closed:
            raise RuntimeError("Archive is closed.")