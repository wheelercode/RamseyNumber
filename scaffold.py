"""Create the RamseySearch refactor filesystem without overwriting work.

Run this script from the ``refactor`` directory:

    python scaffold_ramsey_package.py

Preview the changes without creating anything:

    python scaffold_ramsey_package.py --dry-run

By default, the repository root is the parent of this script's ``refactor``
directory. An alternate root can be supplied with ``--root``. Existing files
are always preserved; this script intentionally has no overwrite option.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parent


DIRECTORIES = (
    "ramsey",
    "ramsey/nn",
    "tests",
    "notebooks",
    "data",
    "checkpoints",
)


MODULE_PURPOSES = {
    "ramsey/RProblem.py": (
        "Mathematical specifications for finite Ramsey coloring problems."
    ),
    "ramsey/RGraph.py": (
        "Immutable host-graph topology and precomputed incidence indexes."
    ),
    "ramsey/RColoring.py": (
        "Validated, immutable edge-coloring values and conversions."
    ),
    "ramsey/RScoring.py": (
        "Exact score, histogram, and monochromatic-subgraph calculations."
    ),
    "ramsey/RState.py": ("Mutable incremental state for one active search attempt."),
    "ramsey/RAction.py": (
        "Consequences and exact score changes for candidate search actions."
    ),
    "ramsey/RObjective.py": (
        "Optimization objectives and reward shaping derived from action data."
    ),
    "ramsey/RMemory.py": (
        "Tabu, visited-state, and other search-history restrictions."
    ),
    "ramsey/REnvironment.py": ("Validation and application of state transitions."),
    "ramsey/RPolicy.py": ("Interchangeable strategies for selecting the next action."),
    "ramsey/RConstruction.py": (
        "Random, cyclic, archived, and other seed-coloring constructors."
    ),
    "ramsey/RSearch.py": ("Lifecycle and stopping rules for complete search attempts."),
    "ramsey/RArchive.py": (
        "Persistent coloring, run-provenance, and leaderboard storage."
    ),
    "ramsey/RVerification.py": (
        "Independent verification of colorings and incremental state."
    ),
    "ramsey/RExperiment.py": (
        "Reproducible assembly and execution of search experiments."
    ),
    "ramsey/RPlot.py": ("Visualization of completed score reports and search results."),
    "ramsey/nn/REncoding.py": (
        "Conversion of Ramsey search information into model inputs."
    ),
    "ramsey/nn/RModel.py": (
        "PyTorch model definitions and neural forward computation."
    ),
    "ramsey/nn/RNeuralPolicy.py": (
        "Adapter connecting neural model output to the search policy interface."
    ),
    "ramsey/nn/RRollout.py": (
        "Collection of policy interactions into training batches."
    ),
    "ramsey/nn/RPPO.py": (
        "PPO configuration, advantage estimation, and parameter updates."
    ),
    "ramsey/nn/RCheckpoint.py": (
        "Versioned persistence and restoration of training state."
    ),
}


TEST_PURPOSES = {
    "tests/test_graph.py": "Graph enumeration and incidence invariants.",
    "tests/test_coloring.py": "Coloring validation and conversion round trips.",
    "tests/test_scoring.py": "Exact score and histogram calculations.",
    "tests/test_state.py": "Incremental state versus full recomputation.",
    "tests/test_action.py": "Predicted versus actual action consequences.",
    "tests/test_environment.py": "Transition, termination, and masking behavior.",
    "tests/test_construction.py": "Seed constructors and known constructions.",
    "tests/test_archive.py": "Exact coloring and metadata persistence.",
    "tests/test_learning.py": "Masks, equivariance, PPO, and checkpoints.",
}


STATIC_FILES = {
    "ramsey/__init__.py": dedent('''\
        """Ramsey graph construction, scoring, search, and verification."""

        # Public exports are added here as implementations migrate into the
        # package. Keeping this file minimal prevents import cycles.
        '''),
    "ramsey/nn/__init__.py": dedent('''\
        """Optional neural encoding, models, rollouts, and training."""

        # The core ramsey package must never import this package.
        '''),
    "tests/__init__.py": (
        '"""Characterization and regression tests for RamseySearch."""\n'
    ),
    "tests/conftest.py": dedent('''\
        """Shared pytest fixtures for RamseySearch tests."""
        '''),
    "notebooks/README.md": dedent("""\
        # Notebooks

        Notebooks are thin clients of the `ramsey` package. They may assemble
        experiments, run commands, and display results, but reusable functions
        and classes belong in the package.

        Move the existing notebooks here only after their reusable definitions
        have been extracted and their replacement imports work.
        """),
    ".gitignore": dedent("""\
        __pycache__/
        *.py[cod]
        .pytest_cache/
        .ipynb_checkpoints/

        # Virtual environments and build products
        .venv/
        build/
        dist/
        *.egg-info/

        # Runtime artifacts
        data/*.sqlite3
        data/*.sqlite3-*
        checkpoints/*.pt
        """),
}


def module_template(purpose: str) -> str:
    """Return the initial contents of one responsibility-focused module."""
    return f'"""{purpose}"""\n'


def build_file_manifest() -> dict[str, str]:
    """Return every relative file path and its initial contents."""
    files = dict(STATIC_FILES)

    files.update(
        {path: module_template(purpose) for path, purpose in MODULE_PURPOSES.items()}
    )

    files.update(
        {path: module_template(purpose) for path, purpose in TEST_PURPOSES.items()}
    )

    return files


def create_directory(path: Path, *, dry_run: bool) -> str:
    """Create one directory if absent and return its status."""
    if path.is_dir():
        return "exists"

    if path.exists():
        raise FileExistsError(
            f"Cannot create directory because a file exists at {path}"
        )

    if not dry_run:
        path.mkdir(parents=True, exist_ok=False)

    return "would create" if dry_run else "created"


def create_file(path: Path, contents: str, *, dry_run: bool) -> str:
    """Create one file if absent and return its status."""
    if path.exists():
        return "exists"

    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8", newline="\n")

    return "would create" if dry_run else "created"


def scaffold(root: Path, *, dry_run: bool = False) -> dict[str, int]:
    """Create the planned filesystem beneath ``root`` without overwrites."""
    root = root.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Repository root does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Repository root is not a directory: {root}")

    counts = {
        "created": 0,
        "would create": 0,
        "exists": 0,
    }

    for relative_path in DIRECTORIES:
        path = root / relative_path
        status = create_directory(path, dry_run=dry_run)
        counts[status] += 1
        print(f"{status.upper():12} directory  {path.relative_to(root)}")

    for relative_path, contents in build_file_manifest().items():
        path = root / relative_path
        status = create_file(path, contents, dry_run=dry_run)
        counts[status] += 1
        print(f"{status.upper():12} file       {path.relative_to(root)}")

    return counts


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create the RamseySearch package scaffold without overwriting "
            "existing files."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_REPOSITORY_ROOT,
        help=("repository root; defaults to the parent of the refactor " "directory"),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be created without changing the filesystem",
    )

    return parser.parse_args()


def main() -> None:
    """Run the scaffold command and print a summary."""
    arguments = parse_arguments()
    counts = scaffold(
        arguments.root,
        dry_run=arguments.dry_run,
    )

    print()
    print("Summary")
    print("-------")

    for status, count in counts.items():
        print(f"{status:12}: {count}")


if __name__ == "__main__":
    main()
