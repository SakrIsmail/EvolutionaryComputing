from __future__ import annotations

import argparse
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEED_PATTERN = re.compile(r"^(?P<algorithm>.+?)_seed_(?P<seed>[^.]+)\.db$")


def find_databases(paths: list[Path]) -> list[Path]:
    """Expand files and directories into sorted SQLite database paths."""
    databases: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix == ".db":
            databases.add(path)
        elif path.is_dir():
            databases.update(path.glob("*_seed_*.db"))
    return sorted(databases)


def algorithm_name(database_path: Path) -> str:
    """Extract the algorithm name from ``algorithm_seed_value.db``."""
    match = SEED_PATTERN.match(database_path.name)
    if match is None:
        return database_path.stem
    return match.group("algorithm")


def best_fitness_by_generation(database_path: Path) -> dict[int, float]:
    """Return the minimum population fitness for each generation in one run."""
    query = """
        SELECT time_of_birth, time_of_death, fitness_
        FROM individual
        WHERE requires_eval = 0 AND fitness_ IS NOT NULL
    """

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(query).fetchall()

    values_by_generation: dict[int, list[float]] = defaultdict(list)
    for birth, death, fitness in rows:
        if birth is None or death is None:
            continue
        for generation in range(int(birth), int(death) + 1):
            values_by_generation[generation].append(float(fitness))

    return {
        generation: min(fitnesses)
        for generation, fitnesses in values_by_generation.items()
    }


def aggregate_runs(
    database_paths: list[Path],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Aggregate per-generation best fitness across runs of each algorithm."""
    runs_by_algorithm: dict[str, list[dict[int, float]]] = defaultdict(list)
    for database_path in database_paths:
        runs_by_algorithm[algorithm_name(database_path)].append(
            best_fitness_by_generation(database_path),
        )

    aggregated: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for algorithm, runs in sorted(runs_by_algorithm.items()):
        generations = np.array(
            sorted(set().union(*(run.keys() for run in runs))),
            dtype=int,
        )
        matrix = np.full((len(runs), len(generations)), np.nan)

        for row, run in enumerate(runs):
            for column, generation in enumerate(generations):
                if generation in run:
                    matrix[row, column] = run[generation]

        with np.errstate(all="ignore"):
            mean = np.nanmean(matrix, axis=0)
            std = np.nanstd(matrix, axis=0)
        aggregated[algorithm] = (generations, mean, std)

    return aggregated


def plot_comparison(
    aggregated: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot mean best fitness with a one-standard-deviation band."""
    if not aggregated:
        raise ValueError("No databases were found to plot.")

    figure, axis = plt.subplots(figsize=(10, 6))
    for algorithm, (generations, mean, std) in aggregated.items():
        axis.plot(generations, mean, linewidth=2, label=algorithm)
        axis.fill_between(
            generations,
            mean - std,
            mean + std,
            alpha=0.2,
        )

    axis.set_title("Algorithm comparison across independent runs")
    axis.set_xlabel("Generation")
    axis.set_ylabel("Best fitness in population")
    axis.grid(alpha=0.3)
    axis.legend(title="Algorithm")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        required=True,
        help="Database files and/or directories containing *_seed_*.db files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("__data__/A1_2026/algorithm_comparison.png"),
    )
    args = parser.parse_args()

    database_paths = find_databases(args.runs)
    if not database_paths:
        raise SystemExit("No databases found. Expected names like uniform_seed_42.db.")

    aggregated = aggregate_runs(database_paths)
    plot_comparison(aggregated, args.output)

    for algorithm, (generations, _, _) in aggregated.items():
        print(
            f"{algorithm}: {len(generations)} generations, "
            f"{sum(1 for path in database_paths if algorithm_name(path) == algorithm)} runs",
        )
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
