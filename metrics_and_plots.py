"""Turn a q_table.py report into the results table and the learning-curve figure."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
CURVE_SERIES = {
    "baseline": ("#2a78d6", "Baseline"),
    "count": ("#eb6834", "+ count"),
    "double": ("#1baf7a", "+ double"),
    "count_double": ("#eda100", "+ count + double"),
}
SHAPING_SERIES = {
    "double": ("#1baf7a", "no shaping"),
    "bust_0.1": ("#e45756", "bust penalty"),
    "shaped_heuristic": ("#7a5cd6", "total/21"),
    "shaped_value": ("#eda100", "BS value"),
}
REFERENCE_LINES = {
    "basic_strategy": "Basic strategy, hit/stand",
    "basic_strategy_double": "Basic strategy + double",
}


def mean_ci(values: list[float]) -> tuple[float, float]:
    """Mean and 95% t-interval half-width across seeds."""
    if len(values) < 2:
        return values[0], float("nan")
    half_width = T_95[len(values)] * statistics.stdev(values) / len(values) ** 0.5
    return statistics.fmean(values), half_width


def _finals_by_seed(rows: list[dict], key: str) -> dict[str, dict[int, dict]]:
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row[key]][row["seed"]] = row["final"]
    return grouped


def merge_reports(reports: list[dict]) -> dict:
    """Combine reports run with the same settings; seeded runs stay paired by seed."""
    merged = {"settings": reports[0]["settings"], "stages": [], "references": [], "runs": []}
    seen_stages, seen_runs, seen_references = set(), set(), set()
    for report in reports:
        settings = report["settings"]
        for key in settings.keys() & merged["settings"].keys() - {"seconds"}:
            if settings[key] != merged["settings"][key]:
                raise ValueError(f"reports differ in {key}: "
                                 f"{merged['settings'][key]!r} vs {settings[key]!r}")
        for stage in report["stages"]:
            if stage["name"] not in seen_stages:
                seen_stages.add(stage["name"])
                merged["stages"].append(stage)
        for run in report["runs"]:
            if (run["stage"], run["seed"]) not in seen_runs:
                seen_runs.add((run["stage"], run["seed"]))
                merged["runs"].append(run)
        for reference in report["references"]:
            if (reference["policy"], reference["seed"]) not in seen_references:
                seen_references.add((reference["policy"], reference["seed"]))
                merged["references"].append(reference)
    return merged


def results_table(
    report: dict,
    stage_names: list[str] | None = None,
    comparisons: list[tuple[str, str]] = (),
) -> str:
    """Markdown table; deltas are paired by seed, so both sides saw the same eval shoe."""
    stages = _finals_by_seed(report["runs"], "stage")
    references = _finals_by_seed(report["references"], "policy")
    states = {row["stage"]: row["states"] for row in report["runs"]}
    labels = {stage["name"]: stage["label"] for stage in report["stages"]}
    baseline = stages["baseline"]

    def row(label: str, finals: dict[int, dict], table_size: str) -> str:
        mean, half_width = mean_ci([final["mean_return"] for final in finals.values()])
        delta, delta_width = mean_ci(
            [finals[seed]["mean_return"] - baseline[seed]["mean_return"] for seed in finals]
        )
        win = statistics.fmean(final["win_rate"] for final in finals.values())
        bust = statistics.fmean(final["bust_rate"] for final in finals.values())
        double = statistics.fmean(final["double_rate"] for final in finals.values())
        return (
            f"| {label} | {table_size} | {mean:+.4f} ± {half_width:.4f} "
            f"| {delta:+.4f} ± {delta_width:.4f} | {win:.1%} | {bust:.1%} | {double:.1%} |"
        )

    lines = [
        "| Policy | Table states | Mean return / hand | Change vs baseline | Win | Bust | Double |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in stage_names or [stage["name"] for stage in report["stages"]]:
        lines.append(row(labels[name], stages[name], f"{states[name]:,}"))
    for name, finals in references.items():
        lines.append(row(f"*Reference: {name.replace('_', ' ')}*", finals, "—"))
    if comparisons:
        lines += ["", "| Paired difference | Mean return / hand |", "|---|---:|"]
        for first, second in comparisons:
            if stages[first].keys() != stages[second].keys():
                raise ValueError(f"{first} and {second} were not run with the same seeds")
            delta, delta_width = mean_ci([
                stages[first][seed]["mean_return"] - stages[second][seed]["mean_return"]
                for seed in stages[first]
            ])
            lines.append(f"| {labels[first]} − ({labels[second]}) "
                         f"| {delta:+.4f} ± {delta_width:.4f} |")
    return "\n".join(lines) + "\n"


def learning_curve(report: dict, path: Path, series: dict = CURVE_SERIES) -> None:
    labels = {stage["name"]: stage["label"] for stage in report["stages"]}
    curves: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in report["runs"]:
        for point in run["curve"]:
            curves[run["stage"]][point["train_hands"]].append(point["mean_return"])
    references = _finals_by_seed(report["references"], "policy")

    figure, axes = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    for name, (color, short_label) in series.items():
        if name not in curves:
            continue
        budgets = sorted(curves[name])
        means = [statistics.fmean(curves[name][budget]) for budget in budgets]
        axes.plot(budgets, means, color=color, linewidth=2, marker="o", markersize=5,
                  label=labels[name])
        if series is CURVE_SERIES:
            axes.annotate(short_label, (budgets[0], means[0]), xytext=(4, -11),
                          textcoords="offset points", fontsize=8, color=INK)
    for name, label in REFERENCE_LINES.items():
        level = statistics.fmean(final["mean_return"] for final in references[name].values())
        axes.axhline(level, color=MUTED, linewidth=1, linestyle="--")
        axes.annotate(label, (0.01, level), xycoords=("axes fraction", "data"),
                      xytext=(0, 3), textcoords="offset points", fontsize=8, color=MUTED)
    axes.set_xscale("log")
    axes.set_xlabel("Training hands", color=INK)
    axes.set_ylabel("Mean return per hand (true reward)", color=INK)
    axes.set_title("Learning curves, mean of training seeds", loc="left", fontsize=11, color=INK)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(MUTED)
    axes.legend(frameon=False, fontsize=8, loc="lower right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+",
                        default=[Path("outputs/q_table_ladder.json")],
                        help="one report, or several run with the same settings")
    parser.add_argument("--table", type=Path, default=Path("outputs/q_table_results.md"))
    parser.add_argument("--figure", type=Path, default=Path("outputs/learning_curves.png"))
    parser.add_argument("--series", choices=("ladder", "shaping", "none"), default="ladder",
                        help="'none' skips the learning-curve figure")
    parser.add_argument("--stages", nargs="*", help="rows to show, in order; default: all")
    parser.add_argument("--compare", nargs="*", default=[], metavar="A:B",
                        help="add the paired difference 'stage A minus stage B'")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = merge_reports([json.loads(path.read_text(encoding="utf-8")) for path in args.input])
    comparisons = [tuple(pair.split(":")) for pair in args.compare]
    table = results_table(report, args.stages, comparisons)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(table, encoding="utf-8")
    if args.series != "none":
        series = SHAPING_SERIES if args.series == "shaping" else CURVE_SERIES
        learning_curve(report, args.figure, series)
    print(table)


if __name__ == "__main__":
    main()
