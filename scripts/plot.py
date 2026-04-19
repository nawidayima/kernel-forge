#!/usr/bin/env python3
"""Plot kernel-variant performance curves from a benchmark CSV.

Reads `benchmark_results/<exercise>.csv`, groups rows by `kernel`, plots
the dominant throughput metric (`gflops` or `gbps`) against the primary size
dimension. Saves a PNG next to the CSV.

Run: python3 plot.py benchmark_results/matmul.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


# Size axis and metric per exercise. Add new rows as exercises are added.
EXERCISE_AXIS: dict[str, tuple[str, str]] = {
    "matmul":           ("M",      "gflops"),
    "reduction":        ("N",      "gbps"),
    "cross_entropy":    ("V",      "gbps"),
    "flash_attention":  ("N",      "gflops"),
    "quantized_gemm":   ("M",      "gflops"),
    "moe_dispatch":     ("T",      "gflops"),
}


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: plot.py <csv>", file=sys.stderr)
        sys.exit(2)
    csv_path = Path(sys.argv[1])
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"no rows in {csv_path}", file=sys.stderr)
        sys.exit(1)

    exercise = rows[0].get("exercise", csv_path.stem)
    if exercise not in EXERCISE_AXIS:
        print(f"unknown exercise '{exercise}'; add it to EXERCISE_AXIS", file=sys.stderr)
        sys.exit(1)
    x_key, y_key = EXERCISE_AXIS[exercise]

    series: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        try:
            k = int(r["kernel"])
            x = float(r[x_key])
            y = float(r[y_key])
        except (KeyError, ValueError):
            continue
        series[k].append((x, y))

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in sorted(series):
        pts = sorted(series[k])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = f"kernel {k}" + (" (cuBLAS)" if k == 0 and exercise == "matmul" else "")
        ax.plot(xs, ys, marker="o", label=label)

    ax.set_xscale("log", base=2)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key.upper())
    ax.set_title(f"{exercise}: kernel variants")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    png = csv_path.with_suffix(".png")
    fig.savefig(png, dpi=120)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
