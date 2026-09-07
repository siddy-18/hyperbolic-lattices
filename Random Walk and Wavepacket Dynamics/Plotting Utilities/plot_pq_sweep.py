#!/usr/bin/env python3
"""
Parse a .log file containing lines like:

2026-09-01 14:13:48,239 - INFO - Computed info for p=3, q=7: Alpha: 0.8146837571514748, Height Diff: 0.4292663849998326

and generate four sets of plots into separate subfolders:

  1. fixed_p_alpha_vs_q/      -> for each p, plot q (x) vs alpha (y)
  2. fixed_p_heightdiff_vs_q/ -> for each p, plot q (x) vs height diff (y)
  3. fixed_q_alpha_vs_p/      -> for each q, plot p (x) vs alpha (y)
  4. fixed_q_heightdiff_vs_p/ -> for each q, plot p (x) vs height diff (y)

Usage:
    python plot_pq_sweep.py path/to/logfile.log [--outdir OUTPUT_DIR]
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

LOG_PATTERN = re.compile(
    r"Computed info for p=(?P<p>\d+),\s*q=(?P<q>\d+):\s*"
    r"Alpha:\s*(?P<alpha>[-+eE0-9.]+),\s*"
    r"Height Diff:\s*(?P<hdiff>[-+eE0-9.]+)"
)


def parse_log(log_path: Path):
    """Parse the log file and return a list of (p, q, alpha, height_diff) tuples."""
    records = []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            # Only consider successfully computed lines; skip ERROR lines etc.
            match = LOG_PATTERN.search(line)
            if match:
                p = int(match.group("p"))
                q = int(match.group("q"))
                alpha = float(match.group("alpha"))
                hdiff = float(match.group("hdiff"))
                records.append((p, q, alpha, hdiff))
    return records


def group_by_p(records):
    """dict: p -> list of (q, alpha, hdiff), sorted by q"""
    grouped = defaultdict(list)
    for p, q, alpha, hdiff in records:
        grouped[p].append((q, alpha, hdiff))
    for p in grouped:
        grouped[p].sort(key=lambda t: t[0])
    return grouped


def group_by_q(records):
    """dict: q -> list of (p, alpha, hdiff), sorted by p"""
    grouped = defaultdict(list)
    for p, q, alpha, hdiff in records:
        grouped[q].append((p, alpha, hdiff))
    for q in grouped:
        grouped[q].sort(key=lambda t: t[0])
    return grouped


def make_plot(x, y, title, xlabel, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, marker="o", linestyle="-")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot p/q sweep quantities from a log file.")
    parser.add_argument("logfile", type=str, help="Path to the .log file")
    parser.add_argument(
        "--outdir",
        type=str,
        default="pq_plots",
        help="Root output directory for the generated plots (default: pq_plots)",
    )
    args = parser.parse_args()

    log_path = Path(args.logfile)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    records = parse_log(log_path)
    if not records:
        raise ValueError("No matching 'Computed info for p=..., q=...' lines found in the log file.")

    print(f"Parsed {len(records)} records from {log_path}")

    out_root = Path(args.outdir)
    subfolders = {
        "fixed_p_alpha_vs_q": out_root / "fixed_p_alpha_vs_q",
        "fixed_p_heightdiff_vs_q": out_root / "fixed_p_heightdiff_vs_q",
        "fixed_q_alpha_vs_p": out_root / "fixed_q_alpha_vs_p",
        "fixed_q_heightdiff_vs_p": out_root / "fixed_q_heightdiff_vs_p",
    }
    for folder in subfolders.values():
        folder.mkdir(parents=True, exist_ok=True)

    # --- 1 & 2: Fix p, vary q ---
    by_p = group_by_p(records)
    for p, entries in by_p.items():
        qs = [e[0] for e in entries]
        alphas = [e[1] for e in entries]
        hdiffs = [e[2] for e in entries]

        # 1. q vs alpha
        make_plot(
            qs, alphas,
            title=f"p={p}: Alpha vs q",
            xlabel="q",
            ylabel="Alpha",
            out_path=subfolders["fixed_p_alpha_vs_q"] / f"p{p}_alpha_vs_q.png",
        )

        # 2. q vs height diff
        make_plot(
            qs, hdiffs,
            title=f"p={p}: Height Diff vs q",
            xlabel="q",
            ylabel="Height Diff",
            out_path=subfolders["fixed_p_heightdiff_vs_q"] / f"p{p}_heightdiff_vs_q.png",
        )

    # --- 3 & 4: Fix q, vary p ---
    by_q = group_by_q(records)
    for q, entries in by_q.items():
        ps = [e[0] for e in entries]
        alphas = [e[1] for e in entries]
        hdiffs = [e[2] for e in entries]

        # 3. p vs alpha
        make_plot(
            ps, alphas,
            title=f"q={q}: Alpha vs p",
            xlabel="p",
            ylabel="Alpha",
            out_path=subfolders["fixed_q_alpha_vs_p"] / f"q{q}_alpha_vs_p.png",
        )

        # 4. p vs height diff
        make_plot(
            ps, hdiffs,
            title=f"q={q}: Height Diff vs p",
            xlabel="p",
            ylabel="Height Diff",
            out_path=subfolders["fixed_q_heightdiff_vs_p"] / f"q{q}_heightdiff_vs_p.png",
        )

    print(f"Done. Plots written under: {out_root.resolve()}")
    for name, folder in subfolders.items():
        n = len(list(folder.glob("*.png")))
        print(f"  {name}: {n} plot(s)")


if __name__ == "__main__":
    main()
