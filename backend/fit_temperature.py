"""
Fit CALIBRATION_TEMPERATURE (T) on a validation CSV by minimizing negative log-likelihood.

CSV format (header required):
  logit,label
  2.3,1
  -1.1,0
  ...

label: 1 = spoof/fake, 0 = bonafide/real (same convention as the model target).

Usage (from repo root or backend):
  python fit_temperature.py path/to/val_logits.csv

Then set in .env:
  CALIBRATION_TEMPERATURE=<printed T>
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def nll_temperature(logits: list[float], labels: list[int], t: float) -> float:
    if t <= 0:
        return float("inf")
    total = 0.0
    n = len(logits)
    eps = 1e-12
    for z, y in zip(logits, labels):
        p = sigmoid(z / t)
        p = min(max(p, eps), 1.0 - eps)
        if y == 1:
            total -= math.log(p)
        else:
            total -= math.log(1.0 - p)
    return total / n


def load_csv(path: Path) -> tuple[list[float], list[int]]:
    logits: list[float] = []
    labels: list[int] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "logit" not in r.fieldnames or "label" not in r.fieldnames:
            raise SystemExit(
                "CSV must have columns: logit,label (header row required)."
            )
        for row in r:
            logits.append(float(row["logit"].strip()))
            labels.append(int(float(row["label"].strip())))
    if not logits:
        raise SystemExit("No rows in CSV.")
    return logits, labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit temperature T for sigmoid(logit/T).")
    ap.add_argument("csv", type=Path, help="CSV with columns logit,label")
    ap.add_argument(
        "--grid",
        type=int,
        default=4000,
        help="Number of T samples between t_min and t_max (default 4000).",
    )
    args = ap.parse_args()
    path = args.csv
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    logits, labels = load_csv(path)
    # Search in a range that avoids T→0 (which overfits separable toy data and is not useful calibration).
    lo, hi = 0.5, 10.0
    best_t, best_nll = lo, float("inf")
    for i in range(args.grid):
        t = lo + (hi - lo) * i / max(args.grid - 1, 1)
        nll = nll_temperature(logits, labels, t)
        if nll < best_nll:
            best_nll = nll
            best_t = t

    # Refine around best (simple local search)
    step = (hi - lo) / args.grid
    for _ in range(50):
        for delta in (-step, step):
            t = max(0.05, best_t + delta)
            nll = nll_temperature(logits, labels, t)
            if nll < best_nll:
                best_nll = nll
                best_t = t
        step *= 0.5
        if step < 1e-6:
            break

    print(f"Samples: {len(logits)}")
    print(f"Best T (min mean NLL): {best_t:.4f}")
    print(f"Mean NLL at T: {best_nll:.6f}")
    print()
    print("Add to backend/.env:")
    print(f"CALIBRATION_TEMPERATURE={best_t:.4f}")


if __name__ == "__main__":
    main()
