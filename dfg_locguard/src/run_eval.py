from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Version 0 results.csv.")
    parser.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "outputs" / "results.csv"))
    args = parser.parse_args()
    rows = []
    with Path(args.results).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        print("No rows found.")
        return
    levels: dict[str, int] = {}
    for row in rows:
        levels[row.get("severity_level", "unknown")] = levels.get(row.get("severity_level", "unknown"), 0) + 1
    print(f"Rows: {len(rows)}")
    print("Severity counts:")
    for key, value in sorted(levels.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

