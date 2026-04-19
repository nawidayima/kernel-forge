#!/usr/bin/env python3
"""Extract `RESULT key=value ...` lines from a log into a CSV.

CSV columns are the union of keys across all RESULT lines (order = first-seen).
Run: python3 parse_results.py <log> <csv>
"""

import re
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: parse_results.py <log> <csv>", file=sys.stderr)
        sys.exit(2)
    log_path = Path(sys.argv[1])
    csv_path = Path(sys.argv[2])

    key_order: list[str] = []
    rows: list[dict[str, str]] = []
    pattern = re.compile(r"(\w+)=(\S+)")
    for raw in log_path.read_text().splitlines():
        if not raw.startswith("RESULT "):
            continue
        row = dict(pattern.findall(raw))
        for k in row:
            if k not in key_order:
                key_order.append(k)
        rows.append(row)

    with csv_path.open("w") as f:
        f.write(",".join(key_order) + "\n")
        for row in rows:
            f.write(",".join(row.get(k, "") for k in key_order) + "\n")


if __name__ == "__main__":
    main()
