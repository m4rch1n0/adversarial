"""I/O utilities for experiment outputs."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def run_dir(domain: str, base: str = "./results") -> Path:
    """Create and return a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base) / domain / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(path: Path, rows: List[List[str]], header: List[str]) -> None:
    """Save CSV file with header and rows."""
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Save JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Save JSONL file (one JSON object per line)."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def save_metadata(path: Path, **kwargs) -> None:
    """Save metadata as JSON with timestamp."""
    data = {
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    save_json(path, data)


def print_csv_stdout(header: List[str], rows: List[List[Any]]):
    sep = "=" * 50
    print("\n" + sep)
    str_rows = [[str(x) for x in row] for row in rows]
    widths = [len(str(h)) for h in header]
    for r in str_rows:
        for i, cell in enumerate(r):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
            else:
                widths.append(len(cell))
    def fmt(cols: List[str]) -> str:
        return "  ".join((cols[i] if i < len(cols) else "").ljust(widths[i]) for i in range(len(widths)))
    
    print(fmt([str(h) for h in header]))
    for r in str_rows:
        print(fmt(r))
    print(sep)
