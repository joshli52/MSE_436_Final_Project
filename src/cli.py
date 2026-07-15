"""
cli.py – Command-line entry point for the conference cost optimizer.

Sources may be supplied as:
  --sources '[{"city":"Chicago, IL","attendees":30},...]'   (JSON string)
  --sources sources.csv                                     (CSV: city,attendees)

Examples
--------
    python cli.py --sources '[{"city":"Chicago, IL","attendees":30},{"city":"Dallas/Fort Worth, TX","attendees":50}]'
    python cli.py --sources attendees.csv --candidates '["Chicago, IL","New York City, NY (Metropolitan Area)"]'
    python cli.py --sources attendees.csv --top 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from optimize import load_artifacts, match_city, run_optimizer, format_ranking


def _parse_sources(raw: str) -> list[dict]:
    """
    Parse sources from a JSON string or a CSV file path.
    Returns list of {"city": str, "attendees": int}.
    """
    raw = raw.strip()

    # Try JSON first
    if raw.startswith("[") or raw.startswith("{"):
        data = json.loads(raw)
        if isinstance(data, dict):
            # Support {"CityA": 30, "CityB": 50} shorthand
            return [{"city": k, "attendees": v} for k, v in data.items()]
        return data   # already a list of dicts

    # Treat as file path
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"Sources file not found: {path}")

    df = pd.read_csv(path)
    required = {"city", "attendees"}
    missing  = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"CSV must have columns 'city' and 'attendees'; missing: {missing}")
    df.columns = df.columns.str.lower()
    return df[["city", "attendees"]].to_dict("records")


def _parse_candidates(raw: str | None, all_cities: list[str]) -> list[str] | None:
    """Parse --candidates JSON list or return None (= all 115 cities)."""
    if raw is None:
        return None
    data = json.loads(raw.strip())
    resolved = []
    for name in data:
        resolved.append(match_city(name, all_cities))
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank conference host cities by total attendee travel cost."
    )
    parser.add_argument(
        "--sources", required=True,
        help="JSON list of {city, attendees} or path to a CSV file with those columns.",
    )
    parser.add_argument(
        "--candidates",
        help="JSON list of candidate host cities to evaluate (default: all 115).",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of hosts to show per-source detail for (default: 5).",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="Path to models/ directory (default: <project-root>/models).",
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).parent.parent,
        help="Project root directory (default: parent of src/).",
    )
    args = parser.parse_args()

    # Load artifacts
    arts = load_artifacts(model_dir=args.model_dir, root=args.root)

    # Parse inputs
    try:
        sources = _parse_sources(args.sources)
    except (json.JSONDecodeError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR parsing --sources: {exc}", file=sys.stderr)
        sys.exit(1)

    if not sources:
        print("ERROR: no sources provided.", file=sys.stderr)
        sys.exit(1)

    # Resolve candidate cities
    try:
        candidates = _parse_candidates(args.candidates, arts["all_cities"])
    except ValueError as exc:
        print(f"ERROR resolving --candidates: {exc}", file=sys.stderr)
        sys.exit(1)

    # Resolve and validate ALL sources before printing anything
    resolved_display: list[tuple[str, str, int]] = []
    errors = []
    for s in sources:
        try:
            city = match_city(s["city"], arts["all_cities"])
            resolved_display.append((s["city"], city, int(s["attendees"])))
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nSource cities:")
    for raw, resolved, att in resolved_display:
        flag = f"  → {resolved}" if resolved != raw else ""
        print(f"  {raw}{flag}  ({att} attendees)")

    total_att = sum(int(s["attendees"]) for s in sources)
    print(f"  Total attendees: {total_att}")

    n_cand = len(candidates) if candidates else len(arts["all_cities"])
    print(f"  Evaluating {n_cand} candidate host cities …")

    # Run optimizer
    result = run_optimizer(sources, arts, candidates=candidates)

    # Print ranked table
    print(format_ranking(result, top_n_detail=args.top))


if __name__ == "__main__":
    main()
