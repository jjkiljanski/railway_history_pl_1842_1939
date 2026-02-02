#!/usr/bin/env python3
"""
Build a CSV of railway line segments (undirected) from events in railway_history.csv.

Output columns (semicolon-delimited):
  line_id;station_1;station_2;opened;closed;electrified;gauge_mm;source;comment

Rules:
- Each consecutive pair in an event's stations list forms a line segment.
- The pair (A,B) is the same as (B,A); we store it canonically sorted by name.
- 'opened' and 'closed' are ISO dates YYYY-MM-DD; if source has only year or year-month,
  missing parts are normalized to '01'. Unknown components like '??' also normalize to '01'.
- On electrification or gauge change, close the current row at event date and start a new
  row with the same line_id and updated state.
- Event types handled: opened (start/open segment), closed (close segment),
  electrification (toggle electrified to true). 'reconstruction' is ignored.

Assumptions:
- Gauge is taken from the event's gauge_mm field when present; gauge changes are applied
  only when the segment is open (otherwise ignored until an open occurs with that gauge).
- Electrification is one-way (true once electrified); de-electrification is not modeled.
- 'source' and 'comment' columns from input events are copied onto produced line rows.
  When a segment remains open at end-of-file, its last known source/comment are emitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


DATE_RE = re.compile(r"^(\d{4})(?:-(\d{2}|\?\?))?(?:-(\d{2}|\?\?))?$")


def normalize_date_iso(s: str) -> str:
    m = DATE_RE.match(s.strip())
    if not m:
        # Fallback: try to keep only the year, fill others with 01
        year = re.findall(r"\d{4}", s)
        y = year[0] if year else "1900"
        return f"{y}-01-01"
    y, mo, d = m.groups()
    mo = mo if mo and mo != "??" else "01"
    d = d if d and d != "??" else "01"
    return f"{y}-{mo}-{d}"


def date_sort_key(s: str) -> Tuple[int, int, int]:
    m = DATE_RE.match(s.strip())
    if not m:
        # Try to extract year best-effort
        year = re.findall(r"\d{4}", s)
        y = int(year[0]) if year else 1900
        return (y, 1, 1)
    y, mo, d = m.groups()
    ym = 1 if mo in (None, "??") else int(mo)
    dd = 1 if d in (None, "??") else int(d)
    return (int(y), ym, dd)


def parse_station_list(raw: str) -> List[str]:
    s = (raw or "").strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    # Split by comma and strip whitespace
    out: List[str] = []
    for part in s.split(","):
        name = part.strip()
        if name:
            out.append(name)
    return out


def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    # Sort using casefold for stability with accents retained as-is
    return tuple(sorted((a, b), key=lambda x: x.casefold()))  # type: ignore[return-value]


def make_line_id(a: str, b: str) -> str:
    a_c, b_c = canonical_pair(a, b)
    key = f"{a_c}|||{b_c}".encode("utf-8")
    h = hashlib.sha1(key).hexdigest()[:10]
    return f"L_{h}"


@dataclass
class State:
    opened: str
    closed: Optional[str]
    electrified: bool
    gauge_mm: Optional[str]
    source: str
    comment: str


def build_lines(events: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    # Prepare events sorted by date
    sorted_events = sorted(
        events,
        key=lambda r: date_sort_key(r.get("date", "1900-01-01")),
    )

    # Active state per undirected pair
    active: Dict[Tuple[str, str], State] = {}
    # Constant line_id per pair
    line_ids: Dict[Tuple[str, str], str] = {}
    # Final rows
    rows: List[Dict[str, str]] = []

    for row in sorted_events:
        date_raw = row.get("date", "1900-01-01")
        date_iso = normalize_date_iso(date_raw)
        event_type = (row.get("event_type") or "").strip().lower()
        gauge_val = (row.get("gauge_mm") or "").strip() or None

        source_val = (row.get("source") or "").strip()
        comment_val = (row.get("comment") or "").strip()

        stations_raw = row.get("stations") or ""
        stations = parse_station_list(stations_raw)
        if len(stations) < 2:
            continue

        # Build consecutive pairs
        pairs: List[Tuple[str, str]] = []
        for i in range(len(stations) - 1):
            a, b = stations[i], stations[i + 1]
            a_c, b_c = canonical_pair(a, b)
            pairs.append((a_c, b_c))

        for a_c, b_c in pairs:
            key = (a_c, b_c)
            lid = line_ids.get(key)
            if lid is None:
                lid = make_line_id(a_c, b_c)
                line_ids[key] = lid

            st = active.get(key)

            if event_type == "opened":
                # If already open, ignore; else open new state
                if st is None:
                    active[key] = State(
                        opened=date_iso,
                        closed=None,
                        electrified=False,
                        gauge_mm=gauge_val,
                        source=source_val,
                        comment=comment_val,
                    )
                else:
                    # Already open; if gauge specified and changes, split state
                    if gauge_val and gauge_val != st.gauge_mm:
                        # Close current
                        rows.append(
                            {
                                "line_id": lid,
                                "station_1": a_c,
                                "station_2": b_c,
                                "opened": st.opened,
                                "closed": date_iso,
                                "electrified": "true" if st.electrified else "false",
                                "gauge_mm": st.gauge_mm or "",
                                "source": source_val,
                                "comment": comment_val,
                            }
                        )
                        # Start new
                        active[key] = State(
                            opened=date_iso,
                            closed=None,
                            electrified=st.electrified,
                            gauge_mm=gauge_val,
                            source=source_val,
                            comment=comment_val,
                        )

            elif event_type == "closed":
                if st is not None:
                    # Close current state
                    rows.append(
                        {
                            "line_id": lid,
                            "station_1": a_c,
                            "station_2": b_c,
                            "opened": st.opened,
                            "closed": date_iso,
                            "electrified": "true" if st.electrified else "false",
                            "gauge_mm": st.gauge_mm or "",
                            "source": source_val,
                            "comment": comment_val,
                        }
                    )
                    active.pop(key, None)

            elif event_type == "electrification":
                if st is not None:
                    if not st.electrified:
                        # Close current, start new electrified segment
                        rows.append(
                            {
                                "line_id": lid,
                                "station_1": a_c,
                                "station_2": b_c,
                                "opened": st.opened,
                                "closed": date_iso,
                                "electrified": "false",
                                "gauge_mm": st.gauge_mm or "",
                                "source": source_val,
                                "comment": comment_val,
                            }
                        )
                        active[key] = State(
                            opened=date_iso,
                            closed=None,
                            electrified=True,
                            gauge_mm=st.gauge_mm,
                            source=source_val,
                            comment=comment_val,
                        )
                # If not open, ignore electrification for this pair

            else:
                # Other events: apply gauge change if segment is open
                if st is not None and gauge_val and gauge_val != st.gauge_mm:
                    rows.append(
                        {
                            "line_id": lid,
                            "station_1": a_c,
                            "station_2": b_c,
                            "opened": st.opened,
                            "closed": date_iso,
                            "electrified": "true" if st.electrified else "false",
                            "gauge_mm": st.gauge_mm or "",
                            "source": source_val,
                            "comment": comment_val,
                        }
                    )
                    active[key] = State(
                        opened=date_iso,
                        closed=None,
                        electrified=st.electrified,
                        gauge_mm=gauge_val,
                        source=source_val,
                        comment=comment_val,
                    )

    # Flush any segments still open (no closed date)
    for (a_c, b_c), st in active.items():
        lid = line_ids[(a_c, b_c)]
        rows.append(
            {
                "line_id": lid,
                "station_1": a_c,
                "station_2": b_c,
                "opened": st.opened,
                "closed": "",
                "electrified": "true" if st.electrified else "false",
                "gauge_mm": st.gauge_mm or "",
                "source": st.source,
                "comment": st.comment,
            }
        )

    return rows


def read_events(path: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            out.append(row)
    return out


def write_lines(rows: List[Dict[str, str]], path: str) -> None:
    fieldnames = [
        "line_id",
        "station_1",
        "station_2",
        "opened",
        "closed",
        "electrified",
        "gauge_mm",
        "source",
        "comment",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build railway line segments CSV from history events"
    )
    parser.add_argument(
        "--input", default="data/railway_history.csv", help="Path to input history CSV"
    )
    parser.add_argument("--output", default="data_preprocessed/lines.csv", help="Path to output lines CSV")
    args = parser.parse_args()

    events = read_events(args.input)
    rows = build_lines(events)
    write_lines(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
