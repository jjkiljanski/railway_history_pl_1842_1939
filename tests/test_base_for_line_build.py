# tests/test_base_for_line_build.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Set


def _parse_station_list(raw: str) -> List[str]:
    """Same logic as in your builder: handles '[A, B]' or 'A, B'."""
    s = (raw or "").strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    out: List[str] = []
    for part in s.split(","):
        name = part.strip()
        if name:
            out.append(name)
    return out


def _read_semicolon_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


def _find_stations_db_file(project_root: Path) -> Path:
    """
    Tries common filenames/locations. Adjust or replace with your real path if you have one.
    """
    candidates = [
        project_root / "stations.csv",
        project_root / "stations_database.csv",
        project_root / "stations_db.csv",
        project_root / "stations_geo.csv",
        project_root / "data" / "stations.csv",
        project_root / "data" / "stations_database.csv",
        project_root / "data" / "stations_db.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise AssertionError(
        "Stations database CSV not found. Looked for:\n"
        + "\n".join(str(p) for p in candidates)
        + "\nUpdate _find_stations_db_file() to your actual filename/path."
    )


def _extract_station_name_field(header_fields: Iterable[str]) -> str:
    """
    Detects which column holds the station name in the stations DB.
    Common options: name, station, station_name.
    """
    fields = [h.strip() for h in header_fields if h]
    lower = {h.casefold(): h for h in fields}

    for key in ("name", "station_name", "station"):
        if key in lower:
            return lower[key]

    raise AssertionError(
        "Could not detect station name column in stations DB. "
        f"Found columns: {fields}. Expected one of: name, station_name, station."
    )


def test_base_for_line_build() -> None:
    """
    Validates:
    1) railway_history.csv has required structure (semicolon-delimited + required columns)
    2) every station mentioned in railway_history.csv exists in stations DB
       (reports ALL missing stations, not just the first)
    """
    project_root = Path(__file__).resolve().parents[1]

    history_path = project_root / "railway_history_updated_new_date_edited.csv"
    assert history_path.exists(), f"Missing input file: {history_path}"

    # --- 1) Structure checks for railway_history.csv ---
    with history_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        assert reader.fieldnames is not None, "railway_history.csv has no header row"

        fieldnames = [h.strip() for h in reader.fieldnames if h]
        fieldset = {h.casefold() for h in fieldnames}

        # Update this list if your schema differs, but these are implied by your builder.
        required_history_cols = {"date", "event_type", "stations", "gauge_mm", "source", "comment"}
        missing_cols = sorted(required_history_cols - fieldset)
        assert not missing_cols, (
            "railway_history.csv missing required columns: "
            + ", ".join(missing_cols)
            + f"\nFound columns: {fieldnames}"
        )

    history_rows = _read_semicolon_csv(history_path)

    # --- 2) Load stations database + validate station existence ---
    stations_db_path = _find_stations_db_file(project_root)
    with stations_db_path.open("r", encoding="utf-8", newline="") as f:
        stations_reader = csv.DictReader(f, delimiter=";")
        assert stations_reader.fieldnames is not None, f"{stations_db_path} has no header row"

        name_col = _extract_station_name_field(stations_reader.fieldnames)

        stations_in_db: Set[str] = set()
        for r in stations_reader:
            v = (r.get(name_col) or "").strip()
            if v:
                stations_in_db.add(v)

    assert stations_in_db, f"No stations loaded from {stations_db_path} (check delimiter/header/data)."

    # Collect ALL stations referenced in history
    mentioned: Set[str] = set()
    for r in history_rows:
        for st in _parse_station_list(r.get("stations") or ""):
            mentioned.add(st)

    # Compute missing (report all)
    missing = sorted(s for s in mentioned if s not in stations_in_db)

    assert not missing, (
        "Stations mentioned in railway_history.csv but not found in stations DB:\n"
        + "\n".join(missing)
        + f"\n\nStations DB used: {stations_db_path}"
    )
