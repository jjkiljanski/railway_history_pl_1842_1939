#!/usr/bin/env python3
"""
Generate yearly plots (1842–1939) of railway lines and stations.

Inputs:
- data/stations.csv (semicolon-delimited): columns station;coords;wiki_link
  - coords format: "lat,lon" (as produced by geocode_stations.py)
- data_preprocessed/lines.csv (semicolon-delimited): columns line_id;station_1;station_2;opened;closed;electrified;gauge_mm

Behavior:
- Matches station coordinates by exact station name. Drops stations without coords.
- For each year, draws:
  - Existing lines/stations (open in that year, built in prior years): black
  - Newly built (opened that year): green
  - Electrified that year (previous interval not electrified, new interval electrified): yellow
  - Gauge change that year (gauge differs from previous interval): purple
- Saves PNG files to the output directory.

Usage:
  python plot_timeline.py --stations data/stations.csv --lines data_preprocessed/lines.csv --outdir out_plots 
  # optional args: --start 1842 --end 1939 --dpi 220 --figw 12 --figh 10

Notes:
- Requires matplotlib. If missing, install via: pip install matplotlib
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Set, Any


def parse_coords(s: str) -> Optional[Tuple[float, float]]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        lat_str, lon_str = s.split(",", 1)
        lat = float(lat_str.strip())
        lon = float(lon_str.strip())
        return (lat, lon)
    except Exception:
        return None


def read_stations(path: str) -> Dict[str, Tuple[float, float]]:
    name_to_ll: Dict[str, Tuple[float, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            name = (row.get("station") or "").strip()
            coords = parse_coords(row.get("coords") or "")
            if name and coords is not None:
                name_to_ll[name] = coords
    return name_to_ll


def parse_date(s: str) -> Optional[dt.date]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except Exception:
        # Fallbacks are not expected because data_preprocessed/lines.csv should already be normalized
        try:
            # Try YYYY-MM
            parts = s.split("-")
            if len(parts) == 2:
                return dt.date(int(parts[0]), int(parts[1]), 1)
            if len(parts) == 1:
                return dt.date(int(parts[0]), 1, 1)
        except Exception:
            return None
        return None


@dataclass
class Interval:
    line_id: str
    a: str
    b: str
    opened: dt.date
    closed: Optional[dt.date]
    electrified: bool
    gauge: Optional[str]


def read_lines(path: str) -> Dict[Tuple[str, str], List[Interval]]:
    pairs: Dict[Tuple[str, str], List[Interval]] = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            line_id = (row.get("line_id") or "").strip()
            a = (row.get("station_1") or "").strip()
            b = (row.get("station_2") or "").strip()
            opened = parse_date(row.get("opened") or "")
            closed = parse_date(row.get("closed") or "")
            if not line_id or not a or not b or not opened:
                continue
            electrified = ((row.get("electrified") or "").strip().lower() == "true")
            gauge = (row.get("gauge_mm") or "").strip() or None
            key = tuple(sorted((a, b), key=str.casefold))  # undirected canonical
            pairs.setdefault(key, []).append(
                Interval(line_id=line_id, a=key[0], b=key[1], opened=opened, closed=closed, electrified=electrified, gauge=gauge)
            )

    # Sort intervals by opened date for each pair
    for k in pairs:
        pairs[k].sort(key=lambda it: it.opened)
    return pairs


def interval_active_in_year(it: Interval, year: int) -> bool:
    start_ok = it.opened <= dt.date(year, 12, 31)
    end_ok = (it.closed is None) or (it.closed >= dt.date(year, 1, 1))
    return start_ok and end_ok


def classify_events_by_year(pairs: Dict[Tuple[str, str], List[Interval]]) -> Dict[int, Dict[str, Set[Tuple[str, str]]]]:
    # Returns: year -> { 'new', 'electrified', 'gauge_change' } -> set of pairs
    out: Dict[int, Dict[str, Set[Tuple[str, str]]]] = {}
    for key, intervals in pairs.items():
        prev: Optional[Interval] = None
        for idx, cur in enumerate(intervals):
            y = cur.opened.year
            bucket = out.setdefault(y, {"new": set(), "electrified": set(), "gauge_change": set()})
            if prev is None:
                bucket["new"].add(key)
            else:
                changed_gauge = (cur.gauge != prev.gauge)
                became_electrified = (not prev.electrified and cur.electrified)
                if changed_gauge:
                    bucket["gauge_change"].add(key)
                if became_electrified:
                    bucket["electrified"].add(key)
            prev = cur
    return out


def ensure_dir(p: str) -> None:
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def load_background_geojson(path: str):
    import geopandas as gpd

    gdf = gpd.read_file(path)
    # If CRS missing, assume WGS84 (common for GeoJSON, but not guaranteed)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    # Reproject to lon/lat for plotting with your station coords
    if "4326" not in gdf.crs.to_string():
        gdf = gdf.to_crs("EPSG:4326")
    return gdf

def compute_global_bbox(
    pairs: Dict[Tuple[str, str], List[Interval]],
    stations_ll: Dict[str, Tuple[float, float]],
    start_year: int,
    end_year: int,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """
    Compute a fixed plot bbox from ALL line endpoints that ever exist
    (optionally limited to intervals overlapping [start_year, end_year]).

    Returns (minx, maxx, miny, maxy, pad_x, pad_y) in lon/lat space.
    """
    pts: List[Tuple[float, float]] = []

    start = dt.date(start_year, 1, 1)
    end = dt.date(end_year, 12, 31)

    def interval_overlaps_window(it: Interval) -> bool:
        # interval [opened, closed or +inf] overlaps [start, end]
        if it.opened > end:
            return False
        if it.closed is None:
            return True
        return it.closed >= start

    for (a, b), intervals in pairs.items():
        # include this pair if ANY interval overlaps the plotting window
        if not any(interval_overlaps_window(it) for it in intervals):
            continue
        if a in stations_ll and b in stations_ll:
            lat_a, lon_a = stations_ll[a]
            lat_b, lon_b = stations_ll[b]
            pts.append((lon_a, lat_a))
            pts.append((lon_b, lat_b))

    if not pts:
        return None

    minx = min(x for x, _ in pts)
    maxx = max(x for x, _ in pts)
    miny = min(y for _, y in pts)
    maxy = max(y for _, y in pts)

    pad_x = (maxx - minx) * 0.05 or 1.0
    pad_y = (maxy - miny) * 0.05 or 1.0
    return (minx, maxx, miny, maxy, pad_x, pad_y)


def plot_year(
    year: int,
    pairs: Dict[Tuple[str, str], List[Interval]],
    stations_ll: Dict[str, Tuple[float, float]],
    yearly_events: Dict[int, Dict[str, Set[Tuple[str, str]]]],
    station_first_year: Dict[str, int],
    outdir: str,
    dpi: int,
    figsize: Tuple[float, float],
    global_bbox: Optional[Tuple[float, float, float, float, float, float]],
    bg_gdf=None,
    bg_style: Optional[dict] = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib")

    # Gather segments active in this year
    active_pairs: List[Tuple[Tuple[str, str], Interval]] = []
    for key, intervals in pairs.items():
        # Choose the interval that is active this year (if multiple overlap, pick the one with latest opened)
        active_for_key: Optional[Interval] = None
        for it in intervals:
            if interval_active_in_year(it, year):
                if (active_for_key is None) or (it.opened > active_for_key.opened):
                    active_for_key = it
        if active_for_key is not None:
            # Both stations must have coords
            if key[0] in stations_ll and key[1] in stations_ll:
                active_pairs.append((key, active_for_key))

    ev = yearly_events.get(year, {"new": set(), "electrified": set(), "gauge_change": set()})
    new_set = ev.get("new", set())
    elec_set = ev.get("electrified", set())
    gauge_set = ev.get("gauge_change", set())

    # Split active pairs into categories for drawing
    baseline: List[Tuple[str, str]] = []
    new_lines: List[Tuple[str, str]] = []
    elec_lines: List[Tuple[str, str]] = []
    gauge_lines: List[Tuple[str, str]] = []
    for key, _it in active_pairs:
        if key in new_set:
            new_lines.append(key)
        elif key in gauge_set:
            gauge_lines.append(key)
        elif key in elec_set:
            elec_lines.append(key)
        else:
            baseline.append(key)

    # Station sets
    def stations_of(lines_list: Iterable[Tuple[str, str]]) -> Set[str]:
        st: Set[str] = set()
        for a, b in lines_list:
            st.add(a); st.add(b)
        return st

    st_new = stations_of(new_lines)
    st_gauge = stations_of(gauge_lines)
    st_elec = stations_of(elec_lines)
    st_base = stations_of(baseline) - st_new - st_gauge - st_elec

    # Fixed bounds (global), fallback to per-year if global not available
    if global_bbox is not None:
        minx, maxx, miny, maxy, pad_x, pad_y = global_bbox
    else:
        # Fallback: compute bounds from points present this year
        all_points: List[Tuple[float, float]] = []
        for name in st_new | st_gauge | st_elec | st_base:
            lat, lon = stations_ll[name]  # lat,lon order from file
            all_points.append((lon, lat))
        if not all_points:
            return
        minx = min(p[0] for p in all_points)
        maxx = max(p[0] for p in all_points)
        miny = min(p[1] for p in all_points)
        maxy = max(p[1] for p in all_points)
        pad_x = (maxx - minx) * 0.05 or 1.0
        pad_y = (maxy - miny) * 0.05 or 1.0

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1,1,1)

    # Background layer (district polygons etc.)
    if bg_gdf is not None and len(bg_gdf) > 0:
        try:
            # GeoPandas plots directly onto the Matplotlib axis
            style = bg_style or {}
            bg_gdf.plot(
                ax=ax,
                zorder=0,
                **style,
            )
        except Exception:
            # If background fails for any reason, continue without it
            pass

    ax.set_title(f"Railway network — {year}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect('equal')
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    # Colors
    C_BASE = "#000000"
    C_NEW = "#2ecc71"
    C_ELEC = "#f1c40f"
    C_GAUGE = "#9b59b6"

    def draw_lines(lines_list: List[Tuple[str, str]], color: str, lw: float, z: int):
        for a, b in lines_list:
            lata, lona = stations_ll[a]
            latb, lonb = stations_ll[b]
            ax.plot([lona, lonb], [lata, latb], color=color, lw=lw, alpha=0.9, zorder=z)

    def draw_points(stations: Set[str], color: str, size: float, z: int):
        if not stations:
            return
        xs = []
        ys = []
        for name in stations:
            lat, lon = stations_ll[name]
            xs.append(lon)
            ys.append(lat)
        ax.scatter(xs, ys, s=size, c=color, edgecolors='white', linewidths=0.2, alpha=0.95, zorder=z)

    # Draw baseline first, then overlays
    draw_lines(baseline, C_BASE, lw=0.6, z=1)
    draw_lines(gauge_lines, C_GAUGE, lw=1.6, z=4)
    draw_lines(elec_lines, C_ELEC, lw=1.6, z=5)
    draw_lines(new_lines, C_NEW, lw=1.8, z=6)

    draw_points(st_base, C_BASE, size=6, z=2)
    draw_points(st_gauge, C_GAUGE, size=12, z=7)
    draw_points(st_elec, C_ELEC, size=12, z=8)
    draw_points(st_new, C_NEW, size=14, z=9)

    # Add labels for stations that are added (first appear) this year
    # We consider a station "added" in the year of its first appearance across any interval
    label_stations: List[str] = [s for s in (st_new | st_gauge | st_elec | st_base) if station_first_year.get(s) == year]
    for name in label_stations:
        lat, lon = stations_ll[name]
        ax.annotate(
            name,
            xy=(lon, lat),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize=6,
            color=C_NEW,
            zorder=10,
            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.6),
        )

    # Simple legend proxies
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], color=C_BASE, lw=1.2, label='Existing'),
        Line2D([0],[0], color=C_NEW, lw=1.8, label='New'),
        Line2D([0],[0], color=C_ELEC, lw=1.8, label='Electrified'),
        Line2D([0],[0], color=C_GAUGE, lw=1.8, label='Gauge change'),
    ]
    ax.legend(handles=legend_elems, loc='lower right')
    ax.grid(True, color="#dddddd", lw=0.4, alpha=0.5)

    ensure_dir(outdir)
    out_path = os.path.join(outdir, f"{year}.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plot yearly railway network status from data_preprocessed/lines.csv and data/stations.csv")
    p.add_argument("--stations", default="data/stations.csv", help="Path to data/stations.csv")
    p.add_argument("--lines", default="data_preprocessed/lines.csv", help="Path to data_preprocessed/lines.csv")
    p.add_argument("--outdir", required=True, help="Directory to write PNGs")
    p.add_argument("--start", type=int, default=1842, help="Start year inclusive")
    p.add_argument("--end", type=int, default=1939, help="End year inclusive")
    p.add_argument("--dpi", type=int, default=220, help="Figure DPI")
    p.add_argument("--figw", type=float, default=12.0, help="Figure width inches")
    p.add_argument("--figh", type=float, default=10.0, help="Figure height inches")

    # Background GeoJSON
    p.add_argument("--bg_geojson", default=None, help="Optional background GeoJSON (polygons/lines) to plot under network")
    p.add_argument("--bg_edge", default="#666666", help="Background edge color")
    p.add_argument("--bg_face", default="#f0f0f0", help="Background face color (for polygons)")
    p.add_argument("--bg_alpha", type=float, default=0.35, help="Background transparency")
    p.add_argument("--bg_lw", type=float, default=0.4, help="Background line width")
    args = p.parse_args(argv)

    stations_ll = read_stations(args.stations)
    if not stations_ll:
        raise SystemExit("No station coordinates found. Ensure data/stations.csv has 'station' and 'coords'.")
    
    bg_gdf = None
    if args.bg_geojson:
        bg_gdf = load_background_geojson(args.bg_geojson)

    pairs = read_lines(args.lines)
    if not pairs:
        raise SystemExit("No line intervals found. Ensure data_preprocessed/lines.csv is generated and non-empty.")
    
    global_bbox = compute_global_bbox(
        pairs=pairs,
        stations_ll=stations_ll,
        start_year=args.start,
        end_year=args.end,
    )
    if global_bbox is None:
        raise SystemExit("Could not compute global bbox (no line endpoints with coordinates).")

    yearly_events = classify_events_by_year(pairs)

    # Compute first appearance year for each station (based on intervals' opened year)
    station_first_year: Dict[str, int] = {}
    for (a, b), intervals in pairs.items():
        for it in intervals:
            y = it.opened.year
            if a in stations_ll:
                station_first_year[a] = min(station_first_year.get(a, y), y)
            if b in stations_ll:
                station_first_year[b] = min(station_first_year.get(b, y), y)

    for year in range(args.start, args.end + 1):
        plot_year(
            year=year,
            pairs=pairs,
            stations_ll=stations_ll,
            yearly_events=yearly_events,
            station_first_year=station_first_year,
            outdir=args.outdir,
            dpi=args.dpi,
            figsize=(args.figw, args.figh),
            global_bbox=global_bbox,
            bg_gdf=bg_gdf,
            bg_style={
                "edgecolor": args.bg_edge,
                "facecolor": args.bg_face,
                "alpha": args.bg_alpha,
                "linewidth": args.bg_lw,
            },
        )

    print(f"Saved plots for years {args.start}–{args.end} to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
