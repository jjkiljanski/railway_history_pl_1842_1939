#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


def parse_coords(s: str) -> Optional[Tuple[float, float]]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        lat_s, lon_s = s.split(",", 1)
        return float(lat_s.strip()), float(lon_s.strip())
    except Exception:
        return None


def parse_date(s: str) -> Optional[dt.date]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except Exception:
        return None


def read_stations(path: str) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            name = (row.get("station") or "").strip()
            ll = parse_coords(row.get("coords") or "")
            if name and ll is not None:
                out[name] = ll
    return out


@dataclass
class Interval:
    a: str
    b: str
    opened: dt.date
    closed: Optional[dt.date]


def read_lines(path: str) -> Dict[Tuple[str, str], List[Interval]]:
    out: Dict[Tuple[str, str], List[Interval]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            a = (row.get("station_1") or "").strip()
            b = (row.get("station_2") or "").strip()
            opened = parse_date(row.get("opened") or "")
            closed = parse_date(row.get("closed") or "")
            if not a or not b or opened is None:
                continue
            key = tuple(sorted((a, b), key=str.casefold))
            out.setdefault(key, []).append(Interval(a=key[0], b=key[1], opened=opened, closed=closed))
    for key in out:
        out[key].sort(key=lambda it: it.opened)
    return out


def interval_active_in_year(it: Interval, year: int) -> bool:
    start_ok = it.opened <= dt.date(year, 12, 31)
    end_ok = (it.closed is None) or (it.closed >= dt.date(year, 1, 1))
    return start_ok and end_ok


def pair_active_in_year(intervals: List[Interval], year: int) -> bool:
    return any(interval_active_in_year(it, year) for it in intervals)


def first_open_year(intervals: List[Interval]) -> int:
    return min(it.opened.year for it in intervals)


def load_background_geojson(path: str):
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if "4326" not in gdf.crs.to_string():
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def stations_of(pairs: Set[Tuple[str, str]]) -> Set[str]:
    out: Set[str] = set()
    for a, b in pairs:
        out.add(a)
        out.add(b)
    return out


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot railways built between start and end year.")
    p.add_argument("--stations", default="data/stations.csv")
    p.add_argument("--lines", default="data_preprocessed/lines.csv")
    p.add_argument("--start_year", type=int, required=True)
    p.add_argument("--end_year", type=int, required=True)
    p.add_argument("--out_png", required=True)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--figw", type=float, default=12.0)
    p.add_argument("--figh", type=float, default=10.0)
    p.add_argument("--bg_geojson", default=None)
    p.add_argument("--bg_edge", default="#666666")
    p.add_argument("--bg_face", default="#f0f0f0")
    p.add_argument("--bg_alpha", type=float, default=0.35)
    p.add_argument("--bg_lw", type=float, default=0.4)
    args = p.parse_args()

    if args.start_year > args.end_year:
        raise SystemExit("--start_year must be <= --end_year")

    stations_ll = read_stations(args.stations)
    if not stations_ll:
        raise SystemExit(f"No station coordinates loaded from {args.stations}")

    pairs = read_lines(args.lines)
    if not pairs:
        raise SystemExit(f"No line intervals loaded from {args.lines}")

    active_start = {
        key for key, intervals in pairs.items()
        if pair_active_in_year(intervals, args.start_year)
    }
    active_end = {
        key for key, intervals in pairs.items()
        if pair_active_in_year(intervals, args.end_year)
    }
    built_between = {
        key for key in (active_end - active_start)
        if args.start_year < first_open_year(pairs[key]) <= args.end_year
    }

    # Keep only pairs with coordinates available for both stations.
    active_start = {k for k in active_start if k[0] in stations_ll and k[1] in stations_ll}
    built_between = {k for k in built_between if k[0] in stations_ll and k[1] in stations_ll}

    st_base = stations_of(active_start)
    st_new = stations_of(built_between) - st_base

    all_pts = []
    for name in st_base | st_new:
        lat, lon = stations_ll[name]
        all_pts.append((lon, lat))
    if not all_pts:
        raise SystemExit("No plottable stations for the selected years.")

    minx = min(x for x, _ in all_pts)
    maxx = max(x for x, _ in all_pts)
    miny = min(y for _, y in all_pts)
    maxy = max(y for _, y in all_pts)
    pad_x = (maxx - minx) * 0.05 or 1.0
    pad_y = (maxy - miny) * 0.05 or 1.0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(args.figw, args.figh), dpi=args.dpi)
    ax = fig.add_subplot(1, 1, 1)

    if args.bg_geojson:
        bg_gdf = load_background_geojson(args.bg_geojson)
        if len(bg_gdf) > 0:
            bg_gdf.plot(
                ax=ax,
                edgecolor=args.bg_edge,
                facecolor=args.bg_face,
                alpha=args.bg_alpha,
                linewidth=args.bg_lw,
                zorder=0,
            )

    ax.set_title(f"Railways built between {args.start_year} and {args.end_year}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    color_base = "#000000"
    color_new = "#2ecc71"

    def draw_lines(lines_list: Set[Tuple[str, str]], color: str, lw: float, z: int) -> None:
        for a, b in lines_list:
            lat_a, lon_a = stations_ll[a]
            lat_b, lon_b = stations_ll[b]
            ax.plot([lon_a, lon_b], [lat_a, lat_b], color=color, lw=lw, alpha=0.9, zorder=z)

    def draw_points(stations: Set[str], color: str, size: float, z: int) -> None:
        if not stations:
            return
        xs = []
        ys = []
        for name in stations:
            lat, lon = stations_ll[name]
            xs.append(lon)
            ys.append(lat)
        ax.scatter(xs, ys, s=size, c=color, edgecolors="white", linewidths=0.2, alpha=0.95, zorder=z)

    draw_lines(active_start, color_base, lw=0.6, z=1)
    draw_lines(built_between, color_new, lw=1.8, z=3)
    draw_points(st_base, color_base, size=6, z=2)
    draw_points(st_new, color_new, size=14, z=4)

    from matplotlib.lines import Line2D

    legend = [
        Line2D([0], [0], color=color_base, lw=1.2, label=f"Existing in {args.start_year}"),
        Line2D([0], [0], color=color_new, lw=1.8, label=f"Built by {args.end_year}"),
    ]
    ax.legend(handles=legend, loc="lower right")
    ax.grid(True, color="#dddddd", lw=0.4, alpha=0.5)

    ensure_parent_dir(args.out_png)
    fig.savefig(args.out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
