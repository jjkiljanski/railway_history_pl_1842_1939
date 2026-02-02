from __future__ import annotations

import argparse
import math
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
from pyproj import Geod


# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------

GEOD = Geod(ellps="WGS84")


def geodesic_m(p1: Point, p2: Point) -> float:
    _, _, dist_m = GEOD.inv(p1.x, p1.y, p2.x, p2.y)
    return float(dist_m)


def parse_coords_to_point(coords: str) -> Optional[Point]:
    if not coords:
        return None
    parts = [p.strip() for p in str(coords).split(",")]
    if len(parts) < 2:
        return None
    try:
        lat = float(parts[0])
        lon = float(parts[1])
    except ValueError:
        return None
    return Point(lon, lat)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

@dataclass
class Station:
    name: str
    point: Point


def build_station_index(path: str) -> Dict[str, Station]:
    log.info("Loading stations...")
    df = pd.read_csv(path, sep=";", dtype=str).fillna("")
    stations = {}

    for _, r in df.iterrows():
        name = r["station"].strip()
        pt = parse_coords_to_point(r["coords"])
        if name and pt is not None:
            stations[name] = Station(name, pt)

    log.info(f"Loaded {len(stations):,} stations with coordinates")
    return stations


def is_active(opened, closed, year: int) -> bool:
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    if pd.isna(opened) or opened > end:
        return False
    if pd.isna(closed):
        return True
    return closed >= start


def load_active_lines(path: str, year: int) -> pd.DataFrame:
    log.info(f"Filtering active lines for year {year}...")
    df = pd.read_csv(path, sep=";", dtype=str).fillna("")
    df["opened_dt"] = pd.to_datetime(df["opened"], errors="coerce")
    df["closed_dt"] = pd.to_datetime(df["closed"].replace("", pd.NA), errors="coerce")
    mask = df.apply(lambda r: is_active(r["opened_dt"], r["closed_dt"], year), axis=1)
    active = df.loc[mask].copy()
    log.info(f"Active lines: {len(active):,}")
    return active


# ------------------------------------------------------------------
# Network construction
# ------------------------------------------------------------------

def gauge_speed_kmh(gauge: str, normal: float, narrow: float) -> float:
    if not gauge:
        return normal
    try:
        return narrow if float(gauge) < 1000 else normal
    except ValueError:
        return normal


def add_rail_edges(G, lines, stations, network_station_names, normal_kmh, narrow_kmh):
    log.info("Building rail edges...")
    added = 0
    for _, r in lines.iterrows():
        a, b = r["station_1"].strip(), r["station_2"].strip()
        if a not in network_station_names or b not in network_station_names:
            continue
        pa, pb = stations[a].point, stations[b].point
        d = geodesic_m(pa, pb)
        speed = gauge_speed_kmh(r["gauge_mm"], normal_kmh, narrow_kmh)
        t_min = d / (speed * 1000 / 3600) / 60

        G.add_edge(
            a, b,
            mode="rail",
            speed_kmh=speed,
            distance_m=d,
            time_min=t_min,
            gauge_mm=r["gauge_mm"] or "1435",
            builder="rail",
        )
        added += 1

    log.info(f"Rail edges added: {added:,}")


def add_transfer_edges(G, stations, network_station_names, radius_m, horse_kmh):
    log.info(f"Adding proximity transfer edges (radius={radius_m:.0f} m)...")
    names = list(network_station_names)
    n = len(names)
    added = 0
    speed_mps = horse_kmh * 1000 / 3600
    t0 = time.time()

    for i, a in enumerate(names):
        pa = stations[a].point
        if i % max(1, n // 10) == 0:
            log.info(f"Transfer scan: {100 * i / n:.0f}% ({i:,}/{n:,})")

        for b in names[i + 1:]:
            if G.has_edge(a, b):
                continue
            pb = stations[b].point
            d = geodesic_m(pa, pb)
            if d <= radius_m:
                t_min = d / speed_mps / 60
                G.add_edge(
                    a, b,
                    mode="horse",
                    speed_kmh=horse_kmh,
                    distance_m=d,
                    time_min=t_min,
                    builder="transfer",
                )
                added += 1

    log.info(f"Transfer edges added: {added:,} ({time.time() - t0:.1f}s)")


def force_connectivity(G, stations, horse_kmh):
    log.info("Forcing graph connectivity...")
    speed_mps = horse_kmh * 1000 / 3600
    pts = {k: stations[k].point for k in G.nodes}
    added = 0

    while True:
        comps = list(nx.connected_components(G))
        if len(comps) <= 1:
            break

        log.info(f"Components remaining: {len(comps)}")
        c0 = comps[0]
        best = (math.inf, None, None)

        for other in comps[1:]:
            for a in c0:
                for b in other:
                    d = geodesic_m(pts[a], pts[b])
                    if d < best[0]:
                        best = (d, a, b)

        d, a, b = best
        t_min = d / speed_mps / 60
        G.add_edge(
            a, b,
            mode="horse",
            speed_kmh=horse_kmh,
            distance_m=d,
            time_min=t_min,
            builder="component_bridge",
        )
        added += 1
        log.info(f"Added bridge {added}: {a} ↔ {b} ({d/1000:.1f} km)")

    log.info("Graph is now fully connected")


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------

def export_geojson(G, stations, path):
    log.info("Writing GeoJSON...")
    feats = []

    for n in G.nodes:
        p = stations[n].point
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": (p.x, p.y)},
            "properties": {"feature": "node", "station": n},
        })

    for u, v, d in G.edges(data=True):
        pu, pv = stations[u].point, stations[v].point
        feats.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [(pu.x, pu.y), (pv.x, pv.y)],
            },
            "properties": {
                "feature": "edge",
                "u": u,
                "v": v,
                **d,
            },
        })

    gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326").to_file(path, driver="GeoJSON")
    log.info(f"Wrote {path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--lines", default="data_preprocessed/lines_cropped.csv")
    ap.add_argument("--stations", default="data/stations.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--speed_normal_kmh", type=float, default=15)
    ap.add_argument("--speed_narrow_kmh", type=float, default=15)
    ap.add_argument("--speed_horse_kmh", type=float, default=4)
    ap.add_argument("--transfer_radius_m", type=float, default=1200)
    args = ap.parse_args()

    out = args.out or f"output/network_{args.year}.geojson"

    stations = build_station_index(args.stations)
    lines = load_active_lines(args.lines, args.year)

    network_station_names = set(lines["station_1"].str.strip()) | set(lines["station_2"].str.strip())
    network_station_names = {s for s in network_station_names if s in stations}
    log.info(f"Stations referenced by active lines: {len(network_station_names):,}")

    G = nx.Graph()
    add_rail_edges(G, lines, stations, network_station_names, args.speed_normal_kmh, args.speed_narrow_kmh)
    add_transfer_edges(G, stations, network_station_names, args.transfer_radius_m, args.speed_horse_kmh)
    force_connectivity(G, stations, args.speed_horse_kmh)

    isolates = list(nx.isolates(G))
    if isolates:
        log.info(f"Removing isolates (degree=0): {len(isolates):,}")
        G.remove_nodes_from(isolates)

    export_geojson(G, stations, out)


if __name__ == "__main__":
    main()
