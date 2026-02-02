from __future__ import annotations

import argparse
import logging
import math
from typing import Dict, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Point
from shapely.ops import unary_union
from pyproj import CRS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def meters_per_minute(speed_kmh: float) -> float:
    return speed_kmh * 1000.0 / 60.0


def load_network_parts(network_path: str, work_crs: CRS) -> Tuple[nx.Graph, Dict[str, Point], gpd.GeoDataFrame]:
    """
    Load routable network GeoJSON and return:
      - Graph with edge weight = time_min (minutes)
      - station points in work CRS
      - rail edges GeoDataFrame in work CRS (builder == 'rail')
    """
    gdf = gpd.read_file(network_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf_w = gdf.to_crs(work_crs)

    nodes = gdf_w[gdf_w["feature"] == "node"].copy()
    edges = gdf_w[gdf_w["feature"] == "edge"].copy()

    station_points: Dict[str, Point] = {str(r["station"]): r.geometry for _, r in nodes.iterrows()}

    G = nx.Graph()
    for st in station_points.keys():
        G.add_node(st)

    missing = 0
    for _, r in edges.iterrows():
        u = str(r["u"])
        v = str(r["v"])
        t = r.get("time_min", None)
        if t is None or (isinstance(t, float) and math.isnan(t)):
            missing += 1
            continue
        try:
            w = float(t)
        except Exception:
            missing += 1
            continue
        G.add_edge(u, v, weight=w)

    if missing:
        log.info(f"{network_path}: skipped {missing:,} edges missing time_min")

    rail_edges = edges[edges.get("builder", "") == "rail"].copy()
    log.info(f"{network_path}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges; rail edges: {len(rail_edges):,}")
    return G, station_points, rail_edges


def dijkstra_from_origin(G: nx.Graph, station_points: Dict[str, Point], origin: Point, horse_kmh: float) -> Dict[str, float]:
    v_mpm = meters_per_minute(horse_kmh)
    H = G.copy()
    H.add_node("ORIGIN")

    for st, pt in station_points.items():
        t_min = origin.distance(pt) / v_mpm
        H.add_edge("ORIGIN", st, weight=float(t_min))

    dist = nx.single_source_dijkstra_path_length(H, "ORIGIN", weight="weight")
    dist.pop("ORIGIN", None)
    return dist


def isochrone_poly(origin: Point, station_points: Dict[str, Point], t_station: Dict[str, float], T: float, horse_kmh: float):
    v_mpm = meters_per_minute(horse_kmh)
    geoms = [origin.buffer(v_mpm * T)]
    for st, t in t_station.items():
        if t <= T:
            r = v_mpm * (T - t)
            if r > 0:
                geoms.append(station_points[st].buffer(r))
    return unary_union(geoms)


def build_rings(
    G: nx.Graph,
    station_points: Dict[str, Point],
    origin: Point,
    boundary_geom,
    work_crs: CRS,
    horse_kmh: float,
    max_minutes: int,
    step_minutes: int,
) -> gpd.GeoDataFrame:
    """
    Build *ring* polygons for arrival time buckets up to max_minutes.
    Each ring has arrive_min = upper bound of bucket (e.g., 30, 60, 90, ...)
    """
    log.info("Computing shortest times to stations...")
    t_station = dijkstra_from_origin(G, station_points, origin, horse_kmh)
    log.info(f"Times computed for {len(t_station):,} stations")

    thresholds = list(range(step_minutes, max_minutes + step_minutes, step_minutes))
    cumulative = []
    for T in thresholds:
        poly = isochrone_poly(origin, station_points, t_station, float(T), horse_kmh).intersection(boundary_geom)
        cumulative.append((T, poly))

    rings = []
    prev = None
    for T, poly in cumulative:
        ring = poly if prev is None else poly.difference(prev)
        rings.append({"arrive_min": T, "geometry": ring})
        prev = poly

    rings_gdf = gpd.GeoDataFrame(rings, crs=work_crs)  # <-- FIX: set CRS explicitly
    rings_gdf = rings_gdf[~rings_gdf.is_empty].copy()
    return rings_gdf


def main() -> int:
    ap = argparse.ArgumentParser(description="Filled time-savings map between two networks (old vs new).")
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--boundary", required=True)

    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)

    ap.add_argument("--horse_kmh", type=float, default=4.0)
    ap.add_argument("--hours", type=int, default=10)
    ap.add_argument("--step_minutes", type=int, default=30)

    ap.add_argument("--work_crs", default="EPSG:3035")
    ap.add_argument("--out_png", default="time_savings_filled.png")
    ap.add_argument("--out_geojson", default="time_savings_filled.geojson")

    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--rail_width", type=float, default=0.45)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    work_crs = CRS.from_user_input(args.work_crs)

    # Boundary
    boundary = gpd.read_file(args.boundary)
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")
    boundary_w = boundary.to_crs(work_crs)
    boundary_geom = boundary_w.geometry.iloc[0]

    # Origin
    origin = gpd.GeoSeries([Point(args.lon, args.lat)], crs="EPSG:4326").to_crs(work_crs).iloc[0]

    # Networks
    log.info("Loading OLD network...")
    G_old, pts_old, rail_old = load_network_parts(args.old, work_crs)
    log.info("Loading NEW network...")
    G_new, pts_new, rail_new = load_network_parts(args.new, work_crs)

    max_minutes = args.hours * 60

    # Rings
    log.info("Building OLD rings...")
    old_rings = build_rings(G_old, pts_old, origin, boundary_geom, work_crs, args.horse_kmh, max_minutes, args.step_minutes)
    old_rings = old_rings.rename(columns={"arrive_min": "old_min"})

    log.info("Building NEW rings...")
    new_rings = build_rings(G_new, pts_new, origin, boundary_geom, work_crs, args.horse_kmh, max_minutes, args.step_minutes)
    new_rings = new_rings.rename(columns={"arrive_min": "new_min"})

    # Overlay → savings polygons
    log.info("Overlaying old/new rings (can take time)...")
    inter = gpd.overlay(old_rings, new_rings, how="intersection", keep_geom_type=False)
    if inter.crs is None:
        inter = inter.set_crs(work_crs)  # <-- FIX: ensure CRS survives overlay

    inter = inter[~inter.is_empty].copy()
    inter["save_min"] = inter["old_min"] - inter["new_min"]
    inter = inter[inter["save_min"] > 0].copy()
    inter["save_hours"] = inter["save_min"]/60

    log.info(f"Areas with positive savings: {len(inter):,} polygons")

    # Save GeoJSON in EPSG:4326
    inter.to_crs("EPSG:4326").to_file(args.out_geojson, driver="GeoJSON")
    log.info(f"Wrote {args.out_geojson}")

    # Plot: filled gradient + rail overlay
    fig, ax = plt.subplots(figsize=(10, 10))

    gpd.GeoSeries([boundary_geom], crs=work_crs).boundary.plot(ax=ax, linewidth=1, zorder=2)

    if len(inter) > 0:
        inter.plot(
            ax=ax,
            column="save_hours",
            cmap=args.cmap,
            alpha=args.alpha,
            linewidth=0.0,
            legend=True,
            legend_kwds={"label": "Time saved (hours)", "shrink": 0.6},
            zorder=1,
        )

    rail_overlay = rail_new if len(rail_new) > 0 else rail_old
    if len(rail_overlay) > 0:
        rail_overlay.plot(
            ax=ax,
            color="black",
            linewidth=args.rail_width,
            alpha=0.9,
            zorder=3,
        )

    gpd.GeoSeries([origin], crs=work_crs).plot(ax=ax, markersize=30, zorder=4)

    ax.set_title(args.title)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=250)
    log.info(f"Wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
