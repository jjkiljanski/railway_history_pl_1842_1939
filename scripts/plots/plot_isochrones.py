from __future__ import annotations

import argparse
import logging
import math
from typing import Dict, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
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


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def meters_per_minute(speed_kmh: float) -> float:
    return speed_kmh * 1000.0 / 60.0


# ------------------------------------------------------------
# Network loading
# ------------------------------------------------------------

def build_graph_and_points(
    network_path: str,
    work_crs: CRS
) -> Tuple[nx.Graph, Dict[str, Point], gpd.GeoDataFrame]:
    """
    Returns:
      - Graph with edge weight = time_min (minutes)
      - station_points in work_crs
      - rail_edges GeoDataFrame in work_crs (builder == 'rail')
    """
    gdf = gpd.read_file(network_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf_w = gdf.to_crs(work_crs)

    nodes = gdf_w[gdf_w["feature"] == "node"].copy()
    edges = gdf_w[gdf_w["feature"] == "edge"].copy()

    station_points: Dict[str, Point] = {
        str(r["station"]): r.geometry for _, r in nodes.iterrows()
    }

    G = nx.Graph()
    for st in station_points:
        G.add_node(st)

    missing = 0
    for _, r in edges.iterrows():
        u = str(r["u"])
        v = str(r["v"])
        t = r.get("time_min")
        if t is None or (isinstance(t, float) and math.isnan(t)):
            missing += 1
            continue
        G.add_edge(u, v, weight=float(t), builder=str(r.get("builder", "")))

    if missing:
        log.info(f"Skipped {missing:,} edges missing time_min")

    rail_edges = edges[edges.get("builder", "") == "rail"].copy()

    log.info(
        f"Loaded graph: {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges"
    )
    log.info(f"Rail edges for overlay: {len(rail_edges):,}")
    return G, station_points, rail_edges


# ------------------------------------------------------------
# Shortest paths
# ------------------------------------------------------------

def dijkstra_from_origin(
    G: nx.Graph,
    station_points: Dict[str, Point],
    origin: Point,
    horse_kmh: float,
) -> Dict[str, float]:
    """
    Returns shortest travel time to each station (minutes).
    """
    v_mpm = meters_per_minute(horse_kmh)

    H = G.copy()
    H.add_node("ORIGIN")

    for st, pt in station_points.items():
        t_min = origin.distance(pt) / v_mpm
        H.add_edge("ORIGIN", st, weight=float(t_min))

    dist = nx.single_source_dijkstra_path_length(H, "ORIGIN", weight="weight")
    dist.pop("ORIGIN", None)
    return dist


# ------------------------------------------------------------
# Isochrone geometry
# ------------------------------------------------------------

def isochrone_polygon(
    origin: Point,
    station_points: Dict[str, Point],
    time_to_station_min: Dict[str, float],
    T_min: float,
    horse_kmh: float,
):
    """
    Reachable area within T_min minutes.
    """
    v_mpm = meters_per_minute(horse_kmh)
    geoms = [origin.buffer(v_mpm * T_min)]

    used = 0
    for st, t in time_to_station_min.items():
        if t <= T_min:
            r = v_mpm * (T_min - t)
            if r > 0:
                geoms.append(station_points[st].buffer(r))
                used += 1

    log.info(f"T={T_min/60:.1f} h: {used:,} station-buffers (+ origin)")
    return unary_union(geoms)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Filled isochrones (hours) + rail overlay")
    ap.add_argument("--network", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)

    ap.add_argument("--horse_kmh", type=float, default=4.0)
    ap.add_argument("--hours", type=int, default=8)
    ap.add_argument("--step_hours", type=float, default=1.0)

    ap.add_argument("--work_crs", default="EPSG:3035")
    ap.add_argument("--out_png", default="isochrones_filled.png")
    ap.add_argument("--out_geojson", default="isochrones_filled.geojson")

    # Styling
    ap.add_argument("--title", default=None)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--rail_width", type=float, default=0.45)
    args = ap.parse_args()

    work_crs = CRS.from_user_input(args.work_crs)

    # Boundary
    boundary = gpd.read_file(args.boundary)
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")
    boundary_w = boundary.to_crs(work_crs)
    boundary_geom = boundary_w.geometry.iloc[0]

    # Network
    G, station_points, rail_edges = build_graph_and_points(args.network, work_crs)

    # Origin
    origin = (
        gpd.GeoSeries([Point(args.lon, args.lat)], crs="EPSG:4326")
        .to_crs(work_crs)
        .iloc[0]
    )

    log.info("Computing shortest times to all stations...")
    time_to_station = dijkstra_from_origin(G, station_points, origin, args.horse_kmh)
    log.info(f"Computed times to {len(time_to_station):,} stations")

    # --- Hour thresholds (converted to minutes internally) ---
    step_min = args.step_hours * 60.0
    max_min = args.hours * 60.0
    thresholds_min = [t for t in frange(step_min, max_min, step_min)]

    cumulative = []
    for T_min in thresholds_min:
        poly = isochrone_polygon(
            origin, station_points, time_to_station, T_min, args.horse_kmh
        ).intersection(boundary_geom)
        cumulative.append((T_min, poly))

    # Rings
    rings = []
    prev = None
    for T_min, poly in cumulative:
        ring = poly if prev is None else poly.difference(prev)
        rings.append({
            "from_h": 0.0 if prev is None else (T_min - step_min) / 60.0,
            "to_h": T_min / 60.0,
            "geometry": ring,
        })
        prev = poly

    rings_gdf = gpd.GeoDataFrame(rings, crs=work_crs)
    rings_gdf = rings_gdf[~rings_gdf.is_empty].copy()

    # Save GeoJSON
    rings_gdf.to_crs("EPSG:4326").to_file(args.out_geojson, driver="GeoJSON")
    log.info(f"Wrote {args.out_geojson} (rings: {len(rings_gdf):,})")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 10))

    gpd.GeoSeries([boundary_geom], crs=work_crs).boundary.plot(ax=ax, linewidth=1)

    rings_gdf.plot(
        ax=ax,
        column="to_h",
        cmap=args.cmap,
        alpha=args.alpha,
        linewidth=0.0,
        legend=True,
        legend_kwds={"label": "Reachable within (hours)", "shrink": 0.6},
        zorder=1,
    )

    if len(rail_edges) > 0:
        rail_edges.plot(
            ax=ax,
            color="black",
            linewidth=args.rail_width,
            alpha=0.9,
            zorder=3,
        )

    gpd.GeoSeries([origin], crs=work_crs).plot(ax=ax, markersize=30, zorder=4)

    title = args.title

    ax.set_title(title)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=250)
    log.info(f"Wrote {args.out_png}")
    return 0


def frange(start, stop, step):
    x = start
    while x <= stop + 1e-9:
        yield x
        x += step


if __name__ == "__main__":
    raise SystemExit(main())
