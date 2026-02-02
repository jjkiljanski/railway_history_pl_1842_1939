from __future__ import annotations

import argparse
import logging
import math
import time
from typing import Dict, Tuple, Optional, List

import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
from pyproj import Geod
import matplotlib.pyplot as plt

import json
import platform
from datetime import datetime, timezone
from pathlib import Path


# -------------------------
# Constants: border points
# -------------------------
# You provided them as (lon, lat) in WGS84.
GDANSK_CROSSINGS_WGS84 = {
    "Kozliny": (18.804394, 54.147935),
    "Kolibki": (18.55937,  54.46801),
    "Sulmin":  (18.47262,  54.30912),
}
ZBASZYN_WGS84 = ("Zbaszyn-Zbaszynek", (15.857159, 52.26061))
MAKOSZOWY_WGS84 = ("Makoszowy-Gliwice", (18.7499923, 50.2650933))


# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ma_net")


# -------------------------
# Geo helpers
# -------------------------
GEOD = Geod(ellps="WGS84")


def geodesic_m(p1: Point, p2: Point) -> float:
    _, _, dist_m = GEOD.inv(p1.x, p1.y, p2.x, p2.y)
    return float(dist_m)


def time_min_from_distance_m(dist_m: float, speed_kmh: float) -> float:
    speed_mps = speed_kmh * 1000.0 / 3600.0
    if speed_mps <= 0:
        return math.inf
    return (dist_m / speed_mps) / 60.0


def safe_point(geom) -> Optional[Point]:
    """If Point, return it; if Polygon/MultiPolygon, return centroid; else None."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Point":
        return geom
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom.centroid
    try:
        # fallback: try centroid
        return geom.centroid
    except Exception:
        return None


# -------------------------
# Load network GeoJSON
# -------------------------
def load_network_geojson_as_graph(path: str) -> Tuple[nx.Graph, Dict[str, Point], gpd.GeoDataFrame]:
    """
    Reads a GeoJSON produced by your network builder:
      - nodes: feature == 'node', station == <name>, geometry Point
      - edges: feature == 'edge', u, v, time_min, builder, ...
    Returns:
      - G graph (undirected) weighted by 'time_min'
      - station_points dict: station_name -> Point
      - original GeoDataFrame (for plotting base network by builder)
    """
    log.info(f"Loading network GeoJSON: {path}")
    gdf = gpd.read_file(path)
    if "feature" not in gdf.columns:
        raise RuntimeError("Network GeoJSON must have 'feature' column with values 'node'/'edge'.")

    nodes = gdf[gdf["feature"] == "node"].copy()
    edges = gdf[gdf["feature"] == "edge"].copy()

    log.info(f"Network features loaded: nodes={len(nodes):,}, edges={len(edges):,}")

    if "station" not in nodes.columns:
        raise RuntimeError("Node features must have 'station' property.")
    for col in ("u", "v", "time_min"):
        if col not in edges.columns:
            raise RuntimeError(f"Edge features must have '{col}' property.")

    station_points: Dict[str, Point] = {}
    bad_nodes = 0
    for _, r in nodes.iterrows():
        name = str(r["station"])
        geom = r.geometry
        if geom is None or geom.is_empty or geom.geom_type != "Point":
            bad_nodes += 1
            continue
        station_points[name] = geom

    if bad_nodes:
        log.info(f"Skipped {bad_nodes:,} network nodes with invalid geometry.")
    log.info(f"Stations with coordinates: {len(station_points):,}")

    G = nx.Graph()
    bad_edges = 0
    for _, r in edges.iterrows():
        u = str(r["u"])
        v = str(r["v"])
        t = r["time_min"]
        if t is None or (isinstance(t, float) and math.isnan(t)):
            bad_edges += 1
            continue

        attrs = r.drop(labels=["geometry"]).to_dict()
        attrs["time_min"] = float(t)

        # ensure numeric if present
        if "distance_m" in attrs and attrs["distance_m"] not in (None, ""):
            try:
                attrs["distance_m"] = float(attrs["distance_m"])
            except Exception:
                pass

        G.add_edge(u, v, **attrs)

    if bad_edges:
        log.info(f"Skipped {bad_edges:,} edges missing time_min.")
    log.info(f"Graph built: nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}")

    return G, station_points, gdf


# -------------------------
# Load external points
# -------------------------
def load_district_points(path: str) -> Tuple[Dict[str, Point], Optional[str]]:
    log.info(f"Loading districts: {path}")
    gdf = gpd.read_file(path)
    if "District" not in gdf.columns:
        raise RuntimeError("districts_1934_10_1_centroids.geojson must have column 'District'.")
    crs = gdf.crs.to_string() if gdf.crs is not None else None

    out: Dict[str, Point] = {}
    skipped = 0
    for _, r in gdf.iterrows():
        did = str(r["District"])
        pt = safe_point(r.geometry)
        if pt is None or pt.is_empty:
            skipped += 1
            continue
        out[did] = pt

    log.info(f"District points: {len(out):,} (skipped {skipped:,})")
    return out, crs


def load_city_points(path: str) -> Tuple[Dict[str, Point], Optional[str]]:
    log.info(f"Loading cities: {path}")
    gdf = gpd.read_file(path)
    if "City" not in gdf.columns:
        raise RuntimeError("cities_coords.geojson must have column 'City'.")
    crs = gdf.crs.to_string() if gdf.crs is not None else None

    out: Dict[str, Point] = {}
    skipped = 0
    for _, r in gdf.iterrows():
        cid = str(r["City"])
        pt = safe_point(r.geometry)
        if pt is None or pt.is_empty or pt.geom_type != "Point":
            skipped += 1
            continue
        out[cid] = pt

    log.info(f"City points: {len(out):,} (skipped {skipped:,})")
    return out, crs


def build_border_crossings() -> Dict[str, Point]:
    out: Dict[str, Point] = {}
    for name, (lon, lat) in GDANSK_CROSSINGS_WGS84.items():
        out[name] = Point(float(lon), float(lat))
    zb_name, (zb_lon, zb_lat) = ZBASZYN_WGS84
    out[zb_name] = Point(float(zb_lon), float(zb_lat))
    mk_name, (mk_lon, mk_lat) = MAKOSZOWY_WGS84
    out[mk_name] = Point(float(mk_lon), float(mk_lat))
    log.info(f"Border crossings created: {len(out):,}")
    return out


def ensure_epsg4326(points: Dict[str, Point], src_crs: Optional[str], label: str) -> Dict[str, Point]:
    """
    If source CRS is not EPSG:4326, reproject points to EPSG:4326.
    If CRS unknown, assume they are already EPSG:4326.
    """
    if src_crs is None:
        log.info(f"{label}: CRS unknown; assuming EPSG:4326.")
        return points
    if "4326" in src_crs:
        return points

    log.info(f"{label}: Reprojecting from {src_crs} to EPSG:4326...")
    gdf = gpd.GeoDataFrame(
        {"id": list(points.keys())},
        geometry=list(points.values()),
        crs=src_crs,
    ).to_crs("EPSG:4326")

    reproj = {row["id"]: row.geometry for _, row in gdf.iterrows()}
    return reproj


# -------------------------
# Nearest station (brute force)
# -------------------------
def find_nearest_station(p: Point, station_points: Dict[str, Point]) -> Tuple[str, float]:
    best_name = None
    best_dist = math.inf
    for name, sp in station_points.items():
        d = geodesic_m(p, sp)
        if d < best_dist:
            best_dist = d
            best_name = name
    if best_name is None:
        raise RuntimeError("No station points available for nearest-station connection.")
    return best_name, best_dist


# -------------------------
# Connect external points to graph
# -------------------------
def connect_external_points(
    G: nx.Graph,
    station_points: Dict[str, Point],
    districts: Dict[str, Point],
    cities: Dict[str, Point],
    borders: Dict[str, Point],
    horse_kmh: float,
    progress_every: int = 200,
) -> pd.DataFrame:
    """
    Adds nodes for District/City/Border_Crossing points and connects each to nearest station node.
    Uses horse speed for connector edge weights.
    Returns points metadata (for matrix and plotting).
    """
    log.info("Connecting external points to network by nearest station...")

    rows: List[dict] = []
    speed = float(horse_kmh)

    def add_group(pt_type: str, items: Dict[str, Point]):
        keys = list(items.keys())
        n = len(keys)
        t0 = time.time()
        for i, name in enumerate(keys, 1):
            pt = items[name]
            node_id = f"{pt_type}:{name}"

            nearest, d_m = find_nearest_station(pt, station_points)
            t_min = time_min_from_distance_m(d_m, speed)

            # Add node + connector edge
            G.add_node(node_id)
            G.add_edge(
                node_id, nearest,
                mode="horse",
                builder="point_connector",
                speed_kmh=speed,
                distance_m=float(d_m),
                time_min=float(t_min),
            )

            rows.append({
                "point_id": node_id,
                "name": name,
                "point_type": pt_type,
                "lon": float(pt.x),
                "lat": float(pt.y),
                "nearest_station": nearest,
                "connector_distance_m": float(d_m),
                "connector_time_min": float(t_min),
            })

            if i == 1 or i % progress_every == 0 or i == n:
                log.info(f"  {pt_type}: {i:,}/{n:,} connected (elapsed {time.time() - t0:.1f}s)")

    add_group("District", districts)
    add_group("City", cities)
    add_group("Border_Crossing", borders)

    meta = pd.DataFrame(rows)
    log.info(f"External points connected: {len(meta):,}")
    return meta


# -------------------------
# Compute matrix (long form)
# -------------------------
def compute_long_matrix(G: nx.Graph, points_meta: pd.DataFrame, weight: str = "time_min") -> pd.DataFrame:
    point_ids = points_meta["point_id"].tolist()
    type_map = dict(zip(points_meta["point_id"], points_meta["point_type"]))

    log.info(f"Computing shortest-path matrix among {len(point_ids):,} points (weight='{weight}')...")
    t0 = time.time()

    rows = []
    total = len(point_ids)
    for idx, src in enumerate(point_ids, 1):
        dist = nx.single_source_dijkstra_path_length(G, src, weight=weight)

        for dst in point_ids:
            val = dist.get(dst, math.inf)
            rows.append({
                "origin_id": src,
                "origin_type": type_map[src],
                "dest_id": dst,
                "dest_type": type_map[dst],
                "time_min": float(val) if math.isfinite(val) else math.inf,
            })

        if idx == 1 or idx % 10 == 0 or idx == total:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0.0
            log.info(f"  Sources done: {idx:,}/{total:,} (elapsed {elapsed:.1f}s, {rate:.2f} src/s)")

    df = pd.DataFrame(rows)
    log.info("Matrix computation finished.")
    return df


# -------------------------
# Plotting
# -------------------------
def plot_network_with_connectors(
    base_network_gdf: gpd.GeoDataFrame,
    station_points: Dict[str, Point],
    points_meta: pd.DataFrame,
    out_path: Optional[str],
    figsize: Tuple[float, float] = (12, 12),
    node_size: float = 2.0,
    ext_point_size: float = 12.0,
) -> None:
    """
    Plots:
      - base edges by builder: rail / transfer / component_bridge
      - connector edges (point_connector) derived from points_meta (as lines to nearest_station)
      - station nodes (black)
      - external points by type (different markers/colors)
    """
    log.info("Preparing plot...")

    edges = base_network_gdf[base_network_gdf["feature"] == "edge"].copy()
    nodes = base_network_gdf[base_network_gdf["feature"] == "node"].copy()

    # Base edges subsets
    rail = edges[edges.get("builder", "") == "rail"]
    transfer = edges[edges.get("builder", "") == "transfer"]
    bridge = edges[edges.get("builder", "") == "component_bridge"]

    # Connector edges: build geometry from external point -> nearest station
    connector_lines = []
    for _, r in points_meta.iterrows():
        pt = Point(float(r["lon"]), float(r["lat"]))
        st_name = r["nearest_station"]
        st_pt = station_points.get(st_name)
        if st_pt is None:
            continue
        connector_lines.append({
            "point_id": r["point_id"],
            "point_type": r["point_type"],
            "geometry": LineString([pt, st_pt]),
        })
    connectors_gdf = gpd.GeoDataFrame(connector_lines, crs="EPSG:4326")

    # External points GeoDF for plotting
    ext_points_gdf = gpd.GeoDataFrame(
        points_meta.copy(),
        geometry=[Point(xy) for xy in zip(points_meta["lon"], points_meta["lat"])],
        crs="EPSG:4326",
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Plot edges first
    if len(rail) > 0:
        rail.plot(ax=ax, color="steelblue", linewidth=0.8, label="Rail", zorder=1)
    if len(transfer) > 0:
        transfer.plot(ax=ax, color="darkorange", linewidth=0.6, alpha=0.7, label="Transfer", zorder=2)
    if len(bridge) > 0:
        bridge.plot(ax=ax, color="red", linewidth=1.2, linestyle="--", label="Component bridge", zorder=3)

    # Connector edges
    if len(connectors_gdf) > 0:
        connectors_gdf.plot(ax=ax, color="purple", linewidth=0.8, alpha=0.8, label="Point connectors", zorder=4)

    # Station nodes
    if len(nodes) > 0:
        nodes.plot(ax=ax, color="black", markersize=node_size, label="Stations", zorder=5)

    # External points by type
    # (Matplotlib markers differ; GeoPandas supports marker style in plot)
    for pt_type, color, marker in [
        ("District", "green", "s"),
        ("City", "magenta", "o"),
        ("Border_Crossing", "cyan", "^"),
    ]:
        sub = ext_points_gdf[ext_points_gdf["point_type"] == pt_type]
        if len(sub) > 0:
            sub.plot(ax=ax, color=color, markersize=ext_point_size, marker=marker, label=pt_type, zorder=6)

    ax.set_title("Network with external point connections")
    ax.set_axis_off()
    ax.legend(loc="lower left")
    plt.tight_layout()

    if out_path:
        log.info(f"Saving plot to: {out_path}")
        plt.savefig(out_path, dpi=300)
    else:
        log.info("Showing interactive plot window...")
        plt.show()

    plt.close(fig)
    log.info("Plot done.")

# -------------------------
# Create a manifest
# -------------------------


def write_manifest(
    *,
    manifest_path: str,
    args: argparse.Namespace,
    run_started_utc: datetime,
    run_duration_s: float,
    derived: dict,
    counts: dict,
):
    mp = Path(manifest_path)
    mp.parent.mkdir(parents=True, exist_ok=True)

    def _file_info(p: str) -> dict:
        pp = Path(p)
        info = {"path": str(pp)}
        try:
            st = pp.stat()
            info.update(
                {
                    "exists": True,
                    "size_bytes": int(st.st_size),
                    "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                }
            )
        except FileNotFoundError:
            info["exists"] = False
        return info

    # Normalize plot_out the same way as main()
    plot_out_effective = None if str(args.plot_out).lower() == "none" else str(args.plot_out)

    manifest = {
        "schema": "network_consumer_manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "started_utc": run_started_utc.isoformat(),
            "duration_s": float(run_duration_s),
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "parameters": {
            "network": str(args.network),
            "centroids": str(args.centroids),
            "cities": str(args.cities),
            "horse_kmh": float(args.horse_kmh),
            "out_points": str(args.out_points),
            "out_matrix": str(args.out_matrix),
            "plot_out": plot_out_effective,
            "figsize": [float(args.figsize[0]), float(args.figsize[1])],
            "node_size": float(args.node_size),
            "ext_point_size": float(args.ext_point_size),
            "progress_every": int(args.progress_every),
        },
        "derived": derived,
        "files": {
            "inputs": {
                "network": _file_info(str(args.network)),
                "centroids": _file_info(str(args.centroids)),
                "cities": _file_info(str(args.cities)),
            },
            "outputs": {
                "out_points": _file_info(str(args.out_points)),
                "out_matrix": _file_info(str(args.out_matrix)),
                **({"plot_out": _file_info(plot_out_effective)} if plot_out_effective else {}),
            },
        },
        "counts": counts,
    }

    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info(f"Wrote manifest {mp}")


# -------------------------
# Main
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Connect District/City/Border points to a routable network, plot connections, and compute distance matrix."
    )
    ap.add_argument("--network", required=True, help="Network GeoJSON (e.g. network_1934.geojson)")
    ap.add_argument("--centroids", required=True, help="districts_1934_10_1_centroids.geojson (District column)")
    ap.add_argument("--cities", required=True, help="cities_coords.geojson (City column)")
    ap.add_argument("--horse_kmh", type=float, default=4.0, help="Horse speed for point connectors (km/h)")

    ap.add_argument("--out_points", default="points_index.csv", help="Output points metadata CSV")
    ap.add_argument("--out_matrix", default="distance_matrix_long.csv", help="Output distance matrix CSV")

    # Plot controls
    ap.add_argument("--plot_out", default="network_with_point_connections.png",
                    help="Output plot image path. Use '--plot_out none' to disable saving and show interactively.")
    ap.add_argument("--figsize", type=float, nargs=2, default=(12, 12), help="Figure size, e.g. 14 14")
    ap.add_argument("--node_size", type=float, default=2.0, help="Station marker size")
    ap.add_argument("--ext_point_size", type=float, default=12.0, help="External point marker size")

    # Performance / logging knobs
    ap.add_argument("--progress_every", type=int, default=100, help="Log progress every N connections per group")
    
    # Manifest
    ap.add_argument("--manifest", default=None,
                help="Path to write JSON manifest (defaults to <out_matrix>.manifest.json)")
    args = ap.parse_args()

    run_started_utc = datetime.now(timezone.utc)
    t_all = time.time()

    # 1) Load network
    G, station_points, base_gdf = load_network_geojson_as_graph(args.network)
    if not station_points:
        raise RuntimeError("No station points found in the network GeoJSON nodes.")

    # 2) Load districts/cities + CRS handling
    districts, d_crs = load_district_points(args.centroids)
    cities, c_crs = load_city_points(args.cities)
    borders = build_border_crossings()  # WGS84 already

    # Convert to EPSG:4326 if needed
    districts = ensure_epsg4326(districts, d_crs, "Districts")
    cities = ensure_epsg4326(cities, c_crs, "Cities")

    # 3) Connect points
    points_meta = connect_external_points(
        G=G,
        station_points=station_points,
        districts=districts,
        cities=cities,
        borders=borders,
        horse_kmh=args.horse_kmh,
        progress_every=args.progress_every,
    )

    log.info(f"Graph after connecting points: nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}")

    # 4) Plot network + point connectors
    plot_out = None if str(args.plot_out).lower() == "none" else args.plot_out
    plot_network_with_connectors(
        base_network_gdf=base_gdf,
        station_points=station_points,
        points_meta=points_meta,
        out_path=plot_out,
        figsize=(args.figsize[0], args.figsize[1]),
        node_size=args.node_size,
        ext_point_size=args.ext_point_size,
    )

    # 5) Compute matrix
    mat = compute_long_matrix(G, points_meta, weight="time_min")

    # 6) Save outputs
    log.info(f"Writing points metadata: {args.out_points}")
    points_meta.to_csv(args.out_points, index=False)

    log.info(f"Writing distance matrix: {args.out_matrix}")
    mat.to_csv(args.out_matrix, index=False)

    log.info(f"All done in {time.time() - t_all:.1f}s")

    # 7) Manifest
    manifest_path = args.manifest or f"{args.out_matrix}.manifest.json"
    plot_out_effective = None if str(args.plot_out).lower() == "none" else str(args.plot_out)

    derived = {
        "plot_out_effective": plot_out_effective,
        "weight": "time_min",
    }

    counts = {
        # network as loaded
        "network_nodes_loaded": int(base_gdf[base_gdf["feature"] == "node"].shape[0]) if "feature" in base_gdf.columns else None,
        "network_edges_loaded": int(base_gdf[base_gdf["feature"] == "edge"].shape[0]) if "feature" in base_gdf.columns else None,

        # graph sizes
        "graph_nodes_after_connect": int(G.number_of_nodes()),
        "graph_edges_after_connect": int(G.number_of_edges()),

        # external points connected
        "external_points_total": int(points_meta.shape[0]),
        "external_points_by_type": points_meta["point_type"].value_counts().to_dict(),

        # matrix rows (n^2)
        "matrix_rows": int(mat.shape[0]),
    }

    write_manifest(
        manifest_path=manifest_path,
        args=args,
        run_started_utc=run_started_utc,
        run_duration_s=float(time.time() - t_all),
        derived=derived,
        counts=counts,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
