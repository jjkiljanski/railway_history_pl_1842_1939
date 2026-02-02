from __future__ import annotations

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

UNION_IN = "data_preprocessed/districts_1934_10_1_union.geojson"
LINES_IN = "data_preprocessed/lines.csv"
STATIONS_IN = "data/stations.csv"
LINES_OUT = "data_preprocessed/lines_cropped.csv"


def parse_coords_to_point(coords: str) -> Point | None:
    """
    Expects coords like: "50.615278, 16.124167" meaning "lat, lon".
    Returns Point(lon, lat) or None if missing/bad.
    """
    if coords is None or (isinstance(coords, float) and pd.isna(coords)):
        return None
    s = str(coords).strip()
    if not s:
        return None

    parts = [p.strip() for p in s.split(",")]
    if len(parts) < 2:
        return None

    try:
        lat = float(parts[0])
        lon = float(parts[1])
    except ValueError:
        return None

    return Point(lon, lat)


def main() -> int:
    # Load union polygon
    union_gdf = gpd.read_file(UNION_IN)
    if union_gdf.empty:
        raise RuntimeError(f"No features found in {UNION_IN}")

    union_geom = union_gdf.geometry.iloc[0]
    union_crs = union_gdf.crs

    # Load stations (semicolon-separated)
    stations = pd.read_csv(STATIONS_IN, sep=";", dtype=str).fillna("")
    if "station" not in stations.columns or "coords" not in stations.columns:
        raise RuntimeError("stations_geo.csv must have columns: station;coords;wiki_link (wiki_link optional)")

    station_to_point: dict[str, Point | None] = {}
    for _, row in stations.iterrows():
        name = str(row["station"]).strip()
        if not name:
            continue
        station_to_point[name] = parse_coords_to_point(row["coords"])

    # Load lines (semicolon-separated)
    lines = pd.read_csv(LINES_IN, sep=";", dtype=str).fillna("")
    if "station_1" not in lines.columns or "station_2" not in lines.columns:
        raise RuntimeError("lines.csv must have columns: station_1;station_2;...")

    # Build endpoint GeoSeries (station coords are lat/lon -> EPSG:4326)
    s1_points = lines["station_1"].map(lambda n: station_to_point.get(str(n).strip()))
    s2_points = lines["station_2"].map(lambda n: station_to_point.get(str(n).strip()))

    g1 = gpd.GeoSeries(s1_points, crs="EPSG:4326")
    g2 = gpd.GeoSeries(s2_points, crs="EPSG:4326")

    # Reproject points to union CRS if needed
    if union_crs is not None and str(union_crs).lower() != "epsg:4326":
        g1 = g1.to_crs(union_crs)
        g2 = g2.to_crs(union_crs)

    # Inside test (covers includes boundary points)
    inside1 = g1.apply(lambda p: False if p is None else union_geom.covers(p))
    inside2 = g2.apply(lambda p: False if p is None else union_geom.covers(p))

    # Keep if at least one endpoint inside
    keep = inside1 | inside2

    cropped = lines.loc[keep].copy()
    cropped.to_csv(LINES_OUT, sep=";", index=False)

    print(f"Kept {len(cropped)} / {len(lines)} lines. Wrote: {LINES_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
