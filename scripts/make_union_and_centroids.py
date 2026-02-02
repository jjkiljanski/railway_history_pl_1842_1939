from __future__ import annotations

import geopandas as gpd

DISTRICTS_IN = "data/districts_1934_10_1.geojson"
UNION_OUT = "data_preprocessed/districts_1934_10_1_union.geojson"
CENTROIDS_OUT = "data_preprocessed/districts_1934_10_1_centroids.geojson"

DISTRICT_NAME_COL = "District"  # change if your column name differs


def main() -> int:
    gdf = gpd.read_file(DISTRICTS_IN)
    if gdf.empty:
        raise RuntimeError(f"No features found in {DISTRICTS_IN}")

    # -----------------------
    # 1) Union of geometries
    # -----------------------
    union_geom = gdf.geometry.union_all()
    union_gdf = gpd.GeoDataFrame([{"id": "union", "geometry": union_geom}], crs=gdf.crs)
    union_gdf.to_file(UNION_OUT, driver="GeoJSON")
    print(f"Wrote union GeoJSON to: {UNION_OUT}")

    # -----------------------
    # 2) District centroids
    # -----------------------
    # NOTE: centroids in geographic CRS (lat/lon) can be slightly off for odd shapes.
    # For better geometric correctness, compute in a projected CRS and convert back.
    # Here we do that automatically: project to EPSG:3857, compute centroids, reproject back.
    gdf_proj = gdf.to_crs("EPSG:3857") if gdf.crs is not None else gdf.set_crs("EPSG:4326").to_crs("EPSG:3857")
    centroids_proj = gdf_proj.geometry.centroid

    centroids_gdf = gdf_proj.copy()
    centroids_gdf["geometry"] = centroids_proj
    centroids_gdf = centroids_gdf.to_crs(gdf.crs if gdf.crs is not None else "EPSG:4326")

    # Keep only a small set of useful columns
    cols_to_keep = [c for c in [DISTRICT_NAME_COL] if c in centroids_gdf.columns]
    centroids_gdf = centroids_gdf[cols_to_keep + ["geometry"]]

    centroids_gdf.to_file(CENTROIDS_OUT, driver="GeoJSON")
    print(f"Wrote centroids GeoJSON to: {CENTROIDS_OUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
