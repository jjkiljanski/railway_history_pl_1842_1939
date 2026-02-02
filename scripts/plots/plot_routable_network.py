import argparse
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description="Plot routable rail network GeoJSON")
    ap.add_argument("geojson", help="Input network GeoJSON (e.g. network_1934.geojson)")
    ap.add_argument("--out", default = None, help="Output image file (PNG).")
    ap.add_argument("--figsize", type=float, nargs=2, default=(10, 10))
    args = ap.parse_args()

    out = args.out or f"output/networks/{Path(args.geojson).stem}.png"

    log.info(f"Loading {args.geojson}...")
    gdf = gpd.read_file(args.geojson)

    nodes = gdf[gdf["feature"] == "node"]
    edges = gdf[gdf["feature"] == "edge"]

    log.info(f"Nodes: {len(nodes):,}, edges: {len(edges):,}")

    # Split edges by builder
    rail = edges[edges["builder"] == "rail"]
    transfer = edges[edges["builder"] == "transfer"]
    bridge = edges[edges["builder"] == "component_bridge"]

    fig, ax = plt.subplots(figsize=args.figsize)

    # Plot order matters (draw bridges on top)
    if not rail.empty:
        rail.plot(
            ax=ax,
            color="steelblue",
            linewidth=0.8,
            label="Railway lines",
            zorder=1,
        )

    if not transfer.empty:
        transfer.plot(
            ax=ax,
            color="darkorange",
            linewidth=0.6,
            alpha=0.7,
            label="Proximity / transfer",
            zorder=2,
        )

    if not bridge.empty:
        bridge.plot(
            ax=ax,
            color="red",
            linewidth=1.2,
            linestyle="--",
            label="Component bridge",
            zorder=3,
        )

    if not nodes.empty:
        nodes.plot(
            ax=ax,
            color="black",
            markersize=2,
            label="Stations",
            zorder=4,
        )

    ax.set_title("Routable Railway Network")
    ax.set_axis_off()
    ax.legend(loc="lower left")

    plt.tight_layout()

    log.info(f"Saving plot to {out}")
    plt.savefig(out, dpi=300)


if __name__ == "__main__":
    main()
