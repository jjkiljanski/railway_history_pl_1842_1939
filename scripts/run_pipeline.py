from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(*args: Any) -> None:
    """Run a subprocess command, echoing it for visibility."""
    cmd = [str(a) for a in args]
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def get_by_path(d: dict, dotted: str) -> Any:
    """Fetch a nested value using a dotted path (e.g., 'defaults.warsaw.lon')."""
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing key '{part}' while resolving '{dotted}'")
        cur = cur[part]
    return cur


def resolve_value(pipeline: dict, value: Any) -> Any:
    """
    Resolve special YAML fields:
      - strings like 'defaults.years' referenced via *_from keys
      - templates with {output_root} etc.
    """
    return value


def format_template(pipeline: dict, template: str, **kwargs: Any) -> Path:
    """Format a template string with common roots and provided kwargs, return an absolute Path."""
    paths = pipeline["paths"]
    mapping = {
        "scripts_root": paths["scripts_root"],
        "data_root": paths["data_root"],
        "data_preprocessed_root": paths["data_preprocessed_root"],
        "output_root": paths["output_root"],
        **kwargs,
    }
    rel = template.format(**mapping)
    return (REPO / rel).resolve()


def script_path(pipeline: dict, rel_script: str) -> Path:
    return (REPO / pipeline["paths"]["scripts_root"] / rel_script).resolve()


def years_from(pipeline: dict, spec: dict) -> list[int]:
    src = spec.get("years_from")
    if not src:
        raise KeyError("Expected 'years_from' in step")
    return list(get_by_path(pipeline, src))


def value_from(pipeline: dict, dotted: str) -> Any:
    return get_by_path(pipeline, dotted)


def profiles_from(pipeline: dict, spec: dict) -> list[str]:
    src = spec.get("profiles_from")
    if not src:
        return []
    return list(get_by_path(pipeline, src))


def run_template_runs(
    *,
    pipeline: dict,
    script: Path,
    runs: list[dict],
) -> None:
    for r in runs:
        cli: list[str] = []
        for k, v in r.items():
            if v is None:
                continue
            if isinstance(v, str) and "{" in v and "}" in v:
                v_path = format_template(pipeline, v)
                cli.extend([f"--{k}", str(v_path)])
            else:
                cli.extend([f"--{k}", str(v)])
        run(PY, script, *cli)

def parse_steps_arg(raw: str | None, available: list[str]) -> set[str] | None:
    """
    Parse --steps "a,b,c" into a set.
    Returns None if raw is None (meaning: run default/all).
    """
    if raw is None:
        return None
    wanted = {s.strip() for s in raw.split(",") if s.strip()}
    unknown = sorted(wanted - set(available))
    if unknown:
        raise SystemExit(
            f"Unknown step(s): {', '.join(unknown)}\n"
            f"Available steps: {', '.join(available)}"
        )
    return wanted


def clean_dir_contents(path: Path) -> None:
    """
    Remove all files and subdirectories under `path` while keeping `path` itself.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    if not path.is_dir():
        raise RuntimeError(f"Expected directory path for cleanup: {path}")

    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> None:
    pipeline = yaml.safe_load((REPO / "pipeline.yaml").read_text(encoding="utf-8"))

    steps = pipeline.get("steps", {})
    step_order = list(steps.keys())  # YAML order is preserved in Python 3.7+

    ap = argparse.ArgumentParser(description="Run the pipeline (YAML-driven).")
    ap.add_argument(
        "--steps",
        default=None,
        help=f"Comma-separated list of steps to run (in pipeline order). Available: {', '.join(step_order)}",
    )
    ap.add_argument("--list-steps", action="store_true", help="List available step names and exit.")
    ap.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean data_preprocessed and output directories before running.",
    )

    # keep your skip flags if you like; they still work
    ap.add_argument("--skip-plots", action="store_true", help="Skip plotting steps.")
    ap.add_argument("--skip-matrices", action="store_true", help="Skip distance matrix computation.")
    ap.add_argument("--skip-isochrones", action="store_true", help="Skip isochrone generation.")
    ap.add_argument("--skip-time-savings", action="store_true", help="Skip time-savings plots.")
    ap.add_argument("--skip-timeline", action="store_true", help="Skip timeline plots.")
    args = ap.parse_args()

    if args.list_steps:
        print("Available steps (in order):")
        for s in step_order:
            enabled = steps.get(s, {}).get("enabled", False)
            print(f"  - {s} {'(enabled)' if enabled else '(disabled)'}")
        return

    selected = parse_steps_arg(args.steps, step_order)  # None means: default/all enabled steps

    def step_is_selected(name: str) -> bool:
        return selected is None or name in selected

    if not args.no_clean:
        data_preprocessed_dir = (REPO / pipeline["paths"]["data_preprocessed_root"]).resolve()
        output_dir = (REPO / pipeline["paths"]["output_root"]).resolve()
        print(f"Cleaning {data_preprocessed_dir}")
        clean_dir_contents(data_preprocessed_dir)
        print(f"Cleaning {output_dir}")
        clean_dir_contents(output_dir)

    # ---- Step: preprocessing ----
    if step_is_selected("preprocessing"):
        prep = steps.get("preprocessing", {})
        if prep.get("enabled", False):
            for s in prep.get("scripts", []):
                run(PY, script_path(pipeline, s))

    # ---- Step: build networks ----
    if step_is_selected("build_networks"):
        bn = steps.get("build_networks", {})
        if bn.get("enabled", False):
            build_script = script_path(pipeline, bn["script"])
            for b in bn.get("builds", []):
                profile = str(b.get("profile", "default"))
                out = (
                    format_template(
                        pipeline,
                        bn["out_template"],
                        year=b["year"],
                        profile=profile,
                    )
                    if bn.get("out_template")
                    else None
                )
                manifest = (
                    format_template(
                        pipeline,
                        bn["manifest_template"],
                        year=b["year"],
                        profile=profile,
                    )
                    if bn.get("manifest_template")
                    else None
                )
                run(
                    PY, build_script,
                    "--year", b["year"],
                    "--speed_normal_kmh", b["speed_normal_kmh"],
                    "--speed_narrow_kmh", b["speed_narrow_kmh"],
                    *(["--out", out] if out else []),
                    *(["--manifest", manifest] if manifest else []),
                )

    # ---- Step: plot networks ----
    if step_is_selected("plot_networks"):
        pn = steps.get("plot_networks", {})
        if pn.get("enabled", False) and not args.skip_plots:
            plot_script = script_path(pipeline, pn["script"])
            years = years_from(pipeline, pn)
            profiles = profiles_from(pipeline, pn) or [None]
            for y in years:
                for profile in profiles:
                    profile_kwargs = {"profile": profile} if profile is not None else {}
                    network = format_template(pipeline, pn["network_path_template"], year=y, **profile_kwargs)
                    out = (
                        format_template(pipeline, pn["out_template"], year=y, **profile_kwargs)
                        if pn.get("out_template")
                        else None
                    )
                    run(
                        PY,
                        plot_script,
                        network,
                        *(["--out", out] if out else []),
                    )

    # ---- Step: distance matrices ----
    if step_is_selected("distance_matrices"):
        dm = steps.get("distance_matrices", {})
        if dm.get("enabled", False) and not args.skip_matrices:
            dm_script = script_path(pipeline, dm["script"])
            years = years_from(pipeline, dm)
            profiles = profiles_from(pipeline, dm) or [None]

            centroids = format_template(pipeline, dm["centroids"])
            cities = format_template(pipeline, dm["cities"])

            for y in years:
                for profile in profiles:
                    profile_kwargs = {"profile": profile} if profile is not None else {}
                    network = format_template(pipeline, dm["network_path_template"], year=y, **profile_kwargs)
                    out_points = format_template(pipeline, dm["out_points_template"], year=y, **profile_kwargs)
                    out_matrix = format_template(pipeline, dm["out_matrix_template"], year=y, **profile_kwargs)
                    plot_out = format_template(pipeline, dm["plot_out_template"], year=y, **profile_kwargs)
                    out_matrix_rail_km = (
                        format_template(pipeline, dm["out_matrix_rail_km_template"], year=y, **profile_kwargs)
                        if dm.get("out_matrix_rail_km_template")
                        else None
                    )
                    out_matrix_horse_km = (
                        format_template(pipeline, dm["out_matrix_horse_km_template"], year=y, **profile_kwargs)
                        if dm.get("out_matrix_horse_km_template")
                        else None
                    )

                    run(
                        PY, dm_script,
                        "--network", network,
                        "--centroids", centroids,
                        "--cities", cities,
                        "--out_points", out_points,
                        "--out_matrix", out_matrix,
                        "--plot_out", plot_out,
                        *(["--out_matrix_rail_km", out_matrix_rail_km] if out_matrix_rail_km else []),
                        *(["--out_matrix_horse_km", out_matrix_horse_km] if out_matrix_horse_km else []),
                    )

    # ---- Step: isochrones ----
    if step_is_selected("isochrones"):
        iso = steps.get("isochrones", {})
        if iso.get("enabled", False) and not args.skip_isochrones:
            iso_script = script_path(pipeline, iso["script"])
            years = years_from(pipeline, iso)
            profiles = profiles_from(pipeline, iso) or [None]

            boundary = format_template(pipeline, iso["boundary"])
            lon = value_from(pipeline, iso["lon_from"])
            lat = value_from(pipeline, iso["lat_from"])
            hours = value_from(pipeline, iso["hours_from"])

            for y in years:
                for profile in profiles:
                    profile_kwargs = {"profile": profile} if profile is not None else {}
                    network = format_template(pipeline, iso["network_path_template"], year=y, **profile_kwargs)
                    out_png = format_template(pipeline, iso["out_png_template"], year=y, **profile_kwargs)
                    out_geojson = format_template(pipeline, iso["out_geojson_template"], year=y, **profile_kwargs)
                    out_png_gray = (
                        format_template(pipeline, iso["out_png_gray_template"], year=y, **profile_kwargs)
                        if iso.get("out_png_gray_template")
                        else None
                    )

                    run(
                        PY, iso_script,
                        "--network", network,
                        "--boundary", boundary,
                        "--lon", lon,
                        "--lat", lat,
                        "--out_png", out_png,
                        "--out_geojson", out_geojson,
                        "--hours", hours,
                        *(["--out_png_gray", out_png_gray] if out_png_gray else []),
                    )

    # ---- Step: time savings ----
    if step_is_selected("time_savings"):
        ts = steps.get("time_savings", {})
        if ts.get("enabled", False) and not args.skip_time_savings:
            ts_script = script_path(pipeline, ts["script"])
            profiles = profiles_from(pipeline, ts) or [None]

            boundary = format_template(pipeline, ts["boundary"])
            lon = value_from(pipeline, ts["lon_from"])
            lat = value_from(pipeline, ts["lat_from"])
            hours = value_from(pipeline, ts["hours_from"])
            step_minutes = value_from(pipeline, ts["step_minutes_from"])

            for profile in profiles:
                profile_kwargs = {"profile": profile} if profile is not None else {}
                for y1, y2 in ts.get("pairs", []):
                    old_net = format_template(pipeline, ts["old_network_template"], year_1=y1, **profile_kwargs)
                    new_net = format_template(pipeline, ts["new_network_template"], year_2=y2, **profile_kwargs)
                    out_png = format_template(
                        pipeline,
                        ts["out_png_template"],
                        year_1=y1,
                        year_2=y2,
                        **profile_kwargs,
                    )
                    out_png_gray = (
                        format_template(
                            pipeline,
                            ts["out_png_gray_template"],
                            year_1=y1,
                            year_2=y2,
                            **profile_kwargs,
                        )
                        if ts.get("out_png_gray_template")
                        else None
                    )
                    out_geojson = format_template(
                        pipeline,
                        ts["out_geojson_template"],
                        year_1=y1,
                        year_2=y2,
                        **profile_kwargs,
                    )

                    run(
                        PY, ts_script,
                        "--old", old_net,
                        "--new", new_net,
                        "--boundary", boundary,
                        "--lon", lon,
                        "--lat", lat,
                        "--out_png", out_png,
                        "--out_geojson", out_geojson,
                        "--hours", hours,
                        "--step_minutes", step_minutes,
                        *(["--out_png_gray", out_png_gray] if out_png_gray else []),
                    )

    # ---- Step: timeline ----
    if step_is_selected("timeline"):
        tl = steps.get("timeline", {})
        if tl.get("enabled", False) and not args.skip_timeline:
            tl_script = script_path(pipeline, tl["script"])
            run_template_runs(
                pipeline=pipeline,
                script=tl_script,
                runs=tl.get("runs", []),
            )

    # ---- Step: built-between ----
    if step_is_selected("built_between"):
        bb = steps.get("built_between", {})
        if bb.get("enabled", False) and not args.skip_plots:
            bb_script = script_path(pipeline, bb["script"])
            run_template_runs(
                pipeline=pipeline,
                script=bb_script,
                runs=bb.get("runs", []),
            )

if __name__ == "__main__":
    main()
