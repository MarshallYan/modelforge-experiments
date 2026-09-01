#!/usr/bin/env python3
"""
download_runs.py

For a list of W&B training runs:
  1. Downloads the best model checkpoint file.
  2. Generates a loss summary (training losses over time).
  3. Extracts val/test metrics from run summary.
  4. Downloads train/val/test regression plot images.
  5. Writes a GitHub-flavoured Markdown report with results, plots, and links to checkpoints.

Usage:
    python download_runs.py \
        --entity  <wandb-entity>  \
        --project <wandb-project> \
        --runs    run1 run2 run3  \   # run IDs or names; omit to use all runs in project
        --output  README.md          # default: README.md
        --checkpoint-dir ./checkpoints

Configuration via environment variable:
    WANDB_API_KEY=<your-key>   (or run `wandb login` first)
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import wandb
except ImportError:
    sys.exit("wandb is not installed. Run:  pip install wandb")


# ---------------------------------------------------------------------------
# Metric key definitions
# ---------------------------------------------------------------------------

LOSS_KEYS = [
    "loss/per_atom_force/mse",
    "loss/per_system_energy/mse",
    "loss/per_system_dipole_moment/mse",
    "loss/per_atom_charge/mse",
    "loss/total_loss/mse",
]

VAL_KEYS_BEST = [
    "val/per_atom_force/mae",
    "val/per_atom_force/rmse",
    "val/per_system_energy/mae",
    "val/per_system_energy/rmse",
    "val/per_system_dipole_moment/mae",
    "val/per_system_dipole_moment/rmse",
    "val/per_atom_charge/mae",
    "val/per_atom_charge/rmse",
]

TEST_KEYS_BEST = [
    "test/per_atom_force/mae",
    "test/per_atom_force/rmse",
    "test/per_system_energy/mae",
    "test/per_system_energy/rmse",
    "test/per_system_dipole_moment/mae",
    "test/per_system_dipole_moment/rmse",
    "test/per_atom_charge/mae",
    "test/per_atom_charge/rmse",
]

REGRESSION_PLOT_KEYS = [
    "train/regression_plot",
    "val/regression_plot",
    "test/regression_plot",
]

PLOT_LABEL = {
    "train/regression_plot": "Train",
    "val/regression_plot":   "Validation",
    "test/regression_plot":  "Test",
}


# ---------------------------------------------------------------------------
# API / run helpers
# ---------------------------------------------------------------------------

def get_api() -> wandb.Api:
    """Get wandb api object."""
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        return wandb.Api(api_key=api_key)
    return wandb.Api()


def fetch_runs(api: wandb.Api, entity: str, project: str, run_ids: list[str]) -> list[wandb.Run]:
    """Fetch runs from wandb api.

    Parameters:
        api (wandb.Api): wandb api object
        entity (str): wandb entity
        project (str): wandb project
        run_ids (list[str]): wandb run IDs

    Returns:
        list[wandb.Artifact]: list of wandb artifact objects
    """
    path = f"{entity}/{project}"
    if run_ids:
        runs = []
        for rid in run_ids:
            try:
                runs.append(api.run(f"{path}/{rid}"))
            except Exception as exc:
                print(f"  ⚠  Could not fetch run '{rid}': {exc}", file=sys.stderr)
        return runs
    print(f"No run IDs specified – fetching all runs in {path} …")
    return list(api.runs(path))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def best_checkpoint(run) -> "wandb.Artifact | None":
    """
    Get the best checkpoint file

    Parameters:
        run (wandb.Run): wandb run object
    Returns:
        wandb.Artifact | None
    """
    try:
        artifacts = list(run.logged_artifacts())
    except Exception:
        return None
    if not artifacts:
        return None
    for art in artifacts:
        if "best" in art.name.lower() or "best" in art.type.lower():
            return art
    for art in artifacts:
        if art.type in ("model", "checkpoint"):
            return art
    return artifacts[0]


def download_checkpoint(artifact: "wandb.Artifact", dest_dir: Path) -> "Path | None":
    """Download checkpoint file
    Parameters:
        artifact (wandb.Artifact): wandb artifact object
        dest_dir (Path): destination directory

    Returns:
        wandb.Artifact | None

    """
    try:
        save_path = dest_dir / artifact.name.replace(":", "_")
        artifact.download(root=str(save_path))
        return save_path
    except Exception as exc:
        print(f"  ⚠  Download failed for {artifact.name}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Loss / metric extraction
# ---------------------------------------------------------------------------

def extract_losses(run) -> dict[str, list[tuple[int, float]]]:
    """Scan run history for LOSS_KEYS; return {key: [(step, value), ...]}."""
    # Discover which keys are actually present
    present = [k for k in LOSS_KEYS if k in (run.summary or {})]
    if not present:
        try:
            sample = next(run.scan_history(page_size=1), {})
            present = [k for k in LOSS_KEYS if k in sample]
        except Exception:
            pass

    losses: dict[str, list[tuple[int, float]]] = {k: [] for k in present}
    if not present:
        return losses

    try:
        for row in run.scan_history(keys=["_step"] + present):
            step = row.get("_step", 0)
            for k in present:
                v = row.get(k)
                if v is not None:
                    losses[k].append((int(step), float(v)))
    except Exception as exc:
        print(f"  ⚠  Could not read history for run {run.id}: {exc}", file=sys.stderr)

    return losses


def extract_summary_metrics(run, keys: list[str]) -> dict[str, float]:
    """Pull a flat list of metric keys from run.summary, returning present ones."""
    out = {}
    summary = run.summary or {}
    for k in keys:
        # W&B sometimes stores nested keys under a dict hierarchy
        v = summary.get(k)
        if v is None:
            # Try navigating nested dict: "val/per_atom_force/mae" → summary["val"]["per_atom_force"]["mae"]
            parts = k.split("/")
            node = summary
            for p in parts:
                if isinstance(node, dict):
                    node = node.get(p)
                else:
                    node = None
                    break
            v = node
        if v is not None:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def best_total_loss(run) -> "float | None":
    """Return the final total training loss, or None."""
    summary = run.summary or {}
    v = summary.get("loss/total_loss/mse")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return None


# ---------------------------------------------------------------------------
# Regression plot download
# ---------------------------------------------------------------------------

def download_regression_plots(run, plots_dir: Path) -> dict[str, Path]:
    """
    Download regression plot images logged under REGRESSION_PLOT_KEYS.

    W&B media images in run.summary are wandb.Image objects (or dicts with
    an '_type' of 'image-file').  We use the public Files API to fetch them.

    Returns {key: local_path} for successfully downloaded plots.
    """
    summary = run.summary or {}
    downloaded: dict[str, Path] = {}

    for key in REGRESSION_PLOT_KEYS:
        media = summary.get(key)
        if media is None:
            continue

        # Resolve the remote path inside the run's file store
        remote_path: str | None = None
        if isinstance(media, dict):
            remote_path = media.get("path") or media.get("_path")
        elif hasattr(media, "path"):          # wandb.Image / wandb.Media
            remote_path = media.path
        elif hasattr(media, "_path"):
            remote_path = media._path

        if not remote_path:
            print(f"  ⚠  Could not resolve file path for '{key}'", file=sys.stderr)
            continue

        # Sanitise key name → filename
        safe_key  = key.replace("/", "_")
        suffix    = Path(remote_path).suffix or ".png"
        local_path = plots_dir / f"{safe_key}{suffix}"

        try:
            f = run.file(remote_path)
            f.download(root=str(plots_dir), replace=True)
            # wandb downloads to <root>/<remote_path>; rename to our flat name
            downloaded_at = plots_dir / remote_path
            if downloaded_at.exists() and downloaded_at != local_path:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                downloaded_at.rename(local_path)
            elif downloaded_at.exists():
                pass   # already the right name
            downloaded[key] = local_path
            print(f"  ✓ Plot saved: {local_path}")
        except Exception as exc:
            print(f"  ⚠  Could not download '{key}': {exc}", file=sys.stderr)

    return downloaded



# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

_METRIC_LABEL = {
    "per_atom_force":         "Force (per atom)",
    "per_system_energy":      "Energy (per system)",
    "per_system_dipole_moment": "Dipole Moment (per system)",
    "per_atom_charge":        "Charge (per atom)",
    "total_loss":             "Total Loss",
}

def _friendly(key: str) -> str:
    """Turn 'val/per_atom_force/mae' into a readable label."""
    parts = key.split("/")
    # parts: [split, quantity, metric]  or  [loss, quantity, mse]
    quantity = _METRIC_LABEL.get(parts[1], parts[1]) if len(parts) > 1 else key
    metric   = parts[-1].upper() if len(parts) > 2 else ""
    return f"{quantity} {metric}".strip()


def _metrics_table(metrics: dict[str, float], caption: str) -> list[str]:
    """Render a two-column metric table."""
    if not metrics:
        return [f"_No {caption} metrics found._", ""]
    lines = [
        f"| Metric | Value |",
        f"|--------|------:|",
    ]
    for k, v in metrics.items():
        lines.append(f"| {_friendly(k)} | `{v:.6g}` |")
    return lines + [""]


def build_markdown(results: list[dict], entity: str, project: str, plots_dir: Path, output_path: Path) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# W&B Training Run Summary",
        "",
        f"> **Project:** `{entity}/{project}`  ",
        f"> **Generated:** {now}",
        "",
        "---",
        "",
    ]

    # ── Overview table ───────────────────────────────────────────────────────
    lines += [
        "## Overview",
        "",
        "| Run | Status | Total Loss (final) | Test Energy MAE (final) | Checkpoint |",
        "|-----|--------|--------------------|-------------------------|------------|",
    ]
    for r in results:
        run_link = f"[{r['name']}]({r['url']})"
        status   = r["state"].capitalize()
        loss     = f"`{r['best_loss']:.6g}`" if r["best_loss"] is not None else "—"
        print(r["test_metrics"].get("test/per_system_energy/mae"))
        test_energy = f"`{r["test_metrics"].get("test/per_system_energy/mae"):.6g}`" if r["test_metrics"].get("test/per_system_energy/mae") is not None else "—"
        print(test_energy)
        ckpt     = f"[Download]({r['checkpoint_rel']})" if r["checkpoint_rel"] else "—"
        lines.append(f"| {run_link} | {status} | {loss} | {test_energy} | {ckpt} |")

    lines += ["", "---", ""]

    # ── Per-run detail ────────────────────────────────────────────────────────
    lines.append("## Run Details")
    lines.append("")

    for r in results:
        lines += [
            f"### {r['name']}",
            "",
            f"- **Run ID:** `{r['id']}`",
            f"- **W&B URL:** {r['url']}",
            f"- **State:** {r['state']}",
            f"- **Created:** {r['created']}",
        ]
        if r["config"]:
            lines.append("<details>- <summary>Config:</summary>")
            for k, v in list(r["config"].items()):
                # we need to break up the lines to have better formatting
                # these are nested dictionaries
                if isinstance(v, dict):
                    lines.append(f"- `{k}`:\n")

                    for kk, vv in list(v.items()):

                        if isinstance(vv, dict):
                            lines.append(f"  - `{kk}`:\n")

                            for kk2, vv2 in list(vv.items()):
                                lines.append(f"    - `{kk2}`: `{vv2}`")
                        else:
                            lines.append(f"  - `{kk}`: `{vv}`\n")

                else:
                    # if we don't have a dictionary, let us break up based on = signs,
                    # but we don't want to break up for things within the parenthesis
                    # break up by =, identify pairs, then reassemble parts that have parenthesis
                    # if isinstance(v, str):
                    #     if "=" in v:
                    #         lines.append(f"- `{k}`:\n")
                    #         new_dict = {}
                    #         temp = v.split("=")
                    #         visited = list(range(len(temp)))
                    #
                    #         for i in range(len(temp)):
                    #             if visited[i] != -1:
                    #                 if i + 1 < len(temp):
                    #
                    #                     if "(" not in temp[i+1]:
                    #                         new_dict[temp[i]] = temp[i+1]
                    #                         visited[i+1] = -1
                    #                     else:
                    #
                    #                         not_terminated= True
                    #                         subs = []
                    #                         subs.append(temp[i+1])
                    #                         visited[i+1] = -1
                    #                         j = i+2
                    #                         visited[j] = -1
                    #                         top = temp[i+1]
                    #                         while not_terminated:
                    #                             print(temp)
                    #                             subs.append(temp[j])
                    #                             if ")" in temp[j]:
                    #                                 not_terminated = False
                    #                             else:
                    #                                 not_terminated = True
                    #
                    #                             j += 1
                    #                             visited[j] = -1
                    #                         # create the pairs
                    #                         new_dict2 = {}
                    #                         for i in range(len(subs), 2):
                    #                             new_dict2[subs[i]] = subs[i+1]
                    #
                    #                         new_dict[temp[i]] = new_dict2
                    #
                    #     else:
                    #         lines.append(f"- `{k}`: `{v}`\n")
                    #
                    # else:
                    #     lines.append(f"- `{k}`: `{v}`\n")
                    lines.append(f"- `{k}`: `{v}`\n")

        lines.append("</details>")


        # ── Validation metrics ────────────────────────────────────────────────
        lines.append("#### Validation Metrics")
        lines.append("")
        lines.extend(_metrics_table(r["val_metrics"], "validation"))

        # ── Test metrics ──────────────────────────────────────────────────────
        lines.append("#### Test Metrics")
        lines.append("")
        lines.extend(_metrics_table(r["test_metrics"], "test"))

        # ── Regression plots ──────────────────────────────────────────────────
        if r["plots"]:
            lines.append("#### Regression Plots")
            lines.append("")
            for key, local_path in r["plots"].items():
                label = PLOT_LABEL.get(key, key)
                # Make path relative to the output .md file so GitHub renders it
                try:
                    rel = Path(local_path).relative_to(output_path.parent)
                except ValueError:
                    rel = Path(local_path)
                lines.append(f"**{label}**")
                lines.append("")
                lines.append(f"![{label} Regression Plot]({rel})")
                lines.append("")
        else:
            lines.append("_No regression plot images found._\n")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if r["checkpoint_rel"]:
            lines += [
                "#### Model Checkpoint",
                "",
                f"Downloaded to [`{r['checkpoint_rel']}`]({r['checkpoint_rel']})",
                "",
            ]
        else:
            lines.append("_No model checkpoint artifact found._\n")

        lines += ["---", ""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Summarise W&B runs → Markdown report")
    parser.add_argument("--config", required=True, help="Path to the yaml configuration file")
    args = parser.parse_args()

    # read in the yaml file
    yaml_file = args.config
    import yaml
    with open(yaml_file) as f:
        config = yaml.safe_load(f)

    entity = config["entity"]
    project = config["project"]
    ckpt_dir = Path(config["checkpoint-dir"])
    plots_dir = Path(config["plots-dir"])
    output = Path(config["output"])

    if "runs" in config:
        runs_to_dl = config["runs"]
    else :
        runs_to_dl = []


    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Authenticating with W&B …")
    api = get_api()

    print(f"Fetching runs from {entity}/{project} …")
    runs = fetch_runs(api, entity, project, runs_to_dl)
    if not runs:
        sys.exit("No runs found – check entity/project names and your API key.")

    results = []
    for run in runs:
        print(f"\n▶ Processing run: {run.name}  ({run.id})")

        # Training losses
        print("  Extracting training loss history …")
        losses = extract_losses(run)
        for k, series in losses.items():
            print(f"    {k}: {len(series)} data points")

        # Val / test summary metrics
        print("  Extracting validation metrics …")
        val_metrics  = extract_summary_metrics(run, VAL_KEYS_BEST)
        print("  Extracting test metrics …")
        test_metrics = extract_summary_metrics(run, TEST_KEYS_BEST)
        for k, v in {**val_metrics, **test_metrics}.items():
            print(f"    {k}: {v:.6g}")

        # Regression plots
        print("  Downloading regression plots …")
        run_plots_dir = plots_dir / run.id
        run_plots_dir.mkdir(parents=True, exist_ok=True)
        plots = download_regression_plots(run, run_plots_dir)

        # Model checkpoint
        print("  Looking for model checkpoint artifact …")
        artifact  = best_checkpoint(run)
        ckpt_local: "Path | None" = None
        if artifact:
            print(f"  Downloading artifact: {artifact.name} …")
            ckpt_local = download_checkpoint(artifact, ckpt_dir)
            if ckpt_local:
                print(f"  ✓ Saved to {ckpt_local}")
        else:
            print("  No checkpoint artifact found.")

        results.append(dict(
            id             = run.id,
            name           = run.name,
            url            = run.url,
            state          = run.state,
            created        = str(run.created_at),
            config         = {k: v for k, v in run.config.items()
                              if not k.startswith("_")},
            best_loss      = best_total_loss(run),
            losses         = losses,
            val_metrics    = val_metrics,
            test_metrics   = test_metrics,
            plots          = plots,
            checkpoint_rel = str(ckpt_local) if ckpt_local else None,
        ))

    # Write report
    md = build_markdown(results, entity, project, plots_dir, output)
    output.write_text(md, encoding="utf-8")
    print(f"\n✅ Report written to {output.resolve()}")


if __name__ == "__main__":
    main()