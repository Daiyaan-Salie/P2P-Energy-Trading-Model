# ==========================================================
# S3b Comparator – Thesis Plots (S3C Ginis + Fair-Share)
# FIXED v4.1:
#   - fixes NumPy truth-value curtailment selection (arrays in `or` chain)
#   - runs Baseline / Fair-Share / Dynamic / Hybrid
#   - outputs KPI CSV
#   - generates the SAME "THESIS__" plots as S3b_comparator_THESIS_PLOTS.py
#   - NaN-safe MCP boxplot (drops NaN-only series)
# ==========================================================

import os
import re
import glob
from contextlib import contextmanager
import importlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config
import Common_Functions as common

# ==========================================================
# Output location (always relative to this script)
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@contextmanager
def pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"WARNING: Skipping '{module_name}' (module not found).")
        return None


def _slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _flatten_sum(x):
    if x is None:
        return np.nan
    try:
        return float(np.nansum(np.array(x, dtype=float)))
    except Exception:
        return np.nan


def _rename_new_pngs(before: set, after: set, model_name: str):
    """
    If any model functions generate PNGs while we're running inside OUTPUT_DIR,
    prefix them with the model name to avoid overwrites.
    """
    slug = _slug(model_name)
    new_files = sorted([f for f in (after - before) if f.lower().endswith(".png")])
    for fname in new_files:
        base = os.path.basename(fname)
        if base.startswith(slug + "__"):
            continue
        try:
            os.replace(fname, os.path.join(OUTPUT_DIR, f"{slug}__{base}"))
        except OSError:
            pass


# ==========================================================
# KPI extraction
# ==========================================================
def process_results(model_name: str, results: dict) -> dict:
    trades_df = results.get("trades_df", pd.DataFrame())

    total_imports = float(np.sum(results.get("imports_kwh", 0)))
    total_exports = float(np.sum(results.get("exports_kwh", 0)))

    # --- FIX: explicit key checks (no boolean eval on arrays) ---
    if "curtailment_kwh" in results:
        curtailed = results["curtailment_kwh"]
    elif "curtailed_kwh" in results:
        curtailed = results["curtailed_kwh"]
    elif "curtailed_energy" in results:
        curtailed = results["curtailed_energy"]
    else:
        curtailed = None

    curtailed_total = _flatten_sum(curtailed)

    if trades_df.empty:
        return {
            "Model": model_name,
            "Total System Benefit (ZAR)": 0.0,
            "Gini_P2P_Benefits": np.nan,
            "Total P2P kWh Traded": 0.0,
            "Avg. MCP (ZAR/kWh)": np.nan,
            "MCP Volatility (Std. Dev.)": np.nan,
            "Total Grid Imports (kWh)": total_imports,
            "Total Grid Exports (kWh)": total_exports,
            "Curtailed Energy (kWh)": curtailed_total,
        }

    welfare = common.calculate_economic_welfare(trades_df, config.UTILITY_TARIFF)
    fairness = common.calculate_fairness_metrics(
        welfare["consumer_surplus_per_agent"],
        welfare["prosumer_profit_per_agent"],
    )

    mcp_series = results.get("mcp_series")
    if mcp_series is None and "market_price" in trades_df.columns:
        mcp_series = trades_df["market_price"]
    mcp_series = pd.Series(mcp_series).dropna() if mcp_series is not None else pd.Series(dtype=float)

    return {
        "Model": model_name,
        "Total System Benefit (ZAR)": float(welfare["total_system_benefit"]),
        "Gini_P2P_Benefits": float(fairness.get("gini_index", np.nan)),
        "Total P2P kWh Traded": float(trades_df["p2p_trade_volume"].sum()) if "p2p_trade_volume" in trades_df.columns else 0.0,
        "Avg. MCP (ZAR/kWh)": float(mcp_series.mean()) if not mcp_series.empty else np.nan,
        "MCP Volatility (Std. Dev.)": float(mcp_series.std()) if not mcp_series.empty else np.nan,
        "Total Grid Imports (kWh)": total_imports,
        "Total Grid Exports (kWh)": total_exports,
        "Curtailed Energy (kWh)": curtailed_total,
    }


def run_all_models():
    all_kpis = []
    all_results = {}

    alpha = getattr(config, "FAIRNESS_ALPHA", 0.15)

    # Baseline
    baseline = _safe_import("S3b_baseline_MCP_model")
    if baseline is not None:
        print("Running Model 1: Baseline (Noisy)...")
        with pushd(OUTPUT_DIR):
            before = set(glob.glob("*.png"))
            res = baseline.run_s3b_simulation()
            after = set(glob.glob("*.png"))
            _rename_new_pngs(before, after, "Baseline (Noisy)")
        all_kpis.append(process_results("Baseline (Noisy)", res))
        all_results["Baseline (Noisy)"] = res

    # Fair-Share
    fairshare = _safe_import("S3b_fairshare_MCP_model")
    if fairshare is not None:
        print("Running Model 2: Fair-Share...")
        with pushd(OUTPUT_DIR):
            before = set(glob.glob("*.png"))
            res = fairshare.run_sim(ALPHA=alpha)
            after = set(glob.glob("*.png"))
            _rename_new_pngs(before, after, "Fair-Share")
        all_kpis.append(process_results("Fair-Share", res))
        all_results["Fair-Share"] = res

    # Dynamic
    dynamic = _safe_import("S3b_dynamic_MCP_model")
    if dynamic is not None:
        print("Running Model 3: Dynamic (Strategic)...")
        with pushd(OUTPUT_DIR):
            before = set(glob.glob("*.png"))
            res = dynamic.run_sim()
            after = set(glob.glob("*.png"))
            _rename_new_pngs(before, after, "Dynamic (Strategic)")
        all_kpis.append(process_results("Dynamic (Strategic)", res))
        all_results["Dynamic (Strategic)"] = res

    # Hybrid
    hybrid = _safe_import("S3b_hybrid_MCP_model")
    if hybrid is not None:
        print(f"Running Model 4: Hybrid (Alpha={alpha})...")
        with pushd(OUTPUT_DIR):
            before = set(glob.glob("*.png"))
            res = hybrid.run_sim(ALPHA=alpha)
            after = set(glob.glob("*.png"))
            _rename_new_pngs(before, after, "Hybrid")
        all_kpis.append(process_results(f"Hybrid (Alpha={alpha})", res))
        all_results[f"Hybrid (Alpha={alpha})"] = res

    if not all_kpis:
        return pd.DataFrame(), all_results

    return pd.DataFrame(all_kpis).set_index("Model"), all_results


# ==========================================================
# Thesis comparison plots
# ==========================================================
def _first_present(d: dict, keys: list):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return None


def extract_mcp_series(res: dict):
    """Try common keys for MCP per-interval series."""
    if not isinstance(res, dict):
        return None
    series = _first_present(res, ["mcp_series", "mcp", "mcp_per_interval", "mcp_interval", "mcp_values"])
    if series is None and isinstance(res.get("prices", None), dict):
        series = _first_present(res["prices"], ["mcp_series", "mcp"])
    if series is None:
        return None
    try:
        return list(series)
    except Exception:
        return None


def plot_kpi_tradeoff(kpi_df: pd.DataFrame):
    """(1) Welfare vs fairness trade-off scatter."""
    if "Gini Index" not in kpi_df.columns or "Total System Benefit (ZAR)" not in kpi_df.columns:
        print("...skipping trade-off plot (missing Gini or Benefit).")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(kpi_df["Gini Index"], kpi_df["Total System Benefit (ZAR)"], s=90)

    for model in kpi_df.index:
        ax.annotate(
            model,
            (kpi_df.loc[model, "Gini Index"], kpi_df.loc[model, "Total System Benefit (ZAR)"]),
            textcoords="offset points",
            xytext=(6, 6),
        )

    ax.set_xlabel("Gini Index (lower = fairer)")
    ax.set_ylabel("Total System Benefit (ZAR)")
    ax.set_title("Pricing Model Trade-off: Welfare vs Fairness")
    ax.grid(True)

    out = os.path.join(OUTPUT_DIR, "THESIS__01_tradeoff_welfare_vs_fairness.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"...saved {out}")


def plot_price_volatility_box(all_results: dict):
    """(2) MCP distribution / volatility boxplot (NaN-safe)."""
    price_data = {}
    for name, res in all_results.items():
        series = extract_mcp_series(res)
        if series is None or len(series) == 0:
            continue

        # Many models log NaN for intervals with no clearing/trades.
        s = pd.Series(series, dtype=float).dropna()
        s = s[np.isfinite(s.values)]

        if len(s) > 0:
            price_data[name] = s.values

    if not price_data:
        print("...skipping MCP volatility boxplot (no valid MCP values after dropping NaNs).")
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot(price_data.values(), tick_labels=list(price_data.keys()), vert=True)
    plt.ylabel("MCP (ZAR/kWh)")
    plt.title("MCP Volatility Across Pricing Models")
    plt.grid(True)

    out = os.path.join(OUTPUT_DIR, "THESIS__02_mcp_volatility_boxplot.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"...saved {out}")


def plot_mcp_time_series_overlay(all_results: dict):
    """(3) MCP time-series overlay across models."""
    series_map = {}
    for name, res in all_results.items():
        s = extract_mcp_series(res)
        if s is None or len(s) == 0:
            continue
        s2 = pd.Series(s, dtype=float)
        if np.isfinite(s2.values).any():
            series_map[name] = s2.values

    if len(series_map) < 2:
        print("...skipping MCP overlay (need >=2 MCP series).")
        return

    plt.figure(figsize=(10, 5))
    for name, svals in series_map.items():
        plt.plot(range(len(svals)), svals, label=name)

    plt.xlabel("Interval (t)")
    plt.ylabel("MCP (ZAR/kWh)")
    plt.title("MCP Time-Series Overlay")
    plt.grid(True)
    plt.legend()

    out = os.path.join(OUTPUT_DIR, "THESIS__03_mcp_time_series_overlay.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"...saved {out}")


def plot_energy_breakdown(kpi_df: pd.DataFrame):
    """(4) Energy flow breakdown by model (stacked bar)."""
    cols = ["Total P2P kWh Traded", "Total Grid Imports (kWh)", "Total Grid Exports (kWh)", "Curtailed Energy (kWh)"]
    present = [c for c in cols if c in kpi_df.columns and kpi_df[c].notna().any()]
    if not present:
        print("...skipping energy breakdown (missing energy KPIs).")
        return

    data = kpi_df[present].copy().fillna(0.0)

    ax = data.plot(kind="bar", stacked=True, figsize=(10, 5))
    ax.set_xlabel("Pricing Model")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("Energy Outcome Breakdown by Pricing Model")
    ax.grid(True, axis="y")

    out = os.path.join(OUTPUT_DIR, "THESIS__04_energy_breakdown_stacked.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"...saved {out}")


def plot_benefit_and_gini_bars(kpi_df: pd.DataFrame):
    """(5) Bars: total benefit and gini (two separate figures)."""
    needed = {"Total System Benefit (ZAR)", "Gini Index"}
    if not needed.issubset(set(kpi_df.columns)):
        print("...skipping benefit+gini bars (missing KPIs).")
        return

    x = list(range(len(kpi_df.index)))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x, kpi_df["Total System Benefit (ZAR)"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(kpi_df.index), rotation=15, ha="right")
    ax1.set_ylabel("Total System Benefit (ZAR)")
    ax1.set_title("Total System Benefit by Pricing Model")
    ax1.grid(True, axis="y")

    out1 = os.path.join(OUTPUT_DIR, "THESIS__05a_total_benefit_bar.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"...saved {out1}")

    fig, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(x, kpi_df["Gini Index"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(kpi_df.index), rotation=15, ha="right")
    ax2.set_ylabel("Gini Index (lower = fairer)")
    ax2.set_title("Fairness (Gini) by Pricing Model")
    ax2.grid(True, axis="y")

    out2 = os.path.join(OUTPUT_DIR, "THESIS__05b_gini_bar.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"...saved {out2}")


def main():
    kpi_df, all_results = run_all_models()

    if kpi_df.empty:
        print("No models were run.")
        return

    print("\n=== KPI Summary (S3b Comparator) ===")
    print(kpi_df)

    out_csv = os.path.join(OUTPUT_DIR, "S3b_comparator_kpis.csv")
    kpi_df.to_csv(out_csv)
    print(f"Saved KPI table to {out_csv}")

    # Thesis plot script expects "Gini Index"
    kpi_plot_df = kpi_df.rename(columns={"Gini_P2P_Benefits": "Gini Index"})

    # Thesis comparison plots
    plot_kpi_tradeoff(kpi_plot_df)
    plot_price_volatility_box(all_results)
    plot_mcp_time_series_overlay(all_results)
    plot_energy_breakdown(kpi_plot_df)
    plot_benefit_and_gini_bars(kpi_plot_df)

    plt.close("all")
    print(f"\nComparator complete. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
