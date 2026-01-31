# S3b_comparator_THESIS_PLOTS_S3C_GINIS.py
#
# Comparator for S3b pricing models (Baseline / Dynamic / Hybrid)
# - Forces all model-generated PNGs and comparator plots into ./outputs (next to this script)
# - Computes fairness using the SAME 3 Gini indices used in S3C results:
#     * Gini P2P Benefits (non-negative)
#     * Gini Utility Costs (non-negative, import-cost-based)
#     * Gini Net Outcomes (allows negatives via shifting)
#
# Notes:
# - P2P benefits are computed from trades_df using Common_Functions.calculate_economic_welfare()
# - Utility costs are estimated from per-agent grid imports (imports_kwh) * UTILITY_TARIFF
# - Net outcomes = P2P benefits - utility costs

import os
import re
import glob
from contextlib import contextmanager
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import Common_Functions as common


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
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "model"


def _rename_new_pngs(before: set, after: set, model_name: str) -> list:
    new_files = sorted([f for f in (after - before) if f.lower().endswith(".png")])
    slug = _slug(model_name)
    renamed = []
    for fname in new_files:
        base = os.path.basename(fname)
        if base.startswith(slug + "__"):
            continue
        new_name = f"{slug}__{base}"
        try:
            os.replace(fname, os.path.join(OUTPUT_DIR, new_name))
            renamed.append(new_name)
        except OSError:
            pass
    return renamed


def _flatten_sum(x):
    if x is None:
        return np.nan
    try:
        arr = np.array(x, dtype=float)
        return float(np.nansum(arr))
    except Exception:
        return np.nan


def _gini_nonneg(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    if np.allclose(x, 0.0):
        return 0.0
    x = np.clip(x, 0.0, None)
    return float(common.calculate_gini_index(x))


def _gini_shift_allow_negative(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    if np.allclose(x, 0.0):
        return 0.0
    mn = float(np.min(x))
    if mn < 0:
        x = x - mn
    if np.allclose(x, 0.0):
        return 0.0
    return float(common.calculate_gini_index(x))


def _infer_total_households(results: dict) -> int:
    n = getattr(config, "TOTAL_HOUSEHOLDS", None)
    if isinstance(n, (int, np.integer)) and n > 0:
        return int(n)

    imp = results.get("imports_kwh")
    if imp is not None:
        try:
            return int(np.asarray(imp).shape[0])
        except Exception:
            pass

    load = results.get("load_kwh")
    if load is not None:
        try:
            return int(np.asarray(load).shape[0])
        except Exception:
            pass

    return 0


def _series_all_agents(n: int) -> pd.Series:
    return pd.Series(0.0, index=pd.Index(range(n), name="agent_id"), dtype=float)


def compute_s3c_aligned_fairness(results: dict, trades_df: pd.DataFrame) -> dict:
    n = _infer_total_households(results)
    if n <= 0:
        ids = []
        if trades_df is not None and not trades_df.empty:
            for c in ("consumer_id", "prosumer_id"):
                if c in trades_df.columns:
                    ids += list(pd.to_numeric(trades_df[c], errors="coerce").dropna().astype(int).unique())
        n = (max(ids) + 1) if ids else 0

    if n <= 0:
        return {"Gini P2P Benefits": np.nan, "Gini Utility Costs": np.nan, "Gini Net Outcomes": np.nan}

    # P2P benefits = consumer surplus + prosumer profit
    if trades_df is None or trades_df.empty:
        benefits = _series_all_agents(n)
    else:
        welfare = common.calculate_economic_welfare(trades_df.copy(), config.UTILITY_TARIFF)
        cs = welfare.get("consumer_surplus_per_agent", pd.Series(dtype=float))
        pp = welfare.get("prosumer_profit_per_agent", pd.Series(dtype=float))

        cs.index = pd.to_numeric(cs.index, errors="coerce").fillna(-1).astype(int)
        pp.index = pd.to_numeric(pp.index, errors="coerce").fillna(-1).astype(int)

        benefits = _series_all_agents(n).add(cs, fill_value=0.0).add(pp, fill_value=0.0)

    # Utility costs = imports_kwh per agent * tariff (non-negative)
    imp = results.get("imports_kwh")
    if imp is None:
        utility_costs = _series_all_agents(n)
    else:
        imp_arr = np.asarray(imp, dtype=float)
        if imp_arr.ndim == 1:
            imp_per_agent = imp_arr
        else:
            imp_per_agent = np.nansum(imp_arr, axis=1)
        imp_per_agent = np.where(np.isfinite(imp_per_agent), imp_per_agent, 0.0)
        utility_costs = pd.Series(imp_per_agent[:n] * float(config.UTILITY_TARIFF), index=benefits.index, dtype=float)

    net_outcomes = benefits - utility_costs

    return {
        "Gini P2P Benefits": _gini_nonneg(benefits.values),
        "Gini Utility Costs": _gini_nonneg(utility_costs.values),
        "Gini Net Outcomes": _gini_shift_allow_negative(net_outcomes.values),
    }


def process_results(model_name: str, results: dict) -> dict:
    print(f"...processing {model_name}")

    trades_df = results.get("trades_df", pd.DataFrame())

    total_imports = float(np.nansum(np.asarray(results.get("imports_kwh", 0.0), dtype=float)))
    total_exports = float(np.nansum(np.asarray(results.get("exports_kwh", 0.0), dtype=float)))

    curtailed = results.get("curtailment_kwh", None)
    if curtailed is None:
        curtailed = results.get("curtailed_kwh", None)
    curtailed_total = _flatten_sum(curtailed)

    total_p2p_kwh = 0.0
    avg_mcp = np.nan
    std_mcp = np.nan

    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
        if "p2p_trade_volume" in trades_df.columns:
            total_p2p_kwh = float(pd.to_numeric(trades_df["p2p_trade_volume"], errors="coerce").fillna(0.0).sum())

        mcp_series = results.get("mcp_series")
        if mcp_series is None and "market_price" in trades_df.columns:
            mcp_series = trades_df["market_price"]

        if mcp_series is not None:
            s = pd.to_numeric(pd.Series(mcp_series), errors="coerce").dropna()
            if not s.empty:
                avg_mcp = float(s.mean())
                std_mcp = float(s.std())

    if not isinstance(trades_df, pd.DataFrame) or trades_df.empty:
        total_system_benefit = 0.0
    else:
        welfare = common.calculate_economic_welfare(trades_df.copy(), config.UTILITY_TARIFF)
        total_system_benefit = float(welfare.get("total_system_benefit", 0.0))

    fairness = compute_s3c_aligned_fairness(results, trades_df)

    row = {
        "Model": model_name,
        "Total System Benefit (ZAR)": total_system_benefit,
        "Total P2P kWh Traded": total_p2p_kwh,
        "Avg. MCP (ZAR/kWh)": avg_mcp,
        "MCP Volatility (Std. Dev.)": std_mcp,
        "Total Grid Imports (kWh)": total_imports,
        "Total Grid Exports (kWh)": total_exports,
        "Curtailed Energy (kWh)": curtailed_total,
    }
    row.update(fairness)
    return row


def _get_mcp_series(model_result: dict):
    if not isinstance(model_result, dict):
        return None
    s = model_result.get("mcp_series")
    if s is not None:
        s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
        return s if not s.empty else None

    trades_df = model_result.get("trades_df")
    if isinstance(trades_df, pd.DataFrame) and "market_price" in trades_df.columns:
        s = pd.to_numeric(trades_df["market_price"], errors="coerce").dropna()
        return s if not s.empty else None
    return None


def run_all_models():
    all_kpis = []
    all_results = {}

    alpha_to_run = getattr(config, "FAIRNESS_ALPHA", 0.15)

    model_baseline = _safe_import("S3b_baseline_MCP_model")
    if model_baseline is not None:
        print("Running Model 1: Baseline (Noisy)...")
        with pushd(OUTPUT_DIR):
            before = set(glob.glob("*.png"))
            res = model_baseline.run_s3b_simulation()
            after = set(glob.glob("*.png"))
            _rename_new_pngs(before, after, "Baseline (Noisy)")
        name = "Baseline (Noisy)"
        all_kpis.append(process_results(name, res))
        all_results[name] = res

    model_dynamic = _safe_import("S3b_dynamic_MCP_model")
    if model_dynamic is not None:
        print("\nRunning Model 2: Dynamic (Strategic)...")
        with pushd(OUTPUT_DIR):
            before = set(glob.glob("*.png"))
            res = model_dynamic.run_sim()
            after = set(glob.glob("*.png"))
            _rename_new_pngs(before, after, "Dynamic (Strategic)")
        name = "Dynamic (Strategic)"
        all_kpis.append(process_results(name, res))
        all_results[name] = res

    model_hybrid = _safe_import("S3b_hybrid_MCP_model")
    if model_hybrid is not None:
        print(f"\nRunning Model 3: Hybrid (Alpha={alpha_to_run})...")
        with pushd(OUTPUT_DIR):
            before = set(glob.glob("*.png"))
            res = model_hybrid.run_sim(ALPHA=alpha_to_run)
            after = set(glob.glob("*.png"))
            _rename_new_pngs(before, after, f"Hybrid (Alpha={alpha_to_run})")
        name = f"Hybrid (Alpha={alpha_to_run})"
        all_kpis.append(process_results(name, res))
        all_results[name] = res

    if not all_kpis:
        return pd.DataFrame(), all_results

    kpi_df = pd.DataFrame(all_kpis).set_index("Model")
    return kpi_df, all_results


def plot_tradeoff_welfare_vs_fairness(kpi_df: pd.DataFrame):
    xcol = "Gini P2P Benefits"
    ycol = "Total System Benefit (ZAR)"
    if xcol not in kpi_df.columns or ycol not in kpi_df.columns:
        print("...skipping THESIS__01 (missing columns).")
        return

    df = kpi_df[[xcol, ycol]].copy()
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    df = df.dropna()
    if df.empty:
        print("...skipping THESIS__01 (no valid numeric points).")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[xcol], df[ycol], s=90)

    for model in df.index:
        ax.annotate(model, (df.loc[model, xcol], df.loc[model, ycol]),
                    textcoords="offset points", xytext=(6, 6))

    ax.set_xlabel("Gini (P2P Benefits) — lower = fairer")
    ax.set_ylabel("Total System Benefit (ZAR)")
    ax.set_title("Pricing Model Trade-off: Welfare vs Fairness (P2P Benefits)")
    ax.grid(True)

    out = os.path.join(OUTPUT_DIR, "THESIS__01_tradeoff_welfare_vs_gini_p2p.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"...saved {out}")


def plot_mcp_volatility_boxplot(all_results: dict):
    price_data = {}
    for name, res in all_results.items():
        s = _get_mcp_series(res)
        if s is not None and len(s) > 0:
            price_data[name] = s.values

    if not price_data:
        print("...skipping THESIS__02 (no MCP series).")
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
    series_map = {}
    for name, res in all_results.items():
        s = _get_mcp_series(res)
        if s is not None and len(s) > 0:
            series_map[name] = s

    if len(series_map) < 2:
        print("...skipping THESIS__03 (need >=2 MCP series).")
        return

    plt.figure(figsize=(10, 5))
    for name, s in series_map.items():
        plt.plot(range(len(s)), s.values, label=name)

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


def plot_energy_breakdown_stacked(kpi_df: pd.DataFrame):
    cols = ["Total P2P kWh Traded", "Total Grid Imports (kWh)", "Total Grid Exports (kWh)", "Curtailed Energy (kWh)"]
    present = [c for c in cols if c in kpi_df.columns]
    if not present:
        print("...skipping THESIS__04 (missing energy columns).")
        return

    data = kpi_df[present].copy()
    for c in present:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0.0)

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


def plot_bars(kpi_df: pd.DataFrame):
    if "Total System Benefit (ZAR)" in kpi_df.columns:
        s = pd.to_numeric(kpi_df["Total System Benefit (ZAR)"], errors="coerce").fillna(0.0)
        x = list(range(len(s.index)))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x, s.values)
        ax.set_xticks(x)
        ax.set_xticklabels(list(s.index), rotation=15, ha="right")
        ax.set_ylabel("Total System Benefit (ZAR)")
        ax.set_title("Total System Benefit by Pricing Model")
        ax.grid(True, axis="y")
        out = os.path.join(OUTPUT_DIR, "THESIS__05a_total_benefit_bar.png")
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"...saved {out}")

    if "Gini P2P Benefits" in kpi_df.columns:
        g = pd.to_numeric(kpi_df["Gini P2P Benefits"], errors="coerce").fillna(0.0)
        x = list(range(len(g.index)))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x, g.values)
        ax.set_xticks(x)
        ax.set_xticklabels(list(g.index), rotation=15, ha="right")
        ax.set_ylabel("Gini (P2P Benefits)")
        ax.set_title("Fairness by Pricing Model (P2P Benefits)")
        ax.grid(True, axis="y")
        out = os.path.join(OUTPUT_DIR, "THESIS__05b_gini_p2p_bar.png")
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"...saved {out}")

    if "Gini Net Outcomes" in kpi_df.columns:
        g = pd.to_numeric(kpi_df["Gini Net Outcomes"], errors="coerce").dropna()
        if not g.empty:
            x = list(range(len(g.index)))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x, g.values)
            ax.set_xticks(x)
            ax.set_xticklabels(list(g.index), rotation=15, ha="right")
            ax.set_ylabel("Gini (Net Outcomes)")
            ax.set_title("Inequality of Final Outcomes (Net Outcomes)")
            ax.grid(True, axis="y")
            out = os.path.join(OUTPUT_DIR, "THESIS__05c_gini_net_outcomes_bar.png")
            plt.tight_layout()
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"...saved {out}")


def main():
    kpi_df, all_results = run_all_models()

    if kpi_df is None or kpi_df.empty:
        print("\nNo model results were produced. Nothing to compare.")
        print(f"Output folder: {OUTPUT_DIR}")
        return

    print("\n================ KPI TABLE ================\n")
    print(kpi_df)

    csv_out = os.path.join(OUTPUT_DIR, "S3b__comparator__kpis.csv")
    kpi_df.to_csv(csv_out)
    print(f"\n...saved {csv_out}")

    if "Total System Benefit (ZAR)" in kpi_df.columns:
        print("\n--- Sorted by Total System Benefit (desc) ---\n")
        print(kpi_df.sort_values("Total System Benefit (ZAR)", ascending=False))

    if "Gini P2P Benefits" in kpi_df.columns:
        print("\n--- Sorted by Gini P2P Benefits (asc, lower = fairer) ---\n")
        print(kpi_df.sort_values("Gini P2P Benefits", ascending=True))

    plot_tradeoff_welfare_vs_fairness(kpi_df)
    plot_mcp_volatility_boxplot(all_results)
    plot_mcp_time_series_overlay(all_results)
    plot_energy_breakdown_stacked(kpi_df)
    plot_bars(kpi_df)

    plt.close("all")
    print(f"\nComparator complete. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
