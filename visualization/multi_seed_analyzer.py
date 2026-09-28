"""
Multi-Seed Statistical Analyzer
================================
Computes statistically valid metrics from N independent simulation runs.

Each run uses a different random seed (traffic matrix, carbon-profile assignments),
so per-run total-carbon values are genuinely i.i.d. across runs.  Within a run,
all methods share the same seed, making the design *paired* -- ideal for
a paired t-test / Wilcoxon signed-rank test.

Usage
-----
    from visualization.multi_seed_analyzer import MultiSeedAnalyzer
    analyzer = MultiSeedAnalyzer(per_seed_results)   # see run_multi_seed_experiment.py
    print(analyzer.generate_report())
    analyzer.save_plots('results/')
"""

import json
import os
import sys
import numpy as np
from scipy import stats

# Ensure UTF-8 output on Windows so report prints without UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python <3.7 fallback -- characters will be replaced silently


# ---------------------------------------------------------------------------
# Sawilowsky (2009) extended thresholds
# ---------------------------------------------------------------------------
def _sawilowsky_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.10:
        return "Negligible"
    elif ad < 0.20:
        return "Very Small"
    elif ad < 0.50:
        return "Small"
    elif ad < 0.80:
        return "Medium"
    elif ad < 1.20:
        return "Large"
    elif ad < 2.00:
        return "Very Large"
    else:
        return "Huge"


def _cohens_d_pooled(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Independent-samples Cohen's d using Hedges pooled-SD (ddof=1)."""
    n1, n2 = len(arr1), len(arr2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1 = np.std(arr1, ddof=1)
    s2 = np.std(arr2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(arr2) - np.mean(arr1)) / pooled)


def _block_bootstrap_ci(
    series: np.ndarray,
    block_size: int = 4,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Moving-block bootstrap CI for the mean of an autocorrelated series.

    Parameters
    ----------
    series     : 1-D array of hourly values (e.g., hourly carbon differences)
    block_size : contiguous block length (4 h approx one autocorrelation decay window)
    n_boot     : bootstrap replicates
    ci         : confidence level (default 0.95)
    rng        : optional numpy Generator for reproducibility

    Returns
    -------
    (lower, upper) CI bounds for the mean
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = len(series)
    # Build all possible starting indices for blocks
    n_blocks_needed = int(np.ceil(n / block_size))
    starts = np.arange(n - block_size + 1)

    boot_means = []
    for _ in range(n_boot):
        chosen = rng.choice(starts, size=n_blocks_needed, replace=True)
        boot_sample = np.concatenate([series[s : s + block_size] for s in chosen])[:n]
        boot_means.append(np.mean(boot_sample))

    alpha = (1.0 - ci) / 2
    lower = np.percentile(boot_means, 100 * alpha)
    upper = np.percentile(boot_means, 100 * (1 - alpha))
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class MultiSeedAnalyzer:
    """
    Analyzes results from N independent simulation runs.

    Parameters
    ----------
    per_seed_results : dict
        {
          seed (int): {
            'gnn':       {'total_carbon': float, 'carbon_history': list[float]},
            'ospf':      {'total_carbon': float, 'carbon_history': list[float]},
            'threshold': {'total_carbon': float, 'carbon_history': list[float]},
          },
          ...
        }
    """

    def __init__(self, per_seed_results: dict, duration_hours: int = None, num_nodes: int = None):
        self.results = per_seed_results
        self.seeds = sorted(per_seed_results.keys())
        self.n_runs = len(self.seeds)

        # Infer duration/nodes from results metadata if not explicitly passed
        first = per_seed_results[self.seeds[0]]
        if duration_hours is not None:
            self.duration_hours = duration_hours
        else:
            # fall back: not stored in old results
            self.duration_hours = None
        if num_nodes is not None:
            self.num_nodes = num_nodes
        else:
            self.num_nodes = first.get("num_nodes", None)

        # Extract per-run totals (one scalar per run -- truly independent)
        self.gnn_totals = np.array(
            [per_seed_results[s]["gnn"]["total_carbon"] for s in self.seeds],
            dtype=float,
        )
        self.ospf_totals = np.array(
            [per_seed_results[s]["ospf"]["total_carbon"] for s in self.seeds],
            dtype=float,
        )
        self.threshold_totals = np.array(
            [per_seed_results[s]["threshold"]["total_carbon"] for s in self.seeds],
            dtype=float,
        )
        # Paper baselines (present only in new-format results; fall back gracefully)
        self.linear_flow_carbon_totals = None
        self.nash_game_totals   = None
        if all("linear_flow_carbon" in per_seed_results[s] for s in self.seeds):
            self.linear_flow_carbon_totals = np.array(
                [per_seed_results[s]["linear_flow_carbon"]["total_carbon"] for s in self.seeds],
                dtype=float,
            )
        if all("nash_game" in per_seed_results[s] for s in self.seeds):
            self.nash_game_totals = np.array(
                [per_seed_results[s]["nash_game"]["total_carbon"] for s in self.seeds],
                dtype=float,
            )

        # Extract MLP validation if available
        self.mlp_totals = None
        if "mlp_predicted_carbon" in first["gnn"] and len(first["gnn"]["mlp_predicted_carbon"]) > 0:
            self.mlp_totals = np.array(
                [sum(per_seed_results[s]["gnn"]["mlp_predicted_carbon"]) for s in self.seeds],
                dtype=float,
            )

    # ------------------------------------------------------------------
    # Core statistical computations
    # ------------------------------------------------------------------

    def between_run_stats(self) -> dict:
        """Mean, SD, 95% CI for each method across N independent runs."""
        out = {}
        for name, arr in [
            ("gnn",          self.gnn_totals),
            ("ospf",         self.ospf_totals),
            ("threshold",    self.threshold_totals),
            ("linear_flow_carbon", self.linear_flow_carbon_totals),
            ("nash_game",      self.nash_game_totals),
        ]:
            if arr is None:
                continue
            n = len(arr)
            mean = float(np.mean(arr))
            sd = float(np.std(arr, ddof=1))
            se = sd / np.sqrt(n)
            t_crit = stats.t.ppf(0.975, df=n - 1)
            out[name] = {
                "n": n,
                "mean": mean,
                "sd": sd,
                "se": se,
                "ci_lower": mean - t_crit * se,
                "ci_upper": mean + t_crit * se,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "cv_pct": sd / mean * 100 if mean != 0 else 0,
            }
        return out

    def cohens_d_between_runs(self) -> dict:
        """
        Proper between-run Cohen's d:
          GNN vs OSPF    -- primary claim
          GNN vs Threshold
          OSPF vs Threshold
        All use independent-samples pooled-SD (Hedges formula, ddof=1).
        """
        d_gnn_ospf = _cohens_d_pooled(self.gnn_totals, self.ospf_totals)
        d_gnn_thr  = _cohens_d_pooled(self.gnn_totals, self.threshold_totals)
        d_ospf_thr = _cohens_d_pooled(self.threshold_totals, self.ospf_totals)

        out = {
            "gnn_vs_ospf": {
                "d": d_gnn_ospf,
                "label": _sawilowsky_label(d_gnn_ospf),
                "note": "OSPF mean - GNN mean; positive = GNN emits less",
            },
            "gnn_vs_threshold": {
                "d": d_gnn_thr,
                "label": _sawilowsky_label(d_gnn_thr),
                "note": "Threshold mean - GNN mean; positive = GNN emits less",
            },
            "threshold_vs_ospf": {
                "d": d_ospf_thr,
                "label": _sawilowsky_label(d_ospf_thr),
                "note": "OSPF mean - Threshold mean; positive = Threshold emits less",
            },
        }
        if self.linear_flow_carbon_totals is not None:
            d = _cohens_d_pooled(self.gnn_totals, self.linear_flow_carbon_totals)
            out["gnn_vs_linear_flow_carbon"] = {
                "d": d,
                "label": _sawilowsky_label(d),
                "note": "LinearFlowCarbon mean - GNN mean; positive = GNN emits less",
            }
        if self.nash_game_totals is not None:
            d = _cohens_d_pooled(self.gnn_totals, self.nash_game_totals)
            out["gnn_vs_nash_game"] = {
                "d": d,
                "label": _sawilowsky_label(d),
                "note": "NashGame mean - GNN mean; positive = GNN emits less",
            }
        return out

    def paired_ttest(self) -> dict:
        """
        Paired t-test (same seeds -> paired design eliminates between-run noise).
        H0: mean(OSPF total) == mean(GNN total)
        """
        t_stat, p_val = stats.ttest_rel(self.gnn_totals, self.ospf_totals)
        t_thr,  p_thr  = stats.ttest_rel(self.gnn_totals, self.threshold_totals)
        out = {
            "gnn_vs_ospf": {
                "t_statistic": float(t_stat),
                "p_value":     float(p_val),
                "significant": bool(p_val < 0.05),
                "df":          self.n_runs - 1,
            },
            "gnn_vs_threshold": {
                "t_statistic": float(t_thr),
                "p_value":     float(p_thr),
                "significant": bool(p_thr < 0.05),
                "df":          self.n_runs - 1,
            },
        }
        if self.linear_flow_carbon_totals is not None:
            t, p = stats.ttest_rel(self.gnn_totals, self.linear_flow_carbon_totals)
            out["gnn_vs_linear_flow_carbon"] = {
                "t_statistic": float(t), "p_value": float(p),
                "significant": bool(p < 0.05), "df": self.n_runs - 1,
            }
        if self.nash_game_totals is not None:
            t, p = stats.ttest_rel(self.gnn_totals, self.nash_game_totals)
            out["gnn_vs_nash_game"] = {
                "t_statistic": float(t), "p_value": float(p),
                "significant": bool(p < 0.05), "df": self.n_runs - 1,
            }
        return out

    def wilcoxon_test(self) -> dict:
        """
        Wilcoxon signed-rank test -- non-parametric backup for small n.
        Does not assume normality of differences.
        """
        try:
            w_stat, p_val = stats.wilcoxon(
                self.ospf_totals - self.gnn_totals, alternative="greater"
            )
            w_thr, p_thr = stats.wilcoxon(
                self.threshold_totals - self.gnn_totals, alternative="greater"
            )
            return {
                "gnn_vs_ospf": {
                    "statistic": float(w_stat),
                    "p_value": float(p_val),
                    "significant": bool(p_val < 0.05),
                    "alternative": "GNN carbon < OSPF carbon (one-sided)",
                },
                "gnn_vs_threshold": {
                    "statistic": float(w_thr),
                    "p_value": float(p_thr),
                    "significant": bool(p_thr < 0.05),
                    "alternative": "GNN carbon < Threshold carbon (one-sided)",
                },
            }
        except ValueError as exc:
            # Wilcoxon fails if all differences are zero (perfect runs)
            return {"error": str(exc)}

    def win_rate(self) -> dict:
        """
        Fraction of runs where GNN total < baseline total.
        This is the valid replacement for the 'Consistency 100%' claim.
        """
        wins_vs_ospf = int(np.sum(self.gnn_totals < self.ospf_totals))
        wins_vs_thr  = int(np.sum(self.gnn_totals < self.threshold_totals))
        out = {
            "gnn_vs_ospf": {
                "wins": wins_vs_ospf,
                "total_runs": self.n_runs,
                "win_rate_pct": wins_vs_ospf / self.n_runs * 100,
            },
            "gnn_vs_threshold": {
                "wins": wins_vs_thr,
                "total_runs": self.n_runs,
                "win_rate_pct": wins_vs_thr / self.n_runs * 100,
            },
        }
        if self.linear_flow_carbon_totals is not None:
            w = int(np.sum(self.gnn_totals < self.linear_flow_carbon_totals))
            out["gnn_vs_linear_flow_carbon"] = {
                "wins": w, "total_runs": self.n_runs,
                "win_rate_pct": w / self.n_runs * 100,
            }
        if self.nash_game_totals is not None:
            w = int(np.sum(self.gnn_totals < self.nash_game_totals))
            out["gnn_vs_nash_game"] = {
                "wins": w, "total_runs": self.n_runs,
                "win_rate_pct": w / self.n_runs * 100,
            }
        return out

    def reduction_stats(self) -> dict:
        """Per-run reduction % (GNN vs OSPF), then mean +/- SD of that distribution."""
        reductions = (
            (self.ospf_totals - self.gnn_totals) / self.ospf_totals * 100
        )
        thr_reductions = (
            (self.ospf_totals - self.threshold_totals) / self.ospf_totals * 100
        )
        n = len(reductions)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        se = np.std(reductions, ddof=1) / np.sqrt(n)
        out = {
            "gnn_vs_ospf": {
                "per_run_pct":  reductions.tolist(),
                "mean_pct":     float(np.mean(reductions)),
                "sd_pct":       float(np.std(reductions, ddof=1)),
                "ci_lower":     float(np.mean(reductions) - t_crit * se),
                "ci_upper":     float(np.mean(reductions) + t_crit * se),
                "min_pct":      float(np.min(reductions)),
                "max_pct":      float(np.max(reductions)),
            },
            "threshold_vs_ospf": {
                "per_run_pct": thr_reductions.tolist(),
                "mean_pct":    float(np.mean(thr_reductions)),
                "sd_pct":      float(np.std(thr_reductions, ddof=1)),
            },
        }
        if self.linear_flow_carbon_totals is not None:
            r = (self.ospf_totals - self.linear_flow_carbon_totals) / self.ospf_totals * 100
            out["linear_flow_carbon_vs_ospf"] = {
                "per_run_pct": r.tolist(),
                "mean_pct":    float(np.mean(r)),
                "sd_pct":      float(np.std(r, ddof=1)),
            }
        if self.nash_game_totals is not None:
            r = (self.ospf_totals - self.nash_game_totals) / self.ospf_totals * 100
            out["nash_game_vs_ospf"] = {
                "per_run_pct": r.tolist(),
                "mean_pct":    float(np.mean(r)),
                "sd_pct":      float(np.std(r, ddof=1)),
            }
        return out

    def block_bootstrap_hourly_ci(
        self, seed_key: int = None, block_size: int = 4, n_boot: int = 2000
    ) -> dict:
        """
        Block bootstrap 95% CI for the mean hourly savings on a single run.

        Uses seed_key (default: first seed) to pick the run.  Accounts for
        autocorrelation in the hourly series by resampling contiguous blocks.
        """
        s = seed_key if seed_key is not None else self.seeds[0]
        run = self.results[s]
        gnn_h = np.array(run["gnn"]["carbon_history"])
        ospf_h = np.array(run["ospf"]["carbon_history"])
        min_len = min(len(gnn_h), len(ospf_h))
        diffs = ospf_h[:min_len] - gnn_h[:min_len]  # positive = GNN saved

        rng = np.random.default_rng(42)
        lower, upper = _block_bootstrap_ci(
            diffs, block_size=block_size, n_boot=n_boot, rng=rng
        )
        return {
            "seed_used": s,
            "n_hours": min_len,
            "block_size_h": block_size,
            "n_bootstrap": n_boot,
            "mean_hourly_saving_gCO2": float(np.mean(diffs)),
            "ci_lower_95": lower,
            "ci_upper_95": upper,
            "note": (
                "Block bootstrap respects autocorrelation in the hourly series. "
                "For the definitive CI, prefer the between-run CI from between_run_stats()."
            ),
        }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        """Return a human-readable multi-seed statistical report (ASCII-safe)."""
        br = self.between_run_stats()
        eff = self.cohens_d_between_runs()
        tt = self.paired_ttest()
        wx = self.wilcoxon_test()
        wr = self.win_rate()
        rd = self.reduction_stats()

        lines = []
        SEP = "=" * 75
        sep = "-" * 75

        lines.append(SEP)
        lines.append("  MULTI-SEED STATISTICAL ANALYSIS")
        dur_str = f"{self.duration_hours} h" if self.duration_hours else "? h"
        lines.append(f"  N = {self.n_runs} independent {dur_str} runs  |  seeds: {self.seeds}")
        lines.append(SEP)

        # -- A. Per-run totals -------------------------------------------------
        lines.append("\nA. PER-RUN TOTAL CARBON (gCO2)  -- independent samples")
        lines.append(sep)
        hdr = f"{'Method':<18} {'Mean':>14} {'SD':>12} {'95% CI':>28} {'CV%':>7}"
        lines.append(hdr)
        lines.append(sep)
        for label, key in [
            ("GNN (Ours)", "gnn"),
            ("Threshold", "threshold"),
            ("LinearFlowCarbon", "linear_flow_carbon"),
            ("NashGame", "nash_game"),
            ("OSPF Baseline", "ospf"),
        ]:
            if key not in br: continue
            s = br[key]
            ci_str = f"[{s['ci_lower']:>10.1f}, {s['ci_upper']:>10.1f}]"
            lines.append(
                f"{label:<18} {s['mean']:>14.2f} {s['sd']:>12.2f}"
                f" {ci_str:>28} {s['cv_pct']:>6.1f}%"
            )

        if self.mlp_totals is not None:
            mlp_mean = np.mean(self.mlp_totals)
            mlp_sd = np.std(self.mlp_totals, ddof=1)
            lines.append(
                f"{'MLP Validator':<18} {mlp_mean:>14.2f} {mlp_sd:>12.2f}"
                f" {'N/A':>28} {(mlp_sd/mlp_mean*100):>6.1f}%"
            )
            # Calculate mean absolute error between GNN and MLP
            mae = np.mean(np.abs(self.gnn_totals - self.mlp_totals))
            mape = np.mean(np.abs(self.gnn_totals - self.mlp_totals) / self.gnn_totals) * 100
            lines.append(f"\n  MLP Validation MAE: {mae:.2f} gCO2 (MAPE: {mape:.2f}%)")


        # -- B. Reduction ------------------------------------------------------
        lines.append("\nB. CARBON REDUCTION  (GNN vs OSPF, per-run %)")
        lines.append(sep)
        r = rd["gnn_vs_ospf"]
        lines.append(f"  Mean reduction:  {r['mean_pct']:.2f}%  +/-  {r['sd_pct']:.2f}% SD")
        lines.append(f"  95% CI:          [{r['ci_lower']:.2f}%, {r['ci_upper']:.2f}%]")
        lines.append(f"  Range:           [{r['min_pct']:.2f}%, {r['max_pct']:.2f}%]")
        r2 = rd["threshold_vs_ospf"]
        lines.append(f"  Threshold vs OSPF: {r2['mean_pct']:.2f}%  +/-  {r2['sd_pct']:.2f}% SD")
        if "linear_flow_carbon_vs_ospf" in rd:
            r3 = rd["linear_flow_carbon_vs_ospf"]
            lines.append(f"  LinearFlow vs OSPF: {r3['mean_pct']:.2f}%  +/-  {r3['sd_pct']:.2f}% SD")
        if "nash_game_vs_ospf" in rd:
            r4 = rd["nash_game_vs_ospf"]
            lines.append(f"  NashGame vs OSPF: {r4['mean_pct']:.2f}%  +/-  {r4['sd_pct']:.2f}% SD")

        # -- C. Hypothesis tests -----------------------------------------------
        lines.append("\nC. HYPOTHESIS TESTS  (paired, same seeds -> same conditions)")
        lines.append(sep)

        gnn_ospf_t = tt["gnn_vs_ospf"]
        lines.append("  Paired t-test (GNN vs OSPF):")
        lines.append(
            f"    t({gnn_ospf_t['df']}) = {gnn_ospf_t['t_statistic']:.4f},  "
            f"p = {gnn_ospf_t['p_value']:.4e}  "
            f"({'significant' if gnn_ospf_t['significant'] else 'NOT significant'}, a=0.05)"
        )

        gnn_thr_t = tt["gnn_vs_threshold"]
        lines.append("  Paired t-test (GNN vs Threshold):")
        lines.append(
            f"    t({gnn_thr_t['df']}) = {gnn_thr_t['t_statistic']:.4f},  "
            f"p = {gnn_thr_t['p_value']:.4e}  "
            f"({'significant' if gnn_thr_t['significant'] else 'NOT significant'}, a=0.05)"
        )
        if "gnn_vs_linear_flow_carbon" in tt:
            t3 = tt["gnn_vs_linear_flow_carbon"]
            lines.append("  Paired t-test (GNN vs LinearFlowCarbon):")
            lines.append(
                f"    t({t3['df']}) = {t3['t_statistic']:.4f},  "
                f"p = {t3['p_value']:.4e}  "
                f"({'significant' if t3['significant'] else 'NOT significant'}, a=0.05)"
            )
        if "gnn_vs_nash_game" in tt:
            t4 = tt["gnn_vs_nash_game"]
            lines.append("  Paired t-test (GNN vs NashGame):")
            lines.append(
                f"    t({t4['df']}) = {t4['t_statistic']:.4f},  "
                f"p = {t4['p_value']:.4e}  "
                f"({'significant' if t4['significant'] else 'NOT significant'}, a=0.05)"
            )

        if "error" not in wx:
            go = wx["gnn_vs_ospf"]
            lines.append("  Wilcoxon signed-rank (GNN vs OSPF, one-sided):")
            lines.append(
                f"    W = {go['statistic']:.1f},  p = {go['p_value']:.4e}  "
                f"({'significant' if go['significant'] else 'NOT significant'}, a=0.05)"
            )
        else:
            lines.append(f"  Wilcoxon: {wx['error']}")

        # -- D. Effect sizes ---------------------------------------------------
        lines.append(
            "\nD. EFFECT SIZES  "
            "(between-run Cohen's d, Hedges pooled-SD, Sawilowsky 2009)"
        )
        lines.append(sep)
        keys_eff = [
            ("gnn_vs_ospf", "GNN vs OSPF"),
            ("gnn_vs_threshold", "GNN vs Threshold"),
            ("threshold_vs_ospf", "Threshold vs OSPF"),
        ]
        if "gnn_vs_linear_flow_carbon" in eff:
            keys_eff.append(("gnn_vs_linear_flow_carbon", "GNN vs LinearFlow"))
        if "gnn_vs_nash_game" in eff:
            keys_eff.append(("gnn_vs_nash_game", "GNN vs NashGame"))
        for key, label in keys_eff:
            e = eff[key]
            lines.append(f"  {label:<28}  d = {e['d']:+.4f}  ({e['label']})")
        lines.append(
            "  [All d values are based on between-run totals -- genuinely i.i.d.]"
        )

        # -- E. Win rate -------------------------------------------------------
        lines.append(
            "\nE. WIN RATE  "
            "(replaces 'Consistency 100%' from single run)"
        )
        lines.append(sep)
        wo = wr["gnn_vs_ospf"]
        wt = wr["gnn_vs_threshold"]
        lines.append(f"  GNN beat OSPF:       {wo['wins']}/{wo['total_runs']} runs  "
                     f"({wo['win_rate_pct']:.1f}%)")
        lines.append(f"  GNN beat Threshold:  {wt['wins']}/{wt['total_runs']} runs  "
                     f"({wt['win_rate_pct']:.1f}%)")
        if "gnn_vs_linear_flow_carbon" in wr:
            wl = wr["gnn_vs_linear_flow_carbon"]
            lines.append(f"  GNN beat LinearFlow: {wl['wins']}/{wl['total_runs']} runs  ({wl['win_rate_pct']:.1f}%)")
        if "gnn_vs_nash_game" in wr:
            wn = wr["gnn_vs_nash_game"]
            lines.append(f"  GNN beat NashGame:   {wn['wins']}/{wn['total_runs']} runs  ({wn['win_rate_pct']:.1f}%)")

        lines.append("\n" + SEP)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    def save_plots(self, output_dir: str = "results") -> list:
        """
        Generate and save publication-quality plots to output_dir.
        Returns list of saved file paths.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.family'] = 'DejaVu Sans'
            from matplotlib.patches import Patch
        except ImportError:
            print("matplotlib not available â€” skipping plots")
            return []

        os.makedirs(output_dir, exist_ok=True)
        saved = []

        # â”€â”€ Color palette â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        BG    = "#ffffff"
        CARD  = "#ffffff"
        BORDER= "#d4d4d4"
        TXT   = "#1a1a1a"
        TXT2  = "#525252"
        TXT_M = "#a3a3a3"
        GRN   = "#16a34a"
        RED   = "#dc2626"
        AMB   = "#d97706"
        BLU   = "#2563eb"
        PUR   = "#9333ea"

        rd = self.reduction_stats()
        br = self.between_run_stats()
        wr = self.win_rate()
        eff = self.cohens_d_between_runs()
        tt = self.paired_ttest()

        # ----------------------------------------------------------
        # Plot 1: Box plot of per-run total carbon (all 3 methods)
        # ----------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
        ax.set_facecolor(CARD)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(BORDER)
        ax.spines["left"].set_color(BORDER)
        ax.tick_params(colors=TXT_M, labelsize=9)

        data_groups = [self.gnn_totals, self.threshold_totals]
        labels_box  = ["GNN\n(Ours)", "Threshold\n(Top-25%)"]
        colors_box  = [GRN, AMB]
        
        if self.linear_flow_carbon_totals is not None:
            data_groups.append(self.linear_flow_carbon_totals)
            labels_box.append("LinearFlow\n(El-Zahr)")
            colors_box.append(BLU)
        if self.nash_game_totals is not None:
            data_groups.append(self.nash_game_totals)
            labels_box.append("NashGame\n(Hogade)")
            colors_box.append(PUR)
            
        data_groups.append(self.ospf_totals)
        labels_box.append("OSPF\n(Baseline)")
        colors_box.append(RED)

        bp = ax.boxplot(
            data_groups, patch_artist=True, widths=0.4,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=BORDER, linewidth=1.2),
            capprops=dict(color=BORDER, linewidth=1.2),
            flierprops=dict(marker="o", markerfacecolor=TXT_M, markersize=5, alpha=0.5),
        )
        for patch, col in zip(bp["boxes"], colors_box):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)

        # Overlay individual seed points
        for i, (arr, col) in enumerate(zip(data_groups, colors_box), start=1):
            jitter = np.random.default_rng(i).uniform(-0.08, 0.08, len(arr))
            ax.scatter(
                np.full_like(arr, i) + jitter, arr,
                color=col, s=40, alpha=0.9, zorder=5, edgecolors="white", linewidths=0.5,
            )

        ax.set_xticklabels(labels_box, fontsize=10, color=TXT)
        ax.set_ylabel("Total Carbon per Run (gCO2)", fontsize=10, color=TXT2)
        ax.set_title(
            f"Per-Run Total Carbon â€” {self.n_runs} Independent Seeds\n"
            f"GNN vs OSPF: d = {eff['gnn_vs_ospf']['d']:.2f} ({eff['gnn_vs_ospf']['label']}), "
            f"p = {tt['gnn_vs_ospf']['p_value']:.3e} (paired t-test)",
            fontsize=11, color=TXT, pad=10, loc="left",
        )
        ax.grid(axis="y", alpha=0.15, color=BORDER)

        legend_patches = [
            Patch(facecolor=c, alpha=0.75, label=l)
            for c, l in zip(colors_box, [l.replace("\n", " ") for l in labels_box])
        ]
        ax.legend(handles=legend_patches, fontsize=8, framealpha=0.9,
                  edgecolor=BORDER, facecolor=CARD, labelcolor=TXT2)

        plt.tight_layout()
        p = os.path.join(output_dir, "ms_01_boxplot_total_carbon.png")
        plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
        saved.append(p)

        # ----------------------------------------------------------
        # Plot 2: Error-bar chart of mean Â± 95% CI per method
        # ----------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
        ax.set_facecolor(CARD)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(BORDER)
        ax.spines["left"].set_color(BORDER)
        ax.tick_params(colors=TXT_M, labelsize=9)

        methods_eb  = ["GNN\n(Ours)", "Threshold\n(Top-25%)"]
        keys_eb     = ["gnn", "threshold"]
        colors_eb   = [GRN, AMB]
        
        if self.linear_flow_carbon_totals is not None:
            methods_eb.append("LinearFlow\n(El-Zahr)")
            keys_eb.append("linear_flow_carbon")
            colors_eb.append(BLU)
        if self.nash_game_totals is not None:
            methods_eb.append("NashGame\n(Hogade)")
            keys_eb.append("nash_game")
            colors_eb.append(PUR)
            
        methods_eb.append("OSPF\n(Baseline)")
        keys_eb.append("ospf")
        colors_eb.append(RED)
        means_eb    = [br[k]["mean"] for k in keys_eb]
        errors_low  = [br[k]["mean"] - br[k]["ci_lower"] for k in keys_eb]
        errors_high = [br[k]["ci_upper"] - br[k]["mean"] for k in keys_eb]

        bars = ax.bar(methods_eb, means_eb, color=colors_eb, alpha=0.72, width=0.45, edgecolor="none")
        ax.errorbar(
            range(len(methods_eb)), means_eb,
            yerr=[errors_low, errors_high],
            fmt="none", color=TXT, capsize=6, capthick=1.5, linewidth=1.5, zorder=5,
        )

        for bar, val in zip(bars, means_eb):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errors_high) * 0.05,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=9, color=TXT,
            )

        ax.set_ylabel("Mean Total Carbon per Run (gCO2)", fontsize=10, color=TXT2)
        ax.set_title(
            f"Mean Â± 95% CI  ({self.n_runs} runs)\n"
            f"GNN reduction: {rd['gnn_vs_ospf']['mean_pct']:.1f}% "
            f"[{rd['gnn_vs_ospf']['ci_lower']:.1f}%, {rd['gnn_vs_ospf']['ci_upper']:.1f}%]",
            fontsize=11, fontweight="bold", color=TXT, pad=10, loc="left",
        )
        ax.grid(axis="y", alpha=0.15, color=BORDER)

        plt.tight_layout()
        p = os.path.join(output_dir, "ms_02_errorbar_mean_ci.png")
        plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
        saved.append(p)

        # ----------------------------------------------------------
        # Plot 3: Per-seed reduction % (strip chart + mean line)
        # ----------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
        ax.set_facecolor(CARD)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(BORDER)
        ax.spines["left"].set_color(BORDER)
        ax.tick_params(colors=TXT_M, labelsize=8)

        gnn_reductions  = rd["gnn_vs_ospf"]["per_run_pct"]
        thr_reductions  = rd["threshold_vs_ospf"]["per_run_pct"]
        seed_labels = [f"Seed {s}" for s in self.seeds]
        x = np.arange(len(self.seeds))

        ax.bar(x - 0.2, gnn_reductions, width=0.35, color=GRN, alpha=0.75, label="GNN vs OSPF")
        ax.bar(x + 0.2, thr_reductions, width=0.35, color=AMB, alpha=0.75, label="Threshold vs OSPF")
        ax.axhline(
            rd["gnn_vs_ospf"]["mean_pct"], color=GRN, linestyle="--",
            linewidth=1.4, label=f"GNN mean ({rd['gnn_vs_ospf']['mean_pct']:.1f}%)",
        )
        ax.axhline(
            rd["threshold_vs_ospf"]["mean_pct"], color=AMB, linestyle=":",
            linewidth=1.4, label=f"Threshold mean ({rd['threshold_vs_ospf']['mean_pct']:.1f}%)",
        )
        ax.axhline(y=0, color=TXT_M, linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(seed_labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Carbon Reduction vs OSPF (%)", fontsize=10, color=TXT2)
        ax.set_title(
            f"Per-Run Reduction  ({self.n_runs} independent seeds)\n"
            f"GNN beats OSPF: {wr['gnn_vs_ospf']['wins']}/{self.n_runs} runs "
            f"({wr['gnn_vs_ospf']['win_rate_pct']:.0f}%)",
            fontsize=11, fontweight="bold", color=TXT, pad=10, loc="left",
        )
        ax.legend(fontsize=8, framealpha=0.9, edgecolor=BORDER, facecolor=CARD, labelcolor=TXT2)
        ax.grid(axis="y", alpha=0.15, color=BORDER)

        plt.tight_layout()
        p = os.path.join(output_dir, "ms_03_per_seed_reduction.png")
        plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
        saved.append(p)

        # ----------------------------------------------------------
        # Plot 4: Statistical summary panel (text figure for paper)
        # ----------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
        ax.set_facecolor(CARD)
        ax.axis("off")

        tt_go = tt["gnn_vs_ospf"]
        wx = self.wilcoxon_test()
        wr_go = wr["gnn_vs_ospf"]
        eff_go = eff["gnn_vs_ospf"]
        red = rd["gnn_vs_ospf"]

        rows = [
            ("MULTI-SEED STATISTICAL SUMMARY", ""),
            ("", ""),
            ("Runs (N)", f"{self.n_runs} independent seeds"),
            ("Simulation", (
                f"{self.duration_hours} h / run" if self.duration_hours else "? h / run"
            ) + (
                f", {self.num_nodes} nodes (base)" if self.num_nodes else ""
            )),
            ("Design", "Paired (same seed â†’ same conditions)"),
            ("", ""),
            ("GNN MEAN TOTAL CARBON", f"{br['gnn']['mean']:,.1f} Â± {br['gnn']['sd']:,.1f} gCO2"),
            ("OSPF MEAN TOTAL CARBON", f"{br['ospf']['mean']:,.1f} Â± {br['ospf']['sd']:,.1f} gCO2"),
            ("", ""),
            ("Mean Reduction", f"{red['mean_pct']:.2f}%  Â±  {red['sd_pct']:.2f}% SD"),
            ("95% CI (reduction)", f"[{red['ci_lower']:.2f}%, {red['ci_upper']:.2f}%]"),
            ("Win Rate (GNN < OSPF)", f"{wr_go['wins']}/{self.n_runs}  ({wr_go['win_rate_pct']:.0f}%)"),
            ("", ""),
            ("Cohen's d (between-run)", f"{eff_go['d']:+.4f}  [{eff_go['label']}]"),
            ("Paired t-test",
             f"t({tt_go['df']}) = {tt_go['t_statistic']:.3f},  p = {tt_go['p_value']:.3e}"),
        ]
        if "error" not in wx:
            rows.append(
                ("Wilcoxon signed-rank",
                 f"W = {wx['gnn_vs_ospf']['statistic']:.1f},  p = {wx['gnn_vs_ospf']['p_value']:.3e}")
            )

        y = 0.96
        for label, value in rows:
            if label in ("MULTI-SEED STATISTICAL SUMMARY",):
                ax.text(0.04, y, label, fontsize=11,
                        color=TXT, transform=ax.transAxes)
            elif label == "":
                pass
            elif label.isupper():
                ax.text(0.04, y, label, fontsize=8,
                        color=TXT_M, transform=ax.transAxes)
                ax.text(0.96, y, value, fontsize=9,
                        color=GRN, transform=ax.transAxes, ha="right")
            else:
                ax.text(0.04, y, label, fontsize=9, color=TXT2, transform=ax.transAxes)
                ax.text(0.96, y, value, fontsize=9,
                        color=TXT, transform=ax.transAxes, ha="right")
            y -= 0.052

        plt.tight_layout()
        p = os.path.join(output_dir, "ms_04_stats_summary_panel.png")
        plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
        saved.append(p)

        return saved

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return all computed stats as a JSON-serializable dict."""
        return {
            "n_runs": self.n_runs,
            "seeds": self.seeds,
            "between_run_stats": self.between_run_stats(),
            "cohens_d": self.cohens_d_between_runs(),
            "paired_ttest": self.paired_ttest(),
            "wilcoxon": self.wilcoxon_test(),
            "win_rate": self.win_rate(),
            "reduction_stats": self.reduction_stats(),
        }

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Multi-seed stats saved: {path}")
