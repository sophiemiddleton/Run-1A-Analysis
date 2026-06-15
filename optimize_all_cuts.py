#!/usr/bin/env python3
"""Multi-background cut optimization interface - Joint simultaneous optimization.

This module provides joint optimization of all cuts simultaneously. Instead of
optimizing each cut independently, all cuts vary together in each epoch to find
the best combination of thresholds that maximizes signal efficiency while
minimizing all backgrounds simultaneously.

The workflow is:
  1. process.py loads data via AnaProcessor (handles branches correctly)
  2. process.py calls optimize_from_loaded_data() with loaded data_dict
  3. This module performs joint multi-dimensional scan over all cuts
  4. Returns best threshold combination and composite score

Usage from process.py:
    from optimize_all_cuts import optimize_from_loaded_data
    
    data_dict = {
        'signal': loaded_signal_data,
        'RPC': loaded_rpc_data,
        'Cosmics': loaded_cosmics_data,
        'DIO': loaded_dio_data
    }
    
    cut_definitions = {
        'd0': {'path': 'trkfit.trksegpars_lh.d0', 'direction': 'less'},
        'trkqual': {'path': 'trk.trkqual.result', 'direction': 'greater'},
    }
    
    result = optimize_from_loaded_data(
        data_dict=data_dict,
        cut_definitions=cut_definitions,
        output_dir='./cut_optimization',
        n_steps=30,  # 30 points per cut dimension → 30^N total combinations
        weights={'signal': 1.0, 'RPC': 0.3, 'Cosmics': 0.3, 'DIO': 0.6}
    )
    
    # All cuts optimized together - result contains best_thresholds for all cuts
"""

import json
import os

import numpy as np
import awkward as ak

from optimize_cuts import (
    optimize_on_combine_result_multi_bkg,
    print_scan_summary_multi_bkg,
    save_csv_multi_bkg,
)


def extract_feature_values(data, feature_path):
    """Extract nested feature path from awkward data and flatten to 1D numpy array.
    
    Handles:
    - Top-level fields: "nST", "nOPA"
    - Nested structures: "trkfit.trksegpars_lh.d0"
    - Field names with dots: "trk.trkqual.result" where "trkqual.result" is a single field
    """
    parts = feature_path.split(".")
    obj = data
    
    # Try to access as top-level field first (e.g., "nST", "nOPA")
    if len(parts) == 1:
        try:
            obj = data[feature_path]
            flat = ak.flatten(obj, axis=None)
            return np.asarray(flat)
        except (KeyError, TypeError, IndexError):
            pass
    
    # Try to navigate the path, handling field names with dots
    i = 0
    while i < len(parts):
        try:
            obj = obj[parts[i]]
            i += 1
        except (KeyError, TypeError, IndexError):
            # If simple access fails and there are more parts, try combining
            # remaining parts with dots (for field names like "trkqual.result")
            if i < len(parts) - 1:
                combined = ".".join(parts[i:])
                try:
                    obj = obj[combined]
                    break
                except (KeyError, TypeError, IndexError):
                    raise ValueError(f"Cannot find field '{combined}' in data structure")
            else:
                raise ValueError(f"Cannot find field '{parts[i]}' in data structure")
    
    flat = ak.flatten(obj, axis=None)
    return np.asarray(flat)


def save_final_cuts(results_dict, output_dir):
    """Save best thresholds from each cut scan into final_cuts.json."""
    final_cuts = {}
    for cut_name, results in results_dict.items():
        if not results:
            continue
        best = results[0]
        final_cuts[cut_name] = {
            "threshold": float(best.threshold),
            "sig_efficiency": float(best.sig_efficiency),
            "bkg_rejections": {
                name: float(best.bkg_rejection.get(name, 0.0))
                for name in best.bkg_rejection.keys()
            },
        }

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "final_cuts.json")
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(final_cuts, handle, indent=2)
    return output_file


def optimize_from_loaded_data(data_dict, cut_definitions, output_dir="./cut_optimization",
                             n_steps=50, weights=None, metric="weighted", verbosity=1):
    """
    Joint multi-dimensional optimization of all cuts simultaneously.
    
    All cuts vary together in each epoch - this finds the best combination of
    thresholds that maximizes signal efficiency while minimizing all backgrounds
    simultaneously.

    Args:
        data_dict (dict): {sample_name: awkward_array}
            Pre-loaded data from process.py for each sample (signal, RPC, Cosmics, DIO)
        cut_definitions (dict): {cut_name: {path, direction, description}}
            Feature paths and optimization direction for each cut
        output_dir (str): Directory to save results
        n_steps (int): Number of threshold scan points per cut dimension
        weights (dict): Sample weights for composite metric
            Default: {'signal': 1.0, 'RPC': 0.3, 'Cosmics': 0.3, 'DIO': 0.6}
        metric (str): 'weighted', 'youden', or 's_over_sqrtb'
        verbosity (int): Verbosity level

    Returns:
        dict: Best thresholds and composite score
    """
    if weights is None:
        weights = {
            "signal": 1.0,
            "RPC": 0.3,
            "Cosmics": 0.3,
            "DIO": 0.6,
        }

    os.makedirs(output_dir, exist_ok=True)

    if verbosity >= 1:
        print("\n" + "=" * 80)
        print("JOINT MULTI-BACKGROUND CUT OPTIMIZATION")
        print("=" * 80)
        print(f"\nSimultaneously optimizing {len(cut_definitions)} cuts across {len(data_dict)} samples")
        for sample_name, data in data_dict.items():
            print(f"  {sample_name:12s}: {len(data)} events")

    # Step 1: Extract all feature arrays
    if verbosity >= 1:
        print(f"\nExtracting feature arrays...")
    
    feature_arrays = {}  # {cut_name: {sample_name: array}}
    for cut_name, cut_def in cut_definitions.items():
        feature_arrays[cut_name] = {}
        for sample_name, sample_data in data_dict.items():
            try:
                feature_arrays[cut_name][sample_name] = extract_feature_values(
                    sample_data, cut_def["path"]
                )
                if verbosity >= 2:
                    print(f"  {cut_name:12s} × {sample_name:12s}: {len(feature_arrays[cut_name][sample_name])} values")
            except Exception as e:
                if verbosity >= 1:
                    print(f"  {cut_name:12s} × {sample_name:12s}: ERROR - {e}")
                return None

    # Step 2: Generate threshold ranges for each cut
    if verbosity >= 1:
        print(f"\nGenerating threshold ranges...")
    
    threshold_ranges = {}
    is_integer_cut = {}  # Track which cuts are integer-valued
    for cut_name, cut_def in cut_definitions.items():
        # Collect all values across samples
        all_vals = np.concatenate([feature_arrays[cut_name][s] for s in data_dict.keys()])
        all_vals = all_vals[~np.isnan(all_vals)]
        
        # Check if this is an integer-valued feature
        # (e.g., NST, NOPA - counts should be integers)
        is_int = np.allclose(all_vals, np.round(all_vals))
        is_integer_cut[cut_name] = is_int
        
        direction = cut_def["direction"]
        
        if is_int:
            # For integer-valued cuts, generate integer thresholds
            min_val = int(np.min(all_vals))
            max_val = int(np.max(all_vals))
            # Create integer thresholds from min to max
            n_unique = max_val - min_val + 1
            if n_unique <= n_steps:
                thresholds = np.arange(min_val, max_val + 1, dtype=float)
            else:
                # If too many unique values, sample evenly
                thresholds = np.round(np.linspace(min_val, max_val, n_steps)).astype(int).astype(float)
            threshold_ranges[cut_name] = np.unique(thresholds)
        elif "pz_pt_lower" in cut_name:
            # For pz/pt lower bound, scan around 0.5 (default)
            # Scan from 0.3 to 0.7
            thresholds = np.linspace(0.3, 0.7, n_steps)
            threshold_ranges[cut_name] = thresholds
        elif "pz_pt_upper" in cut_name:
            # For pz/pt upper bound, scan around 1.0 (default)
            # Scan from 0.8 to 1.2
            thresholds = np.linspace(0.8, 1.2, n_steps)
            threshold_ranges[cut_name] = thresholds
        else:
            # For continuous-valued cuts, use linspace over data range
            thresholds = np.linspace(np.min(all_vals), np.max(all_vals), n_steps)
            threshold_ranges[cut_name] = thresholds
        
        if verbosity >= 2:
            thresh_str = "integer" if is_int else "continuous"
            print(f"  {cut_name:12s}: {len(threshold_ranges[cut_name])} {thresh_str} thresholds [{threshold_ranges[cut_name][0]:.6g}, {threshold_ranges[cut_name][-1]:.6g}]")

    # Step 3: Joint scan - try all combinations
    if verbosity >= 1:
        print(f"\nJoint scanning all cut combinations...")
        total_combos = np.prod([len(threshold_ranges[c]) for c in cut_definitions.keys()])
        print(f"  Total combinations to evaluate: {int(total_combos):,}")

    from itertools import product
    
    best_score = -np.inf
    best_thresholds = {}
    best_efficiencies = {}
    all_results = []
    
    combo_idx = 0
    for threshold_combo in product(*[threshold_ranges[c] for c in cut_definitions.keys()]):
        combo_idx += 1
        
        # Build threshold dict for this combination
        thresholds = {cut_name: thresh for cut_name, thresh in zip(cut_definitions.keys(), threshold_combo)}
        
        # Apply all cuts simultaneously to each sample
        efficiencies = {}
        pass_counts = {}
        total_counts = {}
        
        for sample_name in data_dict.keys():
            mask = np.ones(len(feature_arrays[list(cut_definitions.keys())[0]][sample_name]), dtype=bool)
            
            # Apply each cut with its threshold
            for cut_name, cut_def in cut_definitions.items():
                values = feature_arrays[cut_name][sample_name]
                threshold = thresholds[cut_name]
                direction = cut_def["direction"]
                
                if direction == "less":
                    cut_mask = values < threshold
                else:  # greater
                    cut_mask = values > threshold
                
                mask = mask & cut_mask
            
            pass_counts[sample_name] = np.sum(mask)
            total_counts[sample_name] = len(mask)
            efficiencies[sample_name] = pass_counts[sample_name] / total_counts[sample_name] if total_counts[sample_name] > 0 else 0.0
        
        # Calculate composite score
        sig_eff = efficiencies.get("signal", 0.0)
        
        if metric == "weighted":
            score = weights.get("signal", 1.0) * sig_eff
            for bkg_name, bkg_weight in weights.items():
                if bkg_name != "signal" and bkg_name in efficiencies:
                    bkg_eff = efficiencies[bkg_name]
                    bkg_rejection = 1.0 - bkg_eff  # rejection = 1 - efficiency
                    score -= bkg_weight * bkg_eff
        
        all_results.append({
            "thresholds": thresholds.copy(),
            "efficiencies": efficiencies.copy(),
            "pass_counts": pass_counts.copy(),
            "total_counts": total_counts.copy(),
            "score": score
        })
        
        if score > best_score:
            best_score = score
            best_thresholds = thresholds.copy()
            best_efficiencies = efficiencies.copy()
        
        if verbosity >= 2 and combo_idx % max(1, int(total_combos / 10)) == 0:
            pct = 100.0 * combo_idx / total_combos
            print(f"  [{pct:.0f}%] Best score so far: {best_score:.6g}")
    
    # Sort results by score
    all_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, "joint_optimization_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        # Header
        header_parts = []
        header_parts.extend([f"threshold_{c}" for c in cut_definitions.keys()])
        header_parts.extend([f"sig_eff", "score"])
        for bkg_name in data_dict.keys():
            if bkg_name != "signal":
                header_parts.append(f"{bkg_name}_eff")
        f.write(",".join(header_parts) + "\n")
        
        # Rows (top 100 results)
        for result in all_results[:100]:
            row_parts = []
            for cut_name in cut_definitions.keys():
                row_parts.append(str(result["thresholds"][cut_name]))
            row_parts.append(str(result["efficiencies"].get("signal", 0.0)))
            row_parts.append(str(result["score"]))
            for bkg_name in data_dict.keys():
                if bkg_name != "signal":
                    row_parts.append(str(result["efficiencies"].get(bkg_name, 0.0)))
            f.write(",".join(row_parts) + "\n")
    
    # Save best result with rounded thresholds for integer-valued cuts
    best_thresholds_rounded = {}
    for cut_name, threshold in best_thresholds.items():
        if is_integer_cut.get(cut_name, False):
            best_thresholds_rounded[cut_name] = int(round(threshold))
        else:
            best_thresholds_rounded[cut_name] = threshold
    
    best_result = {
        "best_thresholds": best_thresholds_rounded,
        "best_efficiencies": best_efficiencies,
        "best_score": float(best_score),
        "metric": metric,
        "weights": weights
    }
    
    final_json = os.path.join(output_dir, "joint_optimization_best.json")
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2)
    
    if verbosity >= 1:
        print("\n" + "-" * 80)
        print("BEST JOINT OPTIMIZATION RESULT")
        print("-" * 80)
        print(f"\nBest composite score: {best_score:.6g}\n")
        print("Optimal thresholds:")
        for cut_name, threshold in best_thresholds_rounded.items():
            if is_integer_cut.get(cut_name, False):
                print(f"  {cut_name:12s}: {threshold} (integer)")
            else:
                print(f"  {cut_name:12s}: {threshold:.6g}")
        print("\nEfficiencies at optimal thresholds:")
        for sample_name, eff in best_efficiencies.items():
            print(f"  {sample_name:12s}: {eff:.1%}")
        print("\n" + "=" * 80)
        print(f"Results saved to: {final_json}")
        print(f"Top 100 results saved to: {csv_path}")
        print("=" * 80 + "\n")
    
    # Generate visualizations
    plot_best_result(best_result, output_dir)
    plot_optimization_scores(all_results, output_dir)
    plot_signal_vs_background_eff(all_results, output_dir)
    if len(cut_definitions) >= 2:
        plot_threshold_pairs(all_results, cut_definitions, output_dir)
    
    return best_result


def plot_signal_vs_background_eff(all_results, output_dir):
    """Plot signal efficiency vs background efficiency (ROC-like curve)."""
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        import matplotlib.pyplot as plt
    
    # Extract data from all results
    sig_effs = []
    bkg_effs = []
    scores = []
    
    for result in all_results:
        sig_eff = result['efficiencies'].get("signal", 0.0)
        
        # Average background efficiency across all backgrounds
        bkg_effs_list = []
        for sample_name, eff in result['efficiencies'].items():
            if sample_name != "signal":
                bkg_effs_list.append(eff)
        
        if bkg_effs_list:
            avg_bkg_eff = np.mean(bkg_effs_list)
            sig_effs.append(sig_eff)
            bkg_effs.append(avg_bkg_eff)
            scores.append(result['score'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot as scatter with colors representing score
    scatter = ax.scatter(sig_effs, bkg_effs, c=scores, s=100, cmap='RdYlGn', 
                        alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Mark the best point
    best_idx = np.argmax(scores)
    ax.scatter(sig_effs[best_idx], bkg_effs[best_idx], s=500, marker='*', 
              color='gold', edgecolors='darkred', linewidth=2, 
              label=f'Best Result\n(Sig: {sig_effs[best_idx]:.1%}, Bkg: {bkg_effs[best_idx]:.1%})',
              zorder=5)
    
    # Add ideal point marker
    ax.scatter(1.0, 0.0, s=200, marker='s', color='lime', edgecolors='darkgreen', 
              linewidth=2, label='Ideal Point\n(Sig: 100%, Bkg: 0%)', zorder=4, alpha=0.7)
    
    # Styling
    ax.set_xlabel('Signal Efficiency', fontsize=13, fontweight='bold')
    ax.set_ylabel('Background Efficiency (lower is better)', fontsize=13, fontweight='bold')
    ax.set_title('Signal vs Background Efficiency Trade-off', fontsize=14, fontweight='bold')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Composite Score', fontsize=11, fontweight='bold')
    
    # Add diagonal line showing equal efficiency point
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Signal = Background')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'signal_vs_background_efficiency.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_best_result(best_result, output_dir):
    """Plot best thresholds and their efficiencies."""
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        import matplotlib.pyplot as plt
    
    thresholds = best_result['best_thresholds']
    efficiencies = best_result['best_efficiencies']
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Thresholds
    cut_names = list(thresholds.keys())
    threshold_values = list(thresholds.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(cut_names)))
    
    ax1.bar(cut_names, threshold_values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Threshold Value', fontsize=12, fontweight='bold')
    ax1.set_title('Optimal Thresholds', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(threshold_values):
        ax1.text(i, v, f'{v:.3g}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Efficiencies
    samples = list(efficiencies.keys())
    eff_values = list(efficiencies.values())
    colors_eff = plt.cm.coolwarm(np.linspace(0, 1, len(samples)))
    
    bars = ax2.bar(samples, eff_values, color=colors_eff, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Efficiency', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.set_title('Sample Efficiencies at Optimal Thresholds', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(eff_values):
        ax2.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'best_result_summary.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_optimization_scores(all_results, output_dir, top_n=100):
    """Plot distribution of composite scores across all results."""
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        import matplotlib.pyplot as plt
    
    scores = [r['score'] for r in all_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Score distribution
    ax1.hist(scores, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.max(scores), color='red', linestyle='--', linewidth=2, label=f'Best: {np.max(scores):.3g}')
    ax1.set_xlabel('Composite Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Score Distribution Across All Combinations', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Top results
    top_scores = sorted(scores, reverse=True)[:top_n]
    ranks = np.arange(1, len(top_scores) + 1)
    ax2.plot(ranks, top_scores, 'o-', color='darkgreen', linewidth=2, markersize=4)
    ax2.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title(f'Top {top_n} Results', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'optimization_scores.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_threshold_pairs(all_results, cut_definitions, output_dir, top_n=1000):
    """Plot 2D pairs of threshold combinations colored by score."""
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        import matplotlib.pyplot as plt
    
    cut_names = list(cut_definitions.keys())
    if len(cut_names) < 2:
        return
    
    # Use top N results for clarity
    top_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:top_n]
    
    # Generate 2D plots for each pair of cuts
    n_pairs = len(cut_names) * (len(cut_names) - 1) // 2
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    for i, cut1 in enumerate(cut_names):
        for j, cut2 in enumerate(cut_names[i+1:], start=i+1):
            ax = axes[plot_idx]
            
            # Extract values
            cut1_vals = [r['thresholds'][cut1] for r in top_results]
            cut2_vals = [r['thresholds'][cut2] for r in top_results]
            scores = [r['score'] for r in top_results]
            
            # Scatter plot colored by score
            scatter = ax.scatter(cut1_vals, cut2_vals, c=scores, s=50, cmap='RdYlGn', 
                               alpha=0.6, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel(cut1, fontsize=11, fontweight='bold')
            ax.set_ylabel(cut2, fontsize=11, fontweight='bold')
            ax.set_title(f'{cut1} vs {cut2}', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Score', fontsize=10, fontweight='bold')
            
            plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'threshold_pairs_2d.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


