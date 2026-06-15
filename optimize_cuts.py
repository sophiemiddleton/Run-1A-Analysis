"""optimize_cuts.py

Threshold scanner to optimize single and multiple feature cuts for signal efficiency
vs background rejection. Supports single background (legacy) and multi-background
simultaneous optimization (new).
"""
import argparse
import numpy as np
import os
import csv
import matplotlib.pyplot as plt


# ============================================================================
# Multi-Background Optimization: Data Structure
# ============================================================================

class CutScanResult:
    """Enhanced result structure for tracking signal and multiple backgrounds"""
    
    def __init__(self, threshold):
        self.threshold = threshold
        self.sig_efficiency = None
        self.sig_pass = None
        self.sig_total = None
        
        # Per-background efficiencies: {bkg_name: count_passing}
        self.bkg_pass = {}
        self.bkg_total = {}
        self.bkg_efficiency = {}  # Derived: pass / total
        self.bkg_rejection = {}   # Derived: 1 - efficiency
        
        # Composite metrics
        self.weighted_score = None
        self.youden_vs_all = None
        self.s_over_sqrtb_all = None
        self.primary_metric = None
    
    def to_dict(self):
        """Convert to dict for CSV export"""
        d = {
            'threshold': self.threshold,
            'sig_efficiency': self.sig_efficiency,
            'sig_pass': self.sig_pass,
            'sig_total': self.sig_total,
        }
        # Add per-background metrics
        for bkg_name in sorted(self.bkg_pass.keys()):
            d[f'{bkg_name}_efficiency'] = self.bkg_efficiency.get(bkg_name, None)
            d[f'{bkg_name}_rejection'] = self.bkg_rejection.get(bkg_name, None)
            d[f'{bkg_name}_pass'] = self.bkg_pass.get(bkg_name, None)
            d[f'{bkg_name}_total'] = self.bkg_total.get(bkg_name, None)
        
        d['weighted_score'] = self.weighted_score
        d['youden_vs_all'] = self.youden_vs_all
        d['s_over_sqrtb_all'] = self.s_over_sqrtb_all
        d['primary_metric'] = self.primary_metric
        return d

def to_1d_numpy(x):
    """Convert common array/list/awkward inputs to a 1D numpy array."""
    # Numpy array
    if isinstance(x, np.ndarray):
        return x.ravel()

    # Awkward array
    try:
        import awkward as ak
        if isinstance(x, ak.Array):
            flat = ak.flatten(x, axis=None)
            return np.asarray(flat)
    except Exception:
        pass

    # Python list (possibly nested)
    if isinstance(x, (list, tuple)):
        parts = []
        for el in x:
            arr = to_1d_numpy(el)
            if arr is None:
                continue
            parts.append(arr)
        if parts:
            try:
                return np.concatenate(parts)
            except Exception:
                return np.array([])

    # Fallback: try to coerce
    try:
        return np.asarray(x).ravel()
    except Exception:
        return np.array([])


def load_feature(path, feature_key=None):
    """Load a named feature from .npz or .npy. If .npz and feature_key omitted,
    take the first array in the archive."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith('.npz'):
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())
        if feature_key is None:
            if not keys:
                raise KeyError(f"No arrays found in {path}")
            arr = data[keys[0]]
        else:
            if feature_key not in keys:
                raise KeyError(f"Feature '{feature_key}' not in {path}. Available: {keys}")
            arr = data[feature_key]
        return to_1d_numpy(arr)

    elif path.endswith('.npy'):
        arr = np.load(path, allow_pickle=True)
        return to_1d_numpy(arr)

    else:
        # Try to load as numpy anyway
        arr = np.load(path, allow_pickle=True)
        return to_1d_numpy(arr)


def scan_thresholds(sig_vals, bkg_vals, direction='greater', n_steps=200, metric='youden'):
    """Scan thresholds and return rows with threshold, tpr, bkg_rej, metric, nsig, nbkg."""
    sig = sig_vals[~np.isnan(sig_vals)] if sig_vals.size else np.array([])
    bkg = bkg_vals[~np.isnan(bkg_vals)] if bkg_vals.size else np.array([])

    if sig.size == 0 and bkg.size == 0:
        raise ValueError('Both signal and background arrays are empty')

    combined = np.concatenate([sig, bkg]) if sig.size and bkg.size else (sig if sig.size else bkg)
    lo = float(np.nanmin(combined))
    hi = float(np.nanmax(combined))
    if lo == hi:
        # degenerate range
        thresholds = np.array([lo])
    else:
        thresholds = np.linspace(lo, hi, n_steps)

    rows = []
    eps = 1e-12
    for thr in thresholds:
        if direction == 'greater':
            sig_pass = np.sum(sig >= thr) if sig.size else 0
            bkg_pass = np.sum(bkg >= thr) if bkg.size else 0
        else:
            sig_pass = np.sum(sig <= thr) if sig.size else 0
            bkg_pass = np.sum(bkg <= thr) if bkg.size else 0

        nsig = sig.size
        nbkg = bkg.size
        tpr = float(sig_pass) / nsig if nsig > 0 else 0.0
        fpr = float(bkg_pass) / nbkg if nbkg > 0 else 0.0
        bkg_rej = 1.0 - fpr

        if metric == 'youden':
            val = tpr + bkg_rej - 1.0  # tpr - fpr
        elif metric == 's_over_sqrtb':
            s = sig_pass
            b = bkg_pass
            val = float(s) / np.sqrt(b + eps) if (b > 0 or s > 0) else 0.0
        else:
            raise ValueError('Unknown metric')

        rows.append({'threshold': thr, 'tpr': tpr, 'bkg_rej': bkg_rej, 'metric': val, 'nsig_pass': int(sig_pass), 'nbkg_pass': int(bkg_pass)})

    return rows


# ============================================================================
# Multi-Background Optimization: Core Scanning
# ============================================================================

def scan_thresholds_multi_bkg(
    sig_vals, 
    bkg_dict,
    direction='greater', 
    n_steps=200, 
    metric='weighted',
    weights=None,
    target_rejections=None
):
    """
    Scan thresholds optimizing signal efficiency vs. multiple backgrounds.
    
    Args:
        sig_vals (np.ndarray): Signal feature values
        bkg_dict (dict): {background_name: np.ndarray of values, ...}
        direction (str): 'greater' or 'less'
        n_steps (int): Number of thresholds to scan
        metric (str): 'weighted', 'youden', or 's_over_sqrtb'
        weights (dict): {name: weight} for composite metric
                        (default: equal weights)
        target_rejections (dict): {bkg_name: target_rejection}
                                  (informational, not enforced in scan)
    
    Returns:
        list of CutScanResult objects, sorted by primary_metric (best first)
    """
    # Clean data
    sig = sig_vals[~np.isnan(sig_vals)] if sig_vals.size else np.array([])
    bkg_clean = {}
    for name, vals in bkg_dict.items():
        bkg = vals[~np.isnan(vals)] if vals.size else np.array([])
        bkg_clean[name] = bkg
    
    # Determine threshold range from combined data
    all_vals = [sig] + list(bkg_clean.values())
    all_vals = [x for x in all_vals if x.size > 0]
    
    if not all_vals:
        raise ValueError('All input arrays are empty')
    
    combined = np.concatenate(all_vals)
    lo = float(np.nanmin(combined))
    hi = float(np.nanmax(combined))
    
    if lo == hi:
        thresholds = np.array([lo])
    else:
        thresholds = np.linspace(lo, hi, n_steps)
    
    # Set default weights (equal)
    if weights is None:
        weights = {'signal': 1.0}
        for name in bkg_dict:
            weights[name] = 1.0
    
    # Scan
    results = []
    eps = 1e-12
    
    for thr in thresholds:
        result = CutScanResult(thr)
        
        # Signal
        if direction == 'greater':
            sig_pass = np.sum(sig >= thr) if sig.size else 0
        else:
            sig_pass = np.sum(sig <= thr) if sig.size else 0
        
        sig_total = sig.size
        result.sig_pass = int(sig_pass)
        result.sig_total = sig_total
        result.sig_efficiency = float(sig_pass) / sig_total if sig_total > 0 else 0.0
        
        # Per-background
        for bkg_name, bkg_vals in bkg_clean.items():
            if direction == 'greater':
                bkg_pass = np.sum(bkg_vals >= thr) if bkg_vals.size else 0
            else:
                bkg_pass = np.sum(bkg_vals <= thr) if bkg_vals.size else 0
            
            bkg_total = bkg_vals.size
            result.bkg_pass[bkg_name] = int(bkg_pass)
            result.bkg_total[bkg_name] = bkg_total
            result.bkg_efficiency[bkg_name] = float(bkg_pass) / bkg_total if bkg_total > 0 else 0.0
            result.bkg_rejection[bkg_name] = 1.0 - result.bkg_efficiency[bkg_name]
        
        # Compute composite metrics
        if metric == 'weighted':
            # Weighted sum: maximize signal, minimize backgrounds
            w_sig = weights.get('signal', 1.0)
            score = w_sig * result.sig_efficiency
            for bkg_name in bkg_dict:
                w_bkg = weights.get(bkg_name, 1.0)
                # Subtract background efficiency (want to reject backgrounds)
                score -= w_bkg * result.bkg_efficiency[bkg_name]
            result.weighted_score = score
            result.primary_metric = score
        
        elif metric == 'youden':
            # Average Youden index across all backgrounds
            if bkg_dict:
                youden_vals = [result.sig_efficiency - (1 - result.bkg_rejection[name]) 
                              for name in bkg_dict]
                result.youden_vs_all = float(np.mean(youden_vals))
                result.primary_metric = result.youden_vs_all
            else:
                result.youden_vs_all = 0.0
                result.primary_metric = 0.0
        
        elif metric == 's_over_sqrtb':
            # s / sqrt(sum of all backgrounds)
            total_bkg_pass = sum(result.bkg_pass.values())
            s = result.sig_pass
            result.s_over_sqrtb_all = float(s) / np.sqrt(total_bkg_pass + eps) if (total_bkg_pass > 0 or s > 0) else 0.0
            result.primary_metric = result.s_over_sqrtb_all
        
        else:
            raise ValueError(f'Unknown metric: {metric}')
        
        results.append(result)
    
    # Sort by primary metric (descending)
    results.sort(key=lambda r: r.primary_metric, reverse=True)
    return results


def find_best(rows):
    if not rows:
        return None
    best = max(rows, key=lambda r: r['metric'])
    return best


def save_csv(rows, outpath):
    if not rows:
        return
    keys = ['threshold', 'tpr', 'bkg_rej', 'metric', 'nsig_pass', 'nbkg_pass']
    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_scan(rows, outpath, show=False, xlim=None, ylim=None):
    """Plot signal efficiency (TPR) vs background rejection for a scan.

    Saves PNG to `outpath`. If `show` is True, calls `plt.show()`.
    """
    if not rows:
        raise ValueError('No scan rows to plot')

    thresholds = np.array([r['threshold'] for r in rows])
    tpr = np.array([r['tpr'] for r in rows])
    bkg_rej = np.array([r['bkg_rej'] for r in rows])
    metric = np.array([r['metric'] for r in rows])

    # best point
    best_idx = int(np.argmax(metric))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(bkg_rej, tpr, '-', lw=1, label='scan')
    ax.scatter(bkg_rej, tpr, c=metric, cmap='viridis', s=20)
    ax.scatter([bkg_rej[best_idx]], [tpr[best_idx]], color='red', s=60, label='best')

    txt = f"thr={thresholds[best_idx]:.4g}\nTPR={tpr[best_idx]:.3f}\nBkgRej={bkg_rej[best_idx]:.3f}\nmetric={metric[best_idx]:.3g}"
    ax.annotate(txt, xy=(bkg_rej[best_idx], tpr[best_idx]), xytext=(0.05, 0.95), textcoords='axes fraction',
                fontsize=9, va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.6))

    ax.set_xlabel('Background rejection (1 - FPR)')
    ax.set_ylabel('Signal efficiency (TPR)')
    # Auto-scale limits to data if not provided, with small padding
    if xlim is None:
        x_min = float(np.nanmin(bkg_rej))
        x_max = float(np.nanmax(bkg_rej))
        if x_min == x_max:
            x_min, x_max = 0.0, 1.0
        else:
            pad = max(0.05 * (x_max - x_min), 0.01)
            x_min = max(0.0, x_min - pad)
            x_max = min(1.0, x_max + pad)
        ax.set_xlim(x_min, x_max)
    else:
        ax.set_xlim(*xlim)

    if ylim is None:
        y_min = float(np.nanmin(tpr))
        y_max = float(np.nanmax(tpr))
        if y_min == y_max:
            y_min, y_max = 0.0, 1.0
        else:
            pad = max(0.05 * (y_max - y_min), 0.01)
            y_min = max(0.0, y_min - pad)
            y_max = min(1.0, y_max + pad)
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(*ylim)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    if show:
        plt.show()
    plt.close(fig)


def plot_scan_vs_value(rows, outpath, show=False, xlim=None):
    """Plot signal efficiency and background efficiency vs the threshold value.

    - x axis: threshold/value
    - left y axis: signal efficiency (TPR)
    - right y axis: background efficiency (FPR)

    Saves PNG to `outpath`.
    """
    if not rows:
        raise ValueError('No scan rows to plot')

    thresholds = np.array([r['threshold'] for r in rows])
    tpr = np.array([r['tpr'] for r in rows])
    bkg_rej = np.array([r['bkg_rej'] for r in rows])
    # background efficiency = fraction of background that PASSES = 1 - bkg_rej
    bkg_eff = 1.0 - bkg_rej
    metric = np.array([r['metric'] for r in rows])

    best_idx = int(np.nanargmax(metric))

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.plot(thresholds, tpr, color='tab:blue', lw=1.5, label='Signal eff (TPR)')
    ax2.plot(thresholds, bkg_eff, color='tab:orange', lw=1.5, label='Background eff (FPR)')

    ax1.scatter([thresholds[best_idx]], [tpr[best_idx]], color='tab:blue', s=60)
    ax2.scatter([thresholds[best_idx]], [bkg_eff[best_idx]], color='tab:orange', s=60)

    txt = f"thr={thresholds[best_idx]:.4g}\nTPR={tpr[best_idx]:.3f}\nBkgEff={bkg_eff[best_idx]:.3f}\nmetric={metric[best_idx]:.3g}"
    ax1.annotate(txt, xy=(0.02, 0.98), xycoords='axes fraction', fontsize=9, va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.6))

    ax1.set_xlabel('Threshold / Value')
    ax1.set_ylabel('Signal efficiency (TPR)', color='tab:blue')
    ax2.set_ylabel('Background efficiency (FPR)', color='tab:orange')

    if xlim is not None:
        ax1.set_xlim(*xlim)
    else:
        x_min = float(np.nanmin(thresholds))
        x_max = float(np.nanmax(thresholds))
        if x_min == x_max:
            ax1.set_xlim(x_min - 1, x_max + 1)
        else:
            pad = 0.05 * (x_max - x_min)
            ax1.set_xlim(x_min - pad, x_max + pad)

    ax1.grid(True)

    # legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.tight_layout()
    plt.savefig(outpath)
    if show:
        plt.show()
    plt.close(fig)


# ============================================================================
# Multi-Background Optimization: Visualization & I/O
# ============================================================================

def save_csv_multi_bkg(results, outpath):
    """Save multi-background scan results to CSV"""
    if not results:
        return
    
    # Build fieldnames from first result
    first_result = results[0]
    fieldnames = ['threshold', 'sig_efficiency', 'sig_pass', 'sig_total']
    
    for bkg_name in sorted(first_result.bkg_pass.keys()):
        fieldnames.extend([
            f'{bkg_name}_efficiency',
            f'{bkg_name}_rejection',
            f'{bkg_name}_pass',
            f'{bkg_name}_total'
        ])
    
    fieldnames.extend(['weighted_score', 'youden_vs_all', 's_over_sqrtb_all', 'primary_metric'])
    
    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = result.to_dict()
            writer.writerow(row)


def plot_efficiency_vs_value_multi_bkg(results, bkg_names, outpath=None, show=False):
    """
    Plot signal and background efficiencies vs threshold value.
    
    Args:
        results: [CutScanResult, ...]
        bkg_names: list of background names to plot
        outpath: output path (if None, returns figure)
        show: whether to call plt.show()
    
    Returns:
        fig (if outpath is None)
    """
    if not results:
        raise ValueError('No results to plot')
    
    thresholds = np.array([r.threshold for r in results])
    sig_eff = np.array([r.sig_efficiency for r in results])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Signal efficiency (always the same)
    ax1.plot(thresholds, sig_eff, 'o-', lw=2, label='Signal', color='black', markersize=4)
    ax1.set_ylabel('Signal Efficiency', fontsize=12)
    ax1.set_xlabel('Threshold Value', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_title('Signal Efficiency vs Threshold', fontsize=12)
    
    # Background efficiencies
    colors = plt.cm.Set2(np.linspace(0, 1, len(bkg_names)))
    for bkg_name, color in zip(bkg_names, colors):
        bkg_eff = np.array([r.bkg_efficiency.get(bkg_name, 0) for r in results])
        ax2.plot(thresholds, bkg_eff, 'o-', lw=2, label=bkg_name, color=color, markersize=4)
    
    ax2.set_ylabel('Background Efficiency', fontsize=12)
    ax2.set_xlabel('Threshold Value', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_title('Background Efficiency vs Threshold', fontsize=12)
    
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath, dpi=150)
        plt.close(fig)
        return None
    else:
        if show:
            plt.show()
        return fig


def plot_multi_roc(results_dict, outdir='./'):
    """
    Plot ROC curves for each background separately.
    
    Args:
        results_dict: {bkg_name: [CutScanResult, ...], ...}
        outdir: output directory for plots
    """
    os.makedirs(outdir, exist_ok=True)
    
    n_bkg = len(results_dict)
    n_rows = (n_bkg + 1) // 2
    n_cols = min(2, n_bkg)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
    if n_bkg == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (bkg_name, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        sig_eff = np.array([r.sig_efficiency for r in results])
        bkg_rej = np.array([r.bkg_rejection.get(bkg_name, 0) for r in results])
        metrics = np.array([r.primary_metric for r in results])
        
        # Best is first after sorting
        best_idx = 0
        
        scatter = ax.scatter(bkg_rej, sig_eff, c=metrics, cmap='viridis', s=30, alpha=0.7)
        ax.plot(bkg_rej, sig_eff, '-', lw=0.5, alpha=0.5)
        ax.scatter([bkg_rej[best_idx]], [sig_eff[best_idx]], 
                  color='red', s=150, marker='*', label='best', zorder=5)
        
        ax.set_xlabel(f'{bkg_name} Rejection', fontsize=11)
        ax.set_ylabel('Signal Efficiency', fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_title(f'ROC vs {bkg_name}', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Metric', fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_bkg, len(axes)):
        axes[idx].remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'multi_roc_curves.png'), dpi=150)
    plt.close()


def print_scan_summary_multi_bkg(results, top_n=5):
    """Print summary of top N scan results"""
    if not results:
        print("No results to summarize")
        return
    
    print(f"\n{'='*100}")
    print(f"Top {min(top_n, len(results))} Scan Results (sorted by metric)")
    print(f"{'='*100}\n")
    
    for idx, result in enumerate(results[:top_n]):
        print(f"Rank {idx+1}:")
        print(f"  Threshold: {result.threshold:.6g}")
        print(f"  Signal Efficiency: {result.sig_efficiency:.4f} ({result.sig_pass}/{result.sig_total})")
        for bkg_name in sorted(result.bkg_pass.keys()):
            eff = result.bkg_efficiency[bkg_name]
            rej = result.bkg_rejection[bkg_name]
            print(f"  {bkg_name:12s} Efficiency: {eff:.4f}, Rejection: {rej:.4f} "
                  f"({result.bkg_pass[bkg_name]}/{result.bkg_total[bkg_name]})")
        print(f"  Primary Metric: {result.primary_metric:.6g}")
        if result.weighted_score is not None:
            print(f"  Weighted Score: {result.weighted_score:.6g}")
        print()


def optimize_from_event_arrays(val, mc_count, signal_code=168, background_codes=None, direction='greater', n_steps=200, metric='youden'):
    """Optimize threshold given per-candidate `val` and event-level `mc_count`.

    `val` and `mc_count` can be numpy arrays, lists, or awkward arrays.
    The function will flatten inputs and split into signal (where `mc_count==signal_code`)
    and background (where `mc_count` is in `background_codes` or != `signal_code` if
    `background_codes` is None).

    Returns (rows, best_row).
    """
    # Lazy import awkward if available
    try:
        import awkward as ak
    except Exception:
        ak = None

    # Convert to flat numpy arrays where possible
    val_flat = to_1d_numpy(val)
    mc_flat = to_1d_numpy(mc_count)

    if val_flat.size == 0:
        # try awkward flatten fallback
        if ak is not None and isinstance(val, ak.Array):
            val_flat = np.asarray(ak.flatten(ak.drop_none(val), axis=None))

    if mc_flat.size == 0:
        if ak is not None and isinstance(mc_count, ak.Array):
            mc_flat = np.asarray(ak.flatten(mc_count, axis=None))

    if val_flat.size != mc_flat.size:
        # If lengths mismatch, try to broadcast per-event values (assume val nested per event)
        if ak is not None and isinstance(val, ak.Array):
            sig_vals = np.asarray(ak.flatten(ak.mask(val, mc_count == signal_code), axis=None))
            if background_codes is None:
                bkg_vals = np.asarray(ak.flatten(ak.mask(val, mc_count != signal_code), axis=None))
            else:
                mask = None
                for code in background_codes:
                    this_mask = (mc_count == code)
                    mask = this_mask if mask is None else (mask | this_mask)
                bkg_vals = np.asarray(ak.flatten(ak.mask(val, mask), axis=None))
            rows = scan_thresholds(sig_vals, bkg_vals, direction=direction, n_steps=n_steps, metric=metric)
            best = find_best(rows)
            return rows, best
        else:
            raise ValueError('Input arrays could not be aligned: val and mc_count lengths differ')

    # Now we have flat arrays of equal length
    if background_codes is None:
        sig_mask = (mc_flat == signal_code)
        bkg_mask = (mc_flat != signal_code)
    else:
        sig_mask = (mc_flat == signal_code)
        bkg_mask = np.zeros_like(mc_flat, dtype=bool)
        for code in background_codes:
            bkg_mask = bkg_mask | (mc_flat == code)

    sig_vals = val_flat[sig_mask]
    bkg_vals = val_flat[bkg_mask]

    rows = scan_thresholds(sig_vals, bkg_vals, direction=direction, n_steps=n_steps, metric=metric)
    best = find_best(rows)
    return rows, best


def optimize_on_combine_result(val, mc_count, signal_code=168, background_codes=None, direction='greater', n_steps=200, metric='youden'):
        """Convenience wrapper to run optimization directly on arrays produced from
        `combine_result` (no file I/O).

        Parameters:
            - val: per-candidate value array (awkward or numpy), e.g. ak.mask(combine_result['trkfit']["trksegpars_lh"], test_mask)['maxr']
            - mc_count: per-event particle code array (from `count_particle_types(combine_result)`).
            - signal_code: code identifying signal events (default 168)
            - background_codes: iterable of codes to use as background; if None, all non-signal codes are used

        Returns:
            - rows: list of scan result dicts
            - best: dict for best threshold (same structure as rows entries)
        """
        return optimize_from_event_arrays(val, mc_count, signal_code=signal_code, background_codes=background_codes, direction=direction, n_steps=n_steps, metric=metric)


# ============================================================================
# Multi-Background Optimization: Convenience Wrappers
# ============================================================================

def optimize_on_combine_result_multi_bkg(
    feature_dict,
    direction='greater',
    n_steps=200,
    metric='weighted',
    weights=None,
    target_rejections=None
):
    """
    Optimize cuts directly from analysis data (no file I/O).
    
    Args:
        feature_dict (dict): {sample_name: feature_array}
                            'signal' key identifies signal sample
                            Other keys are background names
        direction (str): 'greater' or 'less'
        n_steps (int): number of threshold points
        metric (str): 'weighted', 'youden', 's_over_sqrtb'
        weights (dict): {name: weight} for composite metric
        target_rejections (dict): informational target rejections per background
    
    Returns:
        list of CutScanResult objects sorted by metric (best first)
    
    Usage:
        results = optimize_on_combine_result_multi_bkg(
            feature_dict={
                'signal': ceMLL_data['d0'],
                'RPC': rpc_data['d0'],
                'Cosmics': cosmics_data['d0'],
                'DIO': dio_data['d0']
            },
            direction='less',
            weights={'signal': 1.0, 'RPC': 0.3, 'Cosmics': 0.3, 'DIO': 0.6}
        )
    """
    # Extract signal
    sig_vals = to_1d_numpy(feature_dict.get('signal'))
    
    # Extract backgrounds
    bkg_dict = {}
    for name in feature_dict:
        if name != 'signal':
            bkg_dict[name] = to_1d_numpy(feature_dict[name])
    
    return scan_thresholds_multi_bkg(
        sig_vals,
        bkg_dict,
        direction=direction,
        n_steps=n_steps,
        metric=metric,
        weights=weights,
        target_rejections=target_rejections
    )


# ============================================================================
# Batch Cut Optimization from ROOT Files
# ============================================================================

def batch_optimize_cuts_from_root(
    root_file_dict,
    cut_definitions,
    output_dir='./cut_optimization',
    n_steps=300,
    weights=None,
    metric='weighted',
    tree_name='ntp1',
    max_events=None,
    verbosity=1
):
    """
    Batch optimize multiple cuts from ROOT files for multi-background analysis.
    
    This is the main entry point for optimizing all cuts simultaneously across
    signal (CeMLL) and multiple backgrounds (RPC, Cosmics, DIO).
    
    Args:
        root_file_dict (dict): Mapping of sample names to file paths
            Example:
            {
                'signal': ['file1.root', 'file2.root'],
                'RPC': ['rpc1.root', 'rpc2.root'],
                'Cosmics': ['cosmics1.root'],
                'DIO': ['dio1.root']
            }
        
        cut_definitions (dict): Dictionary defining cuts to optimize
            Example:
            {
                'd0': {
                    'path': 'trkfit.trksegpars_lh.d0',
                    'direction': 'less',  # or 'greater', 'range'
                    'description': 'Distance of closest approach'
                },
                'trkqual': {
                    'path': 'trk.trkqual',
                    'direction': 'greater',
                    'description': 'Track quality'
                },
                ...
            }
        
        output_dir (str): Directory to save results
        n_steps (int): Number of threshold points to scan
        weights (dict): Sample weights for composite metric
            Default: {'signal': 1.0, 'RPC': 0.3, 'Cosmics': 0.3, 'DIO': 0.6}
        metric (str): 'weighted', 'youden', or 's_over_sqrtb'
        tree_name (str): Name of ROOT TTree
        max_events (int): Max events per file (None = all)
        verbosity (int): Verbosity level
    
    Returns:
        dict: {cut_name: results_list} for all cuts optimized
    
    Usage:
        results = batch_optimize_cuts_from_root(
            root_file_dict={
                'signal': ['CeMLL_sample.root'],
                'RPC': ['RPC_sample.root'],
                'Cosmics': ['Cosmics_sample.root'],
                'DIO': ['DIO_sample.root']
            },
            cut_definitions={
                'd0': {'path': 'trkfit.trksegpars_lh.d0', 'direction': 'less'},
                'trkqual': {'path': 'trk.trkqual', 'direction': 'greater'},
                'mom': {'path': 'trkfit.trksegs.mom', 'direction': 'range'}
            },
            output_dir='./cut_optimization'
        )
    """
    try:
        import uproot
    except ImportError:
        raise ImportError("uproot is required for batch_optimize_cuts_from_root")
    
    try:
        import awkward as ak
    except ImportError:
        raise ImportError("awkward is required for batch_optimize_cuts_from_root")
    
    # Set default weights if not provided
    if weights is None:
        weights = {
            'signal': 1.0,
            'RPC': 0.3,
            'Cosmics': 0.3,
            'DIO': 0.6
        }
    
    os.makedirs(output_dir, exist_ok=True)
    
    if verbosity >= 1:
        print(f"\n{'='*80}")
        print("BATCH CUT OPTIMIZATION FROM ROOT FILES")
        print(f"{'='*80}\n")
    
    # Step 1: Load all data
    if verbosity >= 1:
        print("[1/3] Loading ROOT files...\n")
    
    data_dict = {}
    for sample_name, file_list in root_file_dict.items():
        if not isinstance(file_list, list):
            file_list = [file_list]
        
        arrays = []
        for filepath in file_list:
            if verbosity >= 2:
                print(f"  Loading {sample_name}: {os.path.basename(filepath)}")
            
            try:
                with uproot.open(filepath) as file:
                    tree = file[tree_name]
                    n_events = tree.num_entries
                    if max_events is not None:
                        n_events = min(n_events, max_events)
                    
                    arr = tree.arrays(entry_stop=n_events, how=dict)
                    arrays.append(arr)
                    
                    if verbosity >= 2:
                        print(f"    → {n_events} events loaded")
            except Exception as e:
                if verbosity >= 1:
                    print(f"  WARNING: Could not load {filepath}: {e}")
                continue
        
        # Combine arrays from all files
        if arrays:
            data_dict[sample_name] = ak.concatenate(
                [ak.Array(a) for a in arrays]
            )
            if verbosity >= 1:
                total_events = sum(len(a) for a in arrays)
                print(f"  {sample_name:12s}: {total_events} events total")
    
    if not data_dict:
        raise ValueError("No data loaded from ROOT files")
    
    if verbosity >= 1:
        print()
    
    # Step 2: Optimize each cut
    if verbosity >= 1:
        print("[2/3] Optimizing cuts...\n")
    
    all_results = {}
    
    for cut_idx, (cut_name, cut_def) in enumerate(cut_definitions.items(), 1):
        if verbosity >= 1:
            desc = cut_def.get('description', cut_name)
            print(f"  [{cut_idx}/{len(cut_definitions)}] {cut_name}: {desc}")
        
        feature_path = cut_def['path']
        direction = cut_def['direction']
        
        # Extract feature from each sample
        feature_dict = {}
        try:
            for sample_name, data in data_dict.items():
                # Navigate nested structure (e.g., 'trkfit.trksegpars_lh.d0')
                parts = feature_path.split('.')
                obj = data
                for part in parts:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        obj = obj[part]
                
                # Flatten to 1D
                obj_flat = ak.flatten(obj, axis=None)
                feature_dict[sample_name] = np.asarray(obj_flat)
                
                if verbosity >= 3:
                    print(f"      {sample_name}: {len(feature_dict[sample_name])} values")
        except Exception as e:
            if verbosity >= 1:
                print(f"    ERROR extracting {feature_path}: {e}")
            continue
        
        # Rename 'signal' to 'signal' for optimizer
        if 'signal' in feature_dict:
            opt_dict = {'signal': feature_dict['signal']}
            for name in feature_dict:
                if name != 'signal':
                    opt_dict[name] = feature_dict[name]
        else:
            # If no 'signal' key, use first sample as signal
            opt_dict = feature_dict
        
        # Run optimization
        try:
            results = optimize_on_combine_result_multi_bkg(
                feature_dict=opt_dict,
                direction=direction,
                n_steps=n_steps,
                metric=metric,
                weights=weights
            )
            all_results[cut_name] = results
            
            # Print best result for this cut
            if results:
                best = results[0]
                if verbosity >= 2:
                    print(f"    Best threshold: {best.threshold:.6g}")
                    print(f"    Signal efficiency: {best.sig_efficiency:.1%}")
        except Exception as e:
            if verbosity >= 1:
                print(f"    ERROR optimizing: {e}")
            continue
    
    if verbosity >= 1:
        print()
    
    # Step 3: Save results
    if verbosity >= 1:
        print("[3/3] Saving results...\n")
    
    for cut_name, results in all_results.items():
        # CSV
        csv_path = os.path.join(output_dir, f'optimize_{cut_name}_multi_bkg.csv')
        save_csv_multi_bkg(results, csv_path)
        if verbosity >= 2:
            print(f"  {cut_name}: saved CSV to {csv_path}")
        
        # Plot: efficiency vs value
        try:
            bkg_names = [n for n in all_results[cut_name][0].bkg_rejection.keys()]
            plot_path = os.path.join(output_dir, f'optimize_{cut_name}_efficiency.png')
            plot_efficiency_vs_value_multi_bkg(results, bkg_names, outpath=plot_path)
            if verbosity >= 2:
                print(f"  {cut_name}: saved plot to {plot_path}")
        except Exception as e:
            if verbosity >= 2:
                print(f"  {cut_name}: WARNING - could not generate plot: {e}")
    
    # Summary plot: ROC curves for each cut
    if verbosity >= 2:
        print()
    
    try:
        for cut_name, results in list(all_results.items())[:4]:  # Limit to 4 plots
            results_dict = {}
            for bkg_name in results[0].bkg_rejection.keys():
                results_dict[bkg_name] = results
            
            plot_dir = os.path.join(output_dir, 'roc_curves')
            os.makedirs(plot_dir, exist_ok=True)
            
            # Need to temporarily save figures per cut
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for idx, (bkg_name, _) in enumerate(results_dict.items()):
                if idx >= 4:
                    break
                ax = axes[idx]
                
                sig_eff = np.array([r.sig_efficiency for r in results])
                bkg_rej = np.array([r.bkg_rejection.get(bkg_name, 0) for r in results])
                metrics = np.array([r.primary_metric for r in results])
                
                scatter = ax.scatter(bkg_rej, sig_eff, c=metrics, cmap='viridis', s=20, alpha=0.7)
                ax.plot(bkg_rej, sig_eff, '-', lw=0.5, alpha=0.5)
                ax.scatter([bkg_rej[0]], [sig_eff[0]], color='red', s=100, marker='*', zorder=5)
                
                ax.set_xlabel(f'{bkg_name} Rejection')
                ax.set_ylabel('Signal Efficiency')
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-0.02, 1.02)
                ax.grid(True, alpha=0.3)
                ax.set_title(f'{cut_name}: ROC vs {bkg_name}')
            
            plt.tight_layout()
            roc_path = os.path.join(plot_dir, f'roc_{cut_name}.png')
            plt.savefig(roc_path, dpi=100)
            plt.close()
            if verbosity >= 2:
                print(f"  {cut_name}: saved ROC to {roc_path}")
    except Exception as e:
        if verbosity >= 1:
            print(f"  WARNING: Could not generate ROC plots: {e}")
    
    if verbosity >= 1:
        print(f"\n{'='*80}")
        print(f"Optimization complete. Results saved to: {output_dir}")
        print(f"{'='*80}\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Optimize a single-feature threshold for signal vs background')
    parser.add_argument('--sig', required=True, help='Signal input (.npz or .npy)')
    parser.add_argument('--bkg', required=True, help='Background input (.npz or .npy)')
    parser.add_argument('--feature', required=False, default=None, help='Feature key inside .npz (default: first)')
    parser.add_argument('--direction', choices=['greater', 'less'], default='greater')
    parser.add_argument('--nsteps', type=int, default=200)
    parser.add_argument('--metric', choices=['youden', 's_over_sqrtb'], default='youden')
    parser.add_argument('--out', default='optimize_scan.csv', help='CSV output file')
    args = parser.parse_args()

    sig_vals = load_feature(args.sig, args.feature)
    bkg_vals = load_feature(args.bkg, args.feature)

    sig_vals = np.asarray(sig_vals)
    bkg_vals = np.asarray(bkg_vals)

    rows = scan_thresholds(sig_vals, bkg_vals, direction=args.direction, n_steps=args.nsteps, metric=args.metric)
    best = find_best(rows)
    save_csv(rows, args.out)

    print('Scan complete. CSV written to', args.out)
    if best is not None:
        print('Best threshold: {:.6g}, metric: {:.6g}, TPR: {:.4f}, BkgRej: {:.4f}, nsig_pass: {}, nbkg_pass: {}'.format(
            best['threshold'], best['metric'], best['tpr'], best['bkg_rej'], best['nsig_pass'], best['nbkg_pass']))
    else:
        print('No thresholds evaluated')


if __name__ == '__main__':
    main()
