"""optimize_cuts.py

Simple threshold scanner to optimize a single feature cut for signal efficiency
vs background rejection.
"""
import argparse
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

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
