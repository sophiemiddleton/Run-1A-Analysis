import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from pyutils.pyselect import Select
from pyutils.pyvector import Vector

import matplotlib.font_manager as mfm
import matplotlib as mpl
import matplotlib.ticker as ticker
preferred_serifs = ['DejaVu Serif', 'Times New Roman', 'Times', 'Palatino']
available_fonts = {f.name for f in mfm.fontManager.ttflist}
chosen_serif = next((f for f in preferred_serifs if f in available_fonts), 'DejaVu Serif')

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': [chosen_serif],
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 9,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
})
import zfit


class Cosmics():
    """Class to conduct comparisons between cut or data sets
    """
    def __init__(self ):
      """
      """
      
      # Custom prefix for log messages from this processor
      self.print_prefix = "[Compare] "
      print(f"{self.print_prefix}Initialised")

    def fit_time(self, data_list, labels):
        """
        Plots the reconstructed time data and its statistical uncertainties
        using an extended unbinned maximum likelihood fit with an exponential shape,
        including goodness of fit and pull distribution.
        Styled identically to the momentum fitting method.
        """
        # Create figure matching structure of fit_momentum (Main plot + Pull plot)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        colors = ["#1f77b4", "#ff7f0e"]  # Professional Blue and Orange
        linecolors = ["red", "green"]
        
        # Text box lists to compile multi-dataset information cleanly
        param_text_elements = []
        
        # Consistent layout limits and definition
        fit_range = (400,700)
        n_bins = 50
        bin_width = (fit_range[1] - fit_range[0]) / n_bins
        
        last_norm = 0.0
        for i, data in enumerate(data_list):
            time_skim = ak.nan_to_none(data)
            time_skim = ak.drop_none(time_skim)

            # Define the observable space for the fit matching fit_range
            obs_time = zfit.Space('x', limits=fit_range)
            time_np = ak.to_numpy(ak.flatten(time_skim, axis=None))
            time_zfit = zfit.Data.from_numpy(array=time_np, obs=obs_time)
            n_events_raw = len(time_np)
            
            # Initialize extended parameter at raw count size
            N_Cosmic = zfit.Parameter(f'N_Cosmic_time_{i}', n_events_raw, 100, n_events_raw * 10)
            #N_Cosmic = zfit.Parameter('N_Cosmic', 150000, 100, 500000)
            
            # Create parameters for the coefficients
            c1 = zfit.Parameter("c1", 0.1, -1, 1)
            coeffs = [c1]

            # Create a Chebyshev polynomial PDF
            poly_model = zfit.pdf.Chebyshev(obs=obs_time, coeffs=coeffs, extended=N_Cosmic)

            # Perform the extended unbinned NLL fit
            nll = zfit.loss.ExtendedUnbinnedNLL(model=poly_model, data=time_zfit)
            minimizer = zfit.minimize.Minuit()
            result = minimizer.minimize(loss=nll)
            hesse_errors = result.hesse()
            print(result)
            
            # Normalization factor calculations matching momentum style
            target_events = 100000.0
            norm_factor = target_events / n_events_raw  

            # Histogramming raw data
            counts_raw, bins = np.histogram(time_np, bins=n_bins, range=fit_range)
            data_bin_center = (bins[:-1] + bins[1:]) / 2
            
            counts_norm = counts_raw * norm_factor
            errors_norm = np.sqrt(counts_raw) * norm_factor
            
            nonzero_mask = counts_raw > 0
            
            # Plot markers on Main Axis (ax1)
            ax1.errorbar(data_bin_center[nonzero_mask], counts_norm[nonzero_mask], 
                        yerr=errors_norm[nonzero_mask], 
                        fmt='o', color=colors[i], markerfacecolor='white', markeredgecolor=colors[i], 
                        markersize=4, capsize=0, elinewidth=1, label=f'{labels[i]} Stat. unc.')
            
            # Plot step histogram using weights
            weights = np.full_like(time_np, norm_factor)
            ax1.hist(time_np, bins=n_bins, range=fit_range, weights=weights,
                    color=colors[i], histtype='step', linewidth=1.2, label=labels[i])

            # Generate and plot fit curve
            time_plot = np.linspace(fit_range[0], fit_range[1], 500).reshape(-1, 1)
            fitcurve_curve = zfit.run(poly_model.ext_pdf(time_plot)) * bin_width * norm_factor
            
            ax1.plot(time_plot.flatten(), fitcurve_curve.flatten(), color=linecolors[i], linestyle="--", linewidth=2.5, label=f'{labels[i]} Fit')
            
            # Calculate Chi2 / DOF
            fit_at_bin_center_raw = zfit.run(poly_model.ext_pdf(data_bin_center.reshape(-1, 1))) * bin_width
            chi2 = np.sum(((counts_raw - fit_at_bin_center_raw) ** 2) / (counts_raw + 1e-6))
            dof = n_bins - len(coeffs) - 1
            chi2_dof = chi2 / dof if dof > 0 else 0
            
            # Calculate and plot pulls on Second Axis (ax2)
            residual_norm = counts_norm - (fit_at_bin_center_raw * norm_factor)
            pull_errors = np.where(counts_raw > 0, errors_norm, 1.0 * norm_factor)
            pull = residual_norm / pull_errors
            pull_err = np.ones_like(pull)
            
            ax2.errorbar(data_bin_center, pull, yerr=pull_err, 
                        fmt='.', color=colors[i], capsize=0, markersize=6)
                
            # Formatting text box entries
            clean_label = labels[i].replace(" ", r"\ ")
            dataset_text = (
                f"$\\mathbf{{{clean_label}}}:$\n"
                f"$\\c1 = {result.params[c1]['value']:.4f} \\pm {hesse_errors[c1]['error']:.4f}$\n"
                f"$\\chi^2/\\text{{DOF}} = {chi2_dof:.2f}$"
            )
            param_text_elements.append(dataset_text)
            
            last_norm = result.params[N_Cosmic]['value']

        # Consolidated parameter box presentation
        combined_param_text = "\n\n".join(param_text_elements)
        props = dict(boxstyle='round,pad=0.5', facecolor='lightgrey', alpha=0.75, edgecolor='gray')
        ax1.text(0.95, 0.95, combined_param_text, transform=ax1.transAxes,
            fontsize=14, horizontalalignment='right', verticalalignment='top', bbox=props)
        
        # Uniform layout configuration
        ax1.set_ylabel(f'Events / {bin_width:.1f} ns', fontsize=mpl.rcParams.get('axes.titlesize', 24))
        ax1.text(0.0, 1.02, "Mu2e Simulation", 
                fontsize=20, fontweight='bold', ha='left', va='bottom', transform=ax1.transAxes, zorder=100)
        ax1.legend(fontsize=14)
        
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_ylabel(r'Pull [$\sigma$]', fontsize=mpl.rcParams.get('axes.titlesize', 24))
        ax2.set_xlabel('Track Time [ns]', fontsize=mpl.rcParams.get('axes.titlesize', 24))
        ax2.set_ylim(-3.5, 3.5)
        
        # Matching minor/major tick configurations exactly
        ax1.minorticks_on()
        ax1.tick_params(direction='in', which='both', top=True, right=True, labelsize=14)
        ax1.yaxis.set_minor_formatter(ticker.NullFormatter()) 
        ax2.minorticks_on()
        ax2.tick_params(direction='in', which='both', top=True, right=True, labelsize=14)
        ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        plt.tight_layout()
        plt.savefig("CosmicTime.pdf", bbox_inches='tight')
        plt.show()
        
        return last_norm

    def fit_momentum(self, data_list):
        """
        Fits a simple Polynomial shape to the reconstructed momentum data
        using an extended unbinned maximum likelihood fit.
        """
        import zfit
        from matplotlib.ticker import MultipleLocator, AutoMinorLocator
        
        # Create figure with single main plot for publication
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))
        colors = ["#1f77b4", "#ff7f0e"]  # Professional blue and orange
        labels = ["on-spill", "off-spill"]
        
        # Compute text box y-positions with small gaps, starting at 0.3
        y_start = 0.3
        y_gap = 0.25
        text_y_pos = [y_start + i * y_gap for i in range(len(data_list))]
        norm = 0.
        for i, data in enumerate(data_list):
            mom_mag_skim = ak.nan_to_none(data)
            mom_mag_skim = ak.drop_none(mom_mag_skim)

            # Define the observable space for the fit
            obs_mom = zfit.Space('x', limits=(97,110))#
            mom_np = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))
            mom_zfit = zfit.Data.from_numpy(array=mom_np, obs=obs_mom)
            
            # Define parameters for the Polynomial shape and yield
            #mu = zfit.Parameter("mu", 98, 96, 102)
            #sigma = zfit.Parameter("sigma", 10, 5, 20)
            N_Cosmic = zfit.Parameter('N_Cosmic', 150000, 100, 500000)
            
            # Create parameters for the coefficients
            c1 = zfit.Parameter("c1", 0.1, -1, 1)
            #c2 = zfit.Parameter("c2", 0.1, -1, 1)
            #c3 = zfit.Parameter("c3", 0.1, -1, 1)
            #c4 = zfit.Parameter("c4", 0.1, -1, 1)
            #c5 = zfit.Parameter("c5", 0.1, -1, 1)
            coeffs = [c1]#, c2]#, c3, c4, c5]

            # Create a Chebyshev polynomial PDF
            poly_model = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=coeffs, extended=N_Cosmic)


            # Create the extended Polynomial PDF
            #gauss = zfit.pdf.Gauss(obs=obs_mom, mu=mu, sigma=sigma, extended=N_Cosmic)
            
            # Create the extended unbinned negative log-likelihood loss
            nll = zfit.loss.ExtendedUnbinnedNLL(model=poly_model, data=mom_zfit)
            
            # Minimize the loss and get the result
            minimizer = zfit.minimize.Minuit()
            result = minimizer.minimize(loss=nll)
            hesse_errors = result.hesse()
            print(result)
            
            # --- Plotting the fit result ---
            
            fit_range = (obs_mom.lower[0, 0], obs_mom.upper[0, 0])
            n_bins = 50
            bin_width = (fit_range[1] - fit_range[0]) / n_bins
            
            # --- Main plot ---
            
            mom_plot = np.linspace(fit_range[0], fit_range[1], 200).reshape(-1, 1)

            # Compute histogram counts first for proper error calculation
            data_hist_counts, data_bins = np.histogram(mom_np, bins=n_bins, range=fit_range)
            data_bin_center = (data_bins[:-1] + data_bins[1:]) / 2
            
            # Normalize histogram to density (area under histogram = 1)
            total_count = np.sum(data_hist_counts)
            data_hist = data_hist_counts / (total_count * bin_width)
            
            # Plot histogram with density
            ax1.hist(mom_np, color=colors[i], bins=n_bins, range=fit_range, label=labels[i], histtype='step', density=True)
            
            # Error bars for density histogram
            data_hist_errors = np.sqrt(data_hist_counts) / (total_count * bin_width)
            ax1.errorbar(data_bin_center, data_hist, yerr=data_hist_errors, fmt='.', color=colors[i], capsize=2)
            
            # Plot fit curve as PDF (also normalized to integrate to 1)
            poly_model_curve = zfit.run(poly_model.pdf(mom_plot))
            ax1.plot(mom_plot.flatten(), poly_model_curve.flatten(), color=colors[i], linestyle="-", linewidth=2.0, label=str(labels[i]) + ' Polynomial Fit')
            
            ax1.set_xlabel('Reconstructed Momentum [MeV/c]', fontsize=13, fontweight='bold')
            ax1.set_ylabel('Probability density', fontsize=13, fontweight='bold')
            ax1.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
            ax1.set_title('Polynomial Fit to Momentum Data', fontsize=14, fontweight='bold', pad=15)
            
            # Set major and minor ticks
            ax1.xaxis.set_major_locator(MultipleLocator(5))  # Major ticks every 5 MeV
            ax1.xaxis.set_minor_locator(AutoMinorLocator(2))  # Minor ticks between major ticks
            ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
            
            # Style the ticks
            ax1.tick_params(which='major', length=8, width=1.5, labelsize=11)
            ax1.tick_params(which='minor', length=4, width=1.0)
            
            # Remove top and right spines for cleaner publication look
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_linewidth(1.5)
            ax1.spines['bottom'].set_linewidth(1.5)
            
            # --- Add text box with fit parameters ---
            param_text = (
                f"Fit parameters for {labels[i]}:\n"
                f"$N_{{Cosmic}} = {result.params[N_Cosmic]['value']:.0f} \\pm {hesse_errors[N_Cosmic]['error']:.2f}$\n"
                f"$c_{{1}} = {result.params[c1]['value']:.3f} \\pm {hesse_errors[c1]['error']:.4f}$\n"
                #f"$c_{{2}} = {result.params[c2]['value']:.3f} \\pm {hesse_errors[c2]['error']:.4f}$\n"
                #f"$c_{{3}} = {result.params[c3]['value']:.3f} \\pm {hesse_errors[c3]['error']:.4f}$\n"
                #f"$c_{{4}} = {result.params[c4]['value']:.3f} \\pm {hesse_errors[c4]['error']:.4f}$\n"
                #f"$c_{{5}} = {result.params[c5]['value']:.3f} \\pm {hesse_errors[c5]['error']:.4f}$"
                
            )
            
            props = dict(boxstyle='round,pad=0.8', facecolor=colors[i], alpha=0.15, edgecolor='black', linewidth=1.5)
            
            # Position the text box with offset for each iteration
            ax1.text(0.4, text_y_pos[i], param_text, transform=ax1.transAxes,
                     fontsize=9, verticalalignment='top', bbox=props, family='monospace')
            
            #mean = result.params[mu]['value']
            #meanr_err = hesse_errors[mu]['error']
            #sigma = result.params[sigma]['value']
            #sigma_err = hesse_errors[sigma]['error']
            norm = result.params[N_Cosmic]['value']
        plt.tight_layout()
        plt.savefig("Cosmicfit.pdf")
        plt.show()
        return  norm

    def overlay_fit(self, c1, c2, data_list, mc_count): #, c3, c4, c5,
        """
        Fits a simple Polynomial shape to the reconstructed momentum data
        using an extended unbinned maximum likelihood fit.
        """
        # Create figure with two subplots: main plot and ratio plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        colors = ["black"]
        labels = ["Run-1A"]
        
        # Compute text box y-positions to avoid overlap dynamically
        n_fits = len(data_list)
        y_start = 0.95
        y_step = 0.12
        text_y_pos = [y_start - i * y_step for i in range(n_fits)]

        for i, data in enumerate(data_list):
            mom_mag_skim = ak.nan_to_none(data)
            mom_mag_skim = ak.drop_none(mom_mag_skim)
            
            true_Cosmic = mom_mag_skim.mask[(mc_count[i] == -1) ]
            true_Cosmic = ak.to_numpy((ak.flatten(true_Cosmic,axis=None)))
            print(true_Cosmic)
            print(mc_count)

            # Define the observable space for the fit
            obs_mom = zfit.Space('x', limits=(95,115))
            mom_np = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))
            mom_zfit = zfit.Data.from_numpy(array=true_Cosmic, obs=obs_mom)
            
            # Define parameters for the Polynomial shape and yield
            N_Cosmic = zfit.Parameter('N_Cosmic', 5000, 100, 15000)
            
            # Create parameters for the coefficients
            c1 = zfit.Parameter("c1", c1,floating=False)
            c2 = zfit.Parameter("c2", c2, floating=False)
            #c3 = zfit.Parameter("c3", c3, floating=False)
            #c4 = zfit.Parameter("c4", c4, floating=False)
            #c5 = zfit.Parameter("c5", c5, floating=False)
            coeffs = [c1, c2]#, c3, c4, c5]

            # Create a Chebyshev polynomial PDF
            poly_model = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=coeffs, extended=N_Cosmic)


            
            # Create the extended unbinned negative log-likelihood loss
            nll = zfit.loss.ExtendedUnbinnedNLL(model=poly_model, data=mom_zfit)
            
            # Minimize the loss and get the result
            minimizer = zfit.minimize.Minuit()
            result = minimizer.minimize(loss=nll)
            hesse_errors = result.hesse()
            print(result)
            
            # --- Plotting the fit result ---
            
            fit_range = (obs_mom.lower[0, 0], obs_mom.upper[0, 0])
            n_bins = 50
            bin_width = (fit_range[1] - fit_range[0]) / n_bins
            
            # --- Main plot ---
            
            mom_plot = np.linspace(fit_range[0], fit_range[1], 200).reshape(-1, 1)

            poly_model_curve = zfit.run(poly_model.pdf(mom_plot) * result.params[N_Cosmic]['value'] * bin_width)
            ax1.plot(mom_plot.flatten(), poly_model_curve.flatten(), color=colors[i], linestyle="--", label=str(labels[i])+' Fitted Polynomial')
            ax1.grid(True)
            ax1.set_yscale('log')
            data_hist, data_bins, _ = ax1.hist(mom_np, color=colors[i], bins=n_bins, range=fit_range, label=labels[i], histtype='step')
            true_hist, true_bins, _ = ax1.hist(true_Cosmic, color="orange", bins=n_bins, range=fit_range, label="Cosmic", histtype='bar')
            data_bin_center = (data_bins[:-1] + data_bins[1:]) / 2
            ax1.errorbar(data_bin_center, data_hist, yerr=np.sqrt(data_hist), fmt='.', color=colors[i], capsize=2)
            
            ax1.set_xlabel('Reconstructed Momentum [MeV/c]')
            ax1.set_ylabel('# of events per bin')
            ax1.legend(loc='lower right')
            ax1.set_title('Polynomial Fit to Momentum Data (Extended Unbinned)')
            
            # --- Add text box with fit parameters ---
            param_text = (
                f"Fit parameters for {labels[i]}:\n"
                f"$N_{{Cosmic}} = {result.params[N_Cosmic]['value']:.0f} \\pm {hesse_errors[N_Cosmic]['error']:.2f}$"
            )
            
            props = dict(boxstyle='round', facecolor=colors[i], alpha=0.3)
            
            # Position the text box in the upper left corner of the subplot
            # with an offset for each iteration
            ax1.text(0.4, text_y_pos[i], param_text, transform=ax1.transAxes,
                     fontsize=10, verticalalignment='top', bbox=props)
            
            # --- Ratio plot ---
            
            data_bin_center_2d = data_bin_center.reshape(-1, 1)
            fit_at_bin_center = zfit.run(poly_model.pdf(data_bin_center_2d) * result.params[N_Cosmic]['value'] * bin_width)
            ratio = true_hist / fit_at_bin_center
            
            ax2.errorbar(data_bin_center, ratio, yerr=np.sqrt(data_hist) / fit_at_bin_center, fmt='.', color=colors[i], capsize=2)
            ax2.axhline(1, color='gray', linestyle='--')
            ax2.set_ylabel('Ratio (Cosmic/Fit)')
            ax2.set_xlabel('Reconstructed Momentum [MeV/c]')
            ax2.set_ylim(0.5, 1.5)
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig("CosmicOverlay.pdf")
        plt.show()

    def fit_time_quadratic(self, times, bins=50, range=None, plot=True, filename="time_linear_fit.pdf"):
        """
        HISTOGRAM-BASED LINEAR FIT TO TIME DISTRIBUTION

        This replaces the previous quadratic/zfit approach with a simple
        weighted linear least-squares fit to the histogrammed counts.

        The function produces a two-panel figure (main histogram + fit, and
        a ratio/residual subplot) styled similarly to `fit_momentum`.

        Returns: slope, intercept, and their uncertainties as a dict.
        """
        # Flatten awkward arrays if necessary
        try:
            times_arr = ak.to_numpy(ak.flatten(ak.nan_to_none(times), axis=None))
        except Exception:
            times_arr = np.asarray(times)

        times_arr = times_arr[~np.isnan(times_arr)]
        if times_arr.size == 0:
            raise ValueError("No valid time data provided for fitting.")

        # Determine fit range
        if range is None:
            tmin, tmax = 475,1650
            pad = 1e-6 * (tmax - tmin) if tmax > tmin else 1.0
            fit_range = (tmin - pad, tmax + pad)
        else:
            fit_range = range

        # Histogram the data
        n_bins = bins
        data_hist, data_bins = np.histogram(times_arr, bins=n_bins, range=fit_range)
        bin_centers = (data_bins[:-1] + data_bins[1:]) / 2.0

        # Now fit using zfit with a linear Chebyshev (degree 1) shape
        obs = zfit.Space('x', limits=fit_range)
        times_z = zfit.Data.from_numpy(array=times_arr, obs=obs)

        N_time = zfit.Parameter('N_time', float(len(times_arr)), 1.0, float(len(times_arr)) * 10.0)
        t_c1 = zfit.Parameter('t_c1', 0.0, -5.0, 5.0)
        # Chebyshev with one coeff -> linear dependence in the mapped variable
        poly = zfit.pdf.Chebyshev(obs=obs, coeffs=[t_c1], extended=N_time)
        nll = zfit.loss.ExtendedUnbinnedNLL(model=poly, data=times_z)
        minimizer = zfit.minimize.Minuit()
        zresult = minimizer.minimize(loss=nll)
        hesse_errors = zresult.hesse()

        # Prepare model curve (counts per bin) from zfit result
        bin_width = (fit_range[1] - fit_range[0]) / float(n_bins)
        xx = np.linspace(fit_range[0], fit_range[1], 500).reshape(-1, 1)
        pdf_vals = zfit.run(poly.pdf(xx) * zresult.params[N_time]['value'] * bin_width)

        # Fit a straight line (mx + c) to the zfit model's predicted counts
        xx_flat = xx.flatten()
        # Use np.polyfit to get slope/intercept and covariance
        p_lin, cov_lin = np.polyfit(xx_flat, pdf_vals.flatten(), 1, cov=True)
        slope, intercept = float(p_lin[0]), float(p_lin[1])
        slope_err = float(np.sqrt(cov_lin[0, 0])) if cov_lin is not None else 0.0
        intercept_err = float(np.sqrt(cov_lin[1, 1])) if cov_lin is not None else 0.0

        result = {
          'slope': slope,
          'intercept': intercept,
          'slope_err': slope_err,
          'intercept_err': intercept_err,
          'zfit_result': zresult,
        }

        if plot:
            # Create figure styled like fit_momentum: main + ratio
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                           gridspec_kw={'height_ratios': [3, 1]})
            color = 'blue'

            # Main plot: histogram with fit overlay and error bars
            ax1.grid(True)
            data_plot = ax1.hist(times_arr, bins=n_bins, range=fit_range, color=color, label='Data', histtype='step')
            data_bin_center = bin_centers
            ax1.errorbar(data_bin_center, data_hist, yerr=np.sqrt(data_hist), fmt='.', color=color, capsize=2)

            xx = np.linspace(fit_range[0], fit_range[1], 500)
            fit_line = slope * xx + intercept
            ax1.plot(xx, fit_line, color=color, linestyle='--', linewidth=2, label='Linear fit')

            ax1.set_xlabel('Time [ns]')
            ax1.set_ylabel('Counts per bin')
            ax1.set_title('Linear Fit to Time Distribution')
            ax1.legend()

            # Parameter textbox (similar styling to fit_momentum)
            param_text = (
                f"Fit params:\n"
                f"slope = {slope:.4e} ± {slope_err:.4e}\n"
                f"intercept = {intercept:.2f} ± {intercept_err:.2f}\n"
            )
            props = dict(boxstyle='round', facecolor=color, alpha=0.15)
            ax1.text(0.4, 0.6, param_text, transform=ax1.transAxes,
                     fontsize=10, verticalalignment='top', bbox=props)

            # Ratio / residual plot
            fit_at_bins = slope * data_bin_center + intercept
            valid = fit_at_bins > 0
            ratio = np.zeros_like(data_hist, dtype=float)
            ratio_err = np.zeros_like(data_hist, dtype=float)
            ratio[valid] = data_hist[valid] / fit_at_bins[valid]
            ratio_err[valid] = np.sqrt(data_hist[valid]) / fit_at_bins[valid]

            ax2.errorbar(data_bin_center[valid], ratio[valid], yerr=ratio_err[valid], fmt='.', color=color, capsize=2)
            ax2.axhline(1, color='gray', linestyle='--')
            ax2.set_ylabel('Ratio (Data/Fit)')
            ax2.set_xlabel('Time [ns]')
            ax2.set_ylim(0.8, 1.2)
            ax2.grid(True)
            ax1.set_yscale('log')

            plt.tight_layout()
            plt.savefig(filename)
            plt.show()

        return result
