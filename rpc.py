import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from pyutils.pyselect import Select
from pyutils.pyvector import Vector

import zfit

class RPC():
    """Class to conduct comparisons between cut or data sets
    """
    def __init__(self ):
      """
      """
      
      # Custom prefix for log messages from this processor
      self.print_prefix = "[Compare] "
      print(f"{self.print_prefix}Initialised")


    def compare_resolution(self, recomom, truemom):
      """
      stores difference between recon and true momentum for resolution comparison
      """
      truemom = truemom.mask[truemom > 85] # removes anything that we dont care about on the reconstruction
      recomom = ak.nan_to_none(recomom)
      recomom = ak.drop_none(recomom)
      truemom = ak.nan_to_none(truemom)
      truemom = ak.drop_none(truemom)

      differences = [
        reco[0] - truemom[i][j][0]
        for i, reco_list in enumerate(recomom)
        for j, reco in enumerate(reco_list)
        if len(reco) != 0 and len(truemom[i][j]) != 0
      ]
      
      return differences

    def plot_resolution(self, val_overlay, val_label, filenames, lo, hi, columns=[], density=True):
      """
      Plots distributions of the given parameter (val), splitting by process code

      Args:
          val : list of values e.g. rmax
          val_label : text formated value name e.g. "rmax"
          lo : plot range lower bound
          hi : plot range upper bound

      Returns:
          plots saved as pdfs
      """
      fig, (ax1) = plt.subplots(1,1)
      sets=[]
      cols = ['blue']
      labs = ['e+','e-']
      styles = ['bar','step']
      lines=["","-"]
      alphas = [0.2,1]
      text_contents = []
      for i, val in enumerate(val_overlay):
        val = ak.drop_none(val)
        val = np.array(ak.flatten(val,axis=None))
        mean_val = np.mean(val)
        std_dev = np.std(val)
        text_contents.append(str(labs[i])+ f"Mean: {mean_val:.2f}\nStd Dev: {std_dev:.2f}")
        sets.append([val])

      for i in range(0,len(sets)):
        ax1.set_yscale('log')
        dummy_handle = ax1.plot([], marker="",color='white', label=columns[i])
        n, bins, patch = ax1.hist(sets[i],range=(lo,hi), color=cols, label=labs, bins=50, histtype=styles[i], alpha=alphas[i], stacked=True, density=density)

      ax1.set_xlabel(str(val_label))
      ax1.set_xlim(lo,hi)
      ax1.legend(ncol=len(columns))
      for i in range(0,len(text_contents)):
        plt.text(0.1, 0.95-i*0.1, text_contents[i], 
                 transform=plt.gca().transAxes,
                 fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

      
      plt.savefig(str(filenames)+"_resolution.pdf")
      plt.show()
      

    def fit_momentum(self, data_list, labels):
        """
        Fits a Chebyshev polynomial shape to the reconstructed momentum data
        using an extended unbinned maximum likelihood fit with goodness of fit and residuals.
        """
        from matplotlib.ticker import MultipleLocator, AutoMinorLocator
        import matplotlib.font_manager as mfm
        import matplotlib as mpl
        
        # Publication-style matplotlib defaults: choose an available serif font
        preferred_serifs = ['DejaVu Serif', 'Times New Roman', 'Times', 'Palatino']
        available_fonts = {f.name for f in mfm.fontManager.ttflist}
        chosen_serif = next((f for f in preferred_serifs if f in available_fonts), 'DejaVu Serif')
        
        mpl.rcParams.update({
            'font.family': 'serif',
            'font.serif': [chosen_serif],
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'axes.titleweight': 'bold',
            'axes.labelweight': 'normal',
            'axes.linewidth': 1.0,
            'figure.dpi': 150,
        })
        
        # Create figure with two subplots: main plot and residual plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        colors = ["#1f77b4", "#ff7f0e"]  # Professional blue and orange
        
        # Store text box y-positions to avoid overlap
        text_y_pos = [0.35, 0.6]
        mean = 0.
        mean_err = 0.
        sigma = 0.
        sigma_err = 0.
        norm = 0.
        for i, data in enumerate(data_list):
            mom_mag_skim = ak.nan_to_none(data)
            mom_mag_skim = ak.drop_none(mom_mag_skim)

            # Define the observable space for the fit
            obs_mom = zfit.Space('x', limits=(95, 115))
            mom_np = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))
            mom_zfit = zfit.Data.from_numpy(array=mom_np, obs=obs_mom)
            
            # Define parameters for the Chebyshev polynomial and yield
            c1 = zfit.Parameter(f"c1_{i}", 0.1, -1, 1)
            c2 = zfit.Parameter(f"c2_{i}", 0.1, -1, 1)
            coeffs = [c1, c2]
            N_RPC = zfit.Parameter(f'N_RPC_{i}', len(mom_np), 100, len(mom_np)*10)

            # Create the extended Chebyshev polynomial PDF
            cheby = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=coeffs, extended=N_RPC)
            
            # Create the extended unbinned negative log-likelihood loss
            nll = zfit.loss.ExtendedUnbinnedNLL(model=cheby, data=mom_zfit)
            
            # Minimize the loss and get the result
            minimizer = zfit.minimize.Minuit()
            result = minimizer.minimize(loss=nll)
            hesse_errors = result.hesse()
            print(result)
            
            # --- Plotting the fit result ---
            
            fit_range = (obs_mom.lower[0, 0], obs_mom.upper[0, 0])
            n_bins = 25
            bin_width = (fit_range[1] - fit_range[0]) / n_bins
            
            # --- Compute histogram first for proper normalization ---
            data_hist_counts, data_bins = np.histogram(mom_np, bins=n_bins, range=fit_range)
            data_bin_center = (data_bins[:-1] + data_bins[1:]) / 2
            total_count = np.sum(data_hist_counts)
            
            # Plot histogram with counts
            ax1.hist(mom_np, color=colors[i], bins=n_bins, range=fit_range, histtype='step', linewidth=2.5, label=labels[i])
            ax1.errorbar(data_bin_center, data_hist_counts, yerr=np.sqrt(data_hist_counts), fmt='.', color=colors[i], capsize=2, markersize=6)
            
            # --- Main plot ---
            
            mom_plot = np.linspace(fit_range[0], fit_range[1], 200).reshape(-1, 1)

            # Scale the PDF to match the counts histogram
            cheby_curve = zfit.run(cheby.pdf(mom_plot) * result.params[N_RPC]['value'] * bin_width)
            ax1.plot(mom_plot.flatten(), cheby_curve.flatten(), color=colors[i], linestyle="-", linewidth=2.5, label=f"{labels[i]} Fit")
            
            # --- Calculate goodness of fit (chi-squared) ---
            fit_values = zfit.run(cheby.pdf(data_bin_center.reshape(-1, 1)) * result.params[N_RPC]['value'] * bin_width)
            chi2 = np.sum(((data_hist_counts - fit_values) ** 2) / (data_hist_counts + 1e-6))
            dof = n_bins - len(coeffs) - 1  # number of bins - number of parameters
            chi2_dof = chi2 / dof if dof > 0 else 0
            
            # --- Residual plot ---
            residuals = (data_hist_counts - fit_values) / np.sqrt(data_hist_counts + 1e-6)
            ax2.errorbar(data_bin_center, residuals, yerr=np.ones_like(residuals), fmt='.', color=colors[i], capsize=2, label=labels[i], markersize=6)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            # --- Create combined text box with all information ---
            param_text = (
                f"{labels[i]}:\n"
                f"c₁ = {result.params[c1]['value']:.4f} ± {hesse_errors[c1]['error']:.4f}\n"
                f"c₂ = {result.params[c2]['value']:.4f} ± {hesse_errors[c2]['error']:.4f}\n"
                f"N_RPC = {result.params[N_RPC]['value']:.0f} ± {hesse_errors[N_RPC]['error']:.0f}\n"
                f"χ²/DOF = {chi2_dof:.2f}"
            )
            
            props = dict(boxstyle='round', facecolor=colors[i], alpha=0.2, edgecolor=colors[i], linewidth=1.5)
            
            # Position the text box
            ax1.text(0.70, text_y_pos[i], param_text, transform=ax1.transAxes,
                     fontsize=9, verticalalignment='top', horizontalalignment='right', bbox=props)
            
            mean = result.params[c1]['value']
            sigma = result.params[c2]['value']
            norm = result.params[N_RPC]['value']
        
        # --- Apply professional styling ---
        ax1.set_ylabel('Events / 3.2 MeV')
        ax1.set_title('Chebyshev Polynomial Fit to RPC Momentum Data')
        #ax1.legend(loc='upper left', frameon=True, shadow=True)
        
        ax2.set_xlabel('Reconstructed Momentum [MeV/c]')
        ax2.set_ylabel('Residuals (σ)')
        ax2.legend(loc='upper left', frameon=True, shadow=True)
        
        # Set major and minor ticks
        ax1.xaxis.set_major_locator(MultipleLocator(2.5))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        # Style the ticks - enable minor ticks on all axes
        ax1.tick_params(which='major', length=6, width=1.0)
        ax1.tick_params(which='minor', length=3, width=0.8)
        ax2.tick_params(which='major', length=6, width=1.0)
        ax2.tick_params(which='minor', length=3, width=0.8)
        
        # Make all spines visible for complete box
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        
        plt.subplots_adjust(left=0.12, right=0.95, top=0.94, bottom=0.1)
        plt.savefig("RPCfit.pdf", dpi=300, bbox_inches='tight')
        plt.show()
        return mean, mean_err, sigma, sigma_err, norm

    def overlay_fit(self, mean, mean_err, sigma, sigma_err, norm, data_list, mc_count):
        """
        Fits a simple Gaussian shape to the reconstructed momentum data
        using an extended unbinned maximum likelihood fit.
        """
        # Create figure with two subplots: main plot and ratio plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        colors = ["black"]
        labels = ["MDS3a"]
        
        # Store text box y-positions to avoid overlap
        text_y_pos = [0.8] 

        for i, data in enumerate(data_list):
            mom_mag_skim = ak.nan_to_none(data)
            mom_mag_skim = ak.drop_none(mom_mag_skim)
            
            true_rpc = mom_mag_skim.mask[(mc_count[i] == 999) ]
            true_rpc = ak.to_numpy((ak.flatten(true_rpc,axis=None)))
            print(true_rpc)
            print(mc_count)

            # Define the observable space for the fit
            obs_mom = zfit.Space('x', limits=(95, 115))
            mom_np = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))
            mom_zfit = zfit.Data.from_numpy(array=true_rpc, obs=obs_mom)
            
            # Define parameters for the Gaussian shape and yield
            mu = zfit.Parameter("mu", mean, floating=False)
            sigma = zfit.Parameter("sigma", sigma, floating=False)
            N_RPC = zfit.Parameter('N_RPC', norm, norm-0.05*norm, norm+0.05*norm)

            # Create the extended Gaussian PDF
            gauss = zfit.pdf.Gauss(obs=obs_mom, mu=mu, sigma=sigma, extended=N_RPC)
            
            # Create the extended unbinned negative log-likelihood loss
            nll = zfit.loss.ExtendedUnbinnedNLL(model=gauss, data=mom_zfit)
            
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

            gauss_curve = zfit.run(gauss.pdf(mom_plot) * result.params[N_RPC]['value'] * bin_width)
            ax1.plot(mom_plot.flatten(), gauss_curve.flatten(), color=colors[i], linestyle="--", label=str(labels[i])+' Fitted Gaussian')
            ax1.grid(True)
            ax1.set_yscale('log')
            data_hist, data_bins, _ = ax1.hist(mom_np, color=colors[i], bins=n_bins, range=fit_range, label=labels[i], histtype='step')
            true_hist, true_bins, _ = ax1.hist(true_rpc, color="orange", bins=n_bins, range=fit_range, label="RPC", histtype='bar')
            data_bin_center = (data_bins[:-1] + data_bins[1:]) / 2
            ax1.errorbar(data_bin_center, data_hist, yerr=np.sqrt(data_hist), fmt='.', color=colors[i], capsize=2)
            
            ax1.set_xlabel('Reconstructed Momentum [MeV/c]')
            ax1.set_ylabel('# of events per bin')
            ax1.legend()
            ax1.set_title('Gaussian Fit to Momentum Data (Extended Unbinned)')
            
            # --- Add text box with fit parameters ---
            param_text = (
                f"Fit parameters for {labels[i]}:\n"
                f"$N_{{RPC}} = {result.params[N_RPC]['value']:.0f} \\pm {hesse_errors[N_RPC]['error']:.2f}$"
            )
            
            props = dict(boxstyle='round', facecolor=colors[i], alpha=0.3)
            
            # Position the text box in the upper left corner of the subplot
            # with an offset for each iteration
            ax1.text(0.4, text_y_pos[i], param_text, transform=ax1.transAxes,
                     fontsize=10, verticalalignment='top', bbox=props)
            
            # --- Ratio plot ---
            
            data_bin_center_2d = data_bin_center.reshape(-1, 1)
            fit_at_bin_center = zfit.run(gauss.pdf(data_bin_center_2d) * result.params[N_RPC]['value'] * bin_width)
            ratio = true_hist / fit_at_bin_center
            
            ax2.errorbar(data_bin_center, ratio, yerr=np.sqrt(data_hist) / fit_at_bin_center, fmt='.', color=colors[i], capsize=2)
            ax2.axhline(1, color='gray', linestyle='--')
            ax2.set_ylabel('Ratio (RPC/Fit)')
            ax2.set_xlabel('Reconstructed Momentum [MeV/c]')
            ax2.set_ylim(0.5, 1.5)
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig("RPCfit.pdf")
        plt.show()
