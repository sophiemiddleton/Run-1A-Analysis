import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from pyutils.pyselect import Select
from pyutils.pyvector import Vector
# Publication-style matplotlib defaults
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
      
    def CR_momentum(self, data_list, labels):
        """
        Plots the reconstructed momentum data and its statistical uncertainties
        without applying a fit or scaling on a single axis. Styled identically 
        to the efficiency formatting method.
        """
        # Create a single-panel figure matching the primary style of fit_eff_momentum
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        colors = ["#1f77b4", "#ff7f0e"]  # Professional Blue and Orange requested
        
        # Consistent layout limits and definition
        fit_range = (95.0, 115.0)
        n_bins = 50
        bin_width = (fit_range[1] - fit_range[0]) / n_bins
        
        for i, data in enumerate(data_list):
            mom_mag_skim = ak.nan_to_none(data)
            mom_mag_skim = ak.drop_none(mom_mag_skim)

            mom_np = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))
            
            # Histogramming raw data (no scaling applied)
            counts_raw, bins = np.histogram(mom_np, bins=n_bins, range=fit_range)
            data_bin_center = (bins[:-1] + bins[1:]) / 2
            
            errors_raw = np.sqrt(counts_raw)
            nonzero_mask = counts_raw > 0
            
            # Plot data points on Main Axis (ax1)
            ax1.errorbar(data_bin_center[nonzero_mask], counts_raw[nonzero_mask], 
                        yerr=errors_raw[nonzero_mask], 
                        fmt='o', color=colors[i], markerfacecolor='white', markeredgecolor=colors[i], 
                        markersize=4, capsize=0, elinewidth=1, label=f'{labels[i]} Stat. unc.')
            
            # Plot step histogram on Main Axis (ax1)
            ax1.hist(mom_np, bins=n_bins, range=fit_range, 
                    color=colors[i], histtype='step', linewidth=1.2, label=labels[i])

        # Uniform layout configuration mirroring the structure of fit_eff_momentum
        ax1.set_ylabel(f'Events / {bin_width:.1f} MeV', fontsize=mpl.rcParams.get('axes.titlesize', 24))
        ax1.set_xlabel('Reconstructed Momentum [MeV/c]', fontsize=mpl.rcParams.get('axes.titlesize', 24))
        
        ax1.text(0.0, 1.02, "Mu2e Simulation", 
                fontsize=20, fontweight='bold', ha='left', va='bottom', transform=ax1.transAxes, zorder=100)
        ax1.legend(fontsize=14)
        
        # Matching minor/major tick configurations exactly
        ax1.minorticks_on()
        ax1.tick_params(direction='in', which='both', top=True, right=True, labelsize=14)
        ax1.yaxis.set_minor_formatter(ticker.NullFormatter()) 
        
        plt.tight_layout()
        plt.savefig("CR_RPCfit.pdf", bbox_inches='tight')
        plt.show()
        
        return 1.0  # Returns a default baseline factor since scaling was removed

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
        fit_range = (475, 550)
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
            N_RPC = zfit.Parameter(f'N_RPC_time_{i}', n_events_raw, 100, n_events_raw * 10)
            c1 = zfit.Parameter(f"c1_{i}", 0.001, -1, 1)
            coeffs = [c1]
            fitcurve = zfit.pdf.Exponential(obs=obs_time, lam=c1, extended=N_RPC)

            # Perform the extended unbinned NLL fit
            nll = zfit.loss.ExtendedUnbinnedNLL(model=fitcurve, data=time_zfit)
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
            fitcurve_curve = zfit.run(fitcurve.ext_pdf(time_plot)) * bin_width * norm_factor
            
            ax1.plot(time_plot.flatten(), fitcurve_curve.flatten(), color=linecolors[i], linestyle="--", linewidth=2.5, label=f'{labels[i]} Fit')
            
            # Calculate Chi2 / DOF
            fit_at_bin_center_raw = zfit.run(fitcurve.ext_pdf(data_bin_center.reshape(-1, 1))) * bin_width
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
                f"$\\lambda = {result.params[c1]['value']:.4f} \\pm {hesse_errors[c1]['error']:.4f}$\n"
                f"$\\chi^2/\\text{{DOF}} = {chi2_dof:.2f}$"
            )
            param_text_elements.append(dataset_text)
            
            last_norm = result.params[N_RPC]['value']

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
        plt.savefig("CR_RPCfit.pdf", bbox_inches='tight')
        plt.show()
        
        return last_norm

    def fit_momentum(self, data_list, labels, opt):
        """
        Fits a Chebyshev polynomial shape to the reconstructed momentum data
        using an extended unbinned maximum likelihood fit with goodness of fit and residuals.
        Styled identically to the efficiency formatting method.
        """
        # Create figure matching structure of fit_eff_momentum
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        colors = ["#1f77b4", "#ff7f0e"]  # Professional Blue and Orange requested
        linecolors = [ "red","green"]
        # Text box lists to compile multi-dataset information cleanly
        param_text_elements = []
        
        # Store return states dynamically
        last_c1, last_c1_err, last_c2, last_c2_err, last_norm = 0., 0., 0., 0., 0.
        for i, data in enumerate(data_list):
                mom_mag_skim = ak.nan_to_none(data)
                mom_mag_skim = ak.drop_none(mom_mag_skim)

                # Define the observable space for the fit
                obs_mom = zfit.Space('x', limits=(97,110))
                mom_np = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))
                mom_zfit = zfit.Data.from_numpy(array=mom_np, obs=obs_mom)
                n_events_raw = len(mom_np)
                # Initialize extended parameter at raw count size
                N_RPC = zfit.Parameter(f'N_RPC_{i}', n_events_raw, 100, n_events_raw * 10)
                if opt == "poly":
                    c1 = zfit.Parameter(f"c1_{i}", 0.1, -1, 1)
                    c2 = zfit.Parameter(f"c2_{i}", 0.1, -1, 1)
                    #c3 = zfit.Parameter(f"c3_{i}", 0.1, -1, 1)
                    #c4 = zfit.Parameter(f"c4_{i}", 0.1, -1, 1)
                    #c5 = zfit.Parameter(f"c5_{i}", 0.1, -1, 1)

                    coeffs = [c1, c2]#, c3, c4 ,c5]
                    fitcurve = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=coeffs, extended=N_RPC)

                if opt == "gaus":
                    mu = zfit.Parameter(f"mu_{i}", 100.0, 95.0, 115.0)
                    sigma = zfit.Parameter(f"sigma_{i}", 2.0, 0.1, 10.0)
                    coeffs = [mu,sigma]
                    fitcurve = zfit.pdf.Gauss(obs=obs_mom, mu=mu, sigma=sigma, extended=N_RPC)

                nll = zfit.loss.ExtendedUnbinnedNLL(model=fitcurve, data=mom_zfit)
                
                minimizer = zfit.minimize.Minuit()
                result = minimizer.minimize(loss=nll)
                hesse_errors = result.hesse()
                print(result)
                
                fit_range = (obs_mom.lower[0, 0], obs_mom.upper[0, 0])
                n_bins = 50
                bin_width = (fit_range[1] - fit_range[0]) / n_bins
                
                target_events = 100000.0
                norm_factor = target_events / n_events_raw  
                

                counts_raw, bins = np.histogram(mom_np, bins=n_bins, range=fit_range)
                data_bin_center = (bins[:-1] + bins[1:]) / 2
                
                counts_norm = counts_raw * norm_factor
                errors_norm = np.sqrt(counts_raw) * norm_factor
                
                nonzero_mask = counts_raw > 0
                
                # Plot markers
                ax1.errorbar(data_bin_center[nonzero_mask], counts_norm[nonzero_mask], 
                            yerr=errors_norm[nonzero_mask], 
                            fmt='o', color=colors[i], markerfacecolor='white', markeredgecolor=colors[i], 
                            markersize=4, capsize=0, elinewidth=1, label=f'{labels[i]} Stat. unc.')
                
                # Plot step histogram using weights
                weights = np.full_like(mom_np, norm_factor)
                ax1.hist(mom_np, bins=n_bins, range=fit_range, weights=weights,
                        color=colors[i], histtype='step', linewidth=1.2, label=labels[i])

                mom_plot = np.linspace(fit_range[0], fit_range[1], 500).reshape(-1, 1)
                fitcurve_curve = zfit.run(fitcurve.ext_pdf(mom_plot)) * bin_width * norm_factor
                
                ax1.plot(mom_plot.flatten(), fitcurve_curve.flatten(), color=linecolors[i], linestyle="--", linewidth=2.5, label=f'{labels[i]} Fit')
                
                fit_at_bin_center_raw = zfit.run(fitcurve.ext_pdf(data_bin_center.reshape(-1, 1))) * bin_width
                chi2 = np.sum(((counts_raw - fit_at_bin_center_raw) ** 2) / (counts_raw + 1e-6))
                dof = n_bins - len(coeffs) - 1
                chi2_dof = chi2 / dof if dof > 0 else 0
                
                residual_norm = counts_norm - (fit_at_bin_center_raw * norm_factor)
                
                # Use np.where to dynamically handle division safely without altering true statistical errors
                pull_errors = np.where(counts_raw > 0, errors_norm, 1.0 * norm_factor)
                pull = residual_norm / pull_errors
                pull_err = np.ones_like(pull)
                
                ax2.errorbar(data_bin_center, pull, yerr=pull_err, 
                            fmt='.', color=colors[i], capsize=0, markersize=6)
                    
                clean_label = labels[i].replace(" ", r"\ ")
                if opt == "poly":
                    dataset_text = (
                        f"$\\mathbf{{{clean_label}}}:$\n"
                        #f"$N_{{RPC}} = {result.params[N_RPC]['value']:.0f} \\pm {hesse_errors[N_RPC]['error']:.1f}$\n"
                        f"$c_{{1}} = {result.params[c1]['value']:.3f} \\pm {hesse_errors[c1]['error']:.4f}$\n"
                        f"$c_{{2}} = {result.params[c2]['value']:.3f} \\pm {hesse_errors[c2]['error']:.4f}$\n"
                        #f"$c_{{3}} = {result.params[c1]['value']:.3f} \\pm {hesse_errors[c3]['error']:.4f}$\n"
                        #f"$c_{{4}} = {result.params[c4]['value']:.3f} \\pm {hesse_errors[c4]['error']:.4f}$\n"
                        #f"$c_{{5}} = {result.params[c5]['value']:.3f} \\pm {hesse_errors[c5]['error']:.4f}$"
                        f"$\\chi^2/\\text{{DOF}} = {chi2_dof:.2f}$"
                    )
                if opt == "gaus":
                    dataset_text = (
                        f"$\\mathbf{{{clean_label}}}:$\n"
                        #f"$N_{{RPC}} = {result.params[N_RPC]['value']:.0f} \\pm {hesse_errors[N_RPC]['error']:.1f}$\n"
                        f"$\mu = {result.params[mu]['value']:.3f} \\pm {hesse_errors[mu]['error']:.4f}$\n"
                        f"$\sigma = {result.params[sigma]['value']:.3f} \\pm {hesse_errors[sigma]['error']:.4f}$\n"
                        f"$\\chi^2/\\text{{DOF}} = {chi2_dof:.2f}$"
                    )
                param_text_elements.append(dataset_text)
                
                last_norm = result.params[N_RPC]['value']
        # Text box presentation matching facecolor, layout and alpha weight values of function 1
        combined_param_text = "\n\n".join(param_text_elements)
        props = dict(boxstyle='round,pad=0.5', facecolor='lightgrey', alpha=0.75, edgecolor='gray')
        ax1.text(0.95, 0.55, combined_param_text, transform=ax1.transAxes,
            fontsize=16, horizontalalignment='right', verticalalignment='top', bbox=props)
        
        # Uniform layout configuration mirroring the structure of fit_eff_momentum
        ax1.set_ylabel(f'Events / {bin_width:.1f} MeV', fontsize=mpl.rcParams.get('axes.titlesize', 24))
        ax1.text(0.0, 1.02, "Mu2e Simulation", 
                fontsize=20, fontweight='bold', ha='left', va='bottom', transform=ax1.transAxes, zorder=100)
        ax1.legend(fontsize=16, loc='lower left')
        
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_ylabel(r'Pull [$\sigma$]', fontsize=mpl.rcParams.get('axes.titlesize', 24))
        ax2.set_xlabel('Reconstructed Momentum [MeV/c]', fontsize=mpl.rcParams.get('axes.titlesize', 24))
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
        plt.savefig("RPCfit.pdf", bbox_inches='tight')
        plt.show()
        
        return last_norm

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
