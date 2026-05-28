import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import awkward as ak
import pickle as pkl

from pyutils.pyselect import Select
from pyutils.pyvector import Vector
from helper import make_HistogramPDF

# Publication-style matplotlib defaults: choose an available serif font
import matplotlib.font_manager as mfm
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
  'grid.linewidth': 0.5,
  'figure.dpi': 150,
})



class Compare():
    """Class to conduct comparisons between cut or data sets
    """
    def __init__(self ):
      """
      """
      
      # Custom prefix for log messages from this processor
      self.print_prefix = "[Compare] "
      print(f"{self.print_prefix}Initialised")

    def plot_variable(self, val_overlay, val_label, filenames, lo, hi, cut_lo, cut_hi, mc_count, columns=[], nbins = 25, use_log=False, density=False, residuals = False):
      """
      Plots distributions of the given parameter (val), splitting by process code

      Args:
          val : list of values e.g. rmax
          val_label : text formated value name e.g. "rmax"
          lo : plot range lower bound
          hi : plot range upper bound
          cut_lo : lower cut choice
          cut_hi : upper cut choice
          mc_counts : list of process codes

      Returns:
          plots saved as pdfs
      """
      sets = []

      if residuals:
        fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])
      else:
        fig, (ax1) = plt.subplots(1,1)

      # publication-friendly palette (matplotlib 'tab10' style, 8 colors)
      cols = ['#1f77b4', "#ffe30e", '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', "#ff8000"]
      labs = ['Cosmic','int. RPC','ext. RPC','int. RMC','ext. RMC','IPA Decays','DIO', 'Signal']
      styles = ['bar','step','step']
      lines=["","-","--"]
      alphas = [1,1,1]
      
      for i, val in enumerate(val_overlay):
        val = ak.drop_none(val)
        val_signal = val.mask[(mc_count[i] == 168) | (mc_count[i] == 176)]
        val_signal = np.array(ak.flatten(val_signal, axis=None))
        val_cosmics = val.mask[mc_count[i] == -1]
        val_cosmics = np.array(ak.flatten(val_cosmics, axis=None))
        val_dio = val.mask[mc_count[i] == 166]
        val_dio = np.array(ak.flatten(val_dio, axis=None))
        val_erpc = val.mask[mc_count[i] == 178]
        val_erpc = np.array(ak.flatten(val_erpc,axis=None))
        val_irpc = val.mask[mc_count[i] == 179]
        val_irpc = np.array(ak.flatten(val_irpc,axis=None))
        val_ermc = val.mask[mc_count[i] == 171]
        val_ermc = np.array(ak.flatten(val_ermc,axis=None))
        val_irmc = val.mask[mc_count[i] == 172]
        val_irmc = np.array(ak.flatten(val_irmc,axis=None))
        val_ipa = val.mask[mc_count[i] == 0]
        val_ipa = np.array(ak.flatten(val_ipa,axis=None))
        sets.append([val_cosmics,val_irpc,val_erpc,val_irmc,val_ermc, val_ipa, val_dio, val_signal])
      bin_centers = []
      bin_contents = []
      bin_errors = []
      combined_data_per_dataset = []
      
      for i in range(0,len(sets)):
        if use_log:
          ax1.set_yscale('log')
        dummy_handle = ax1.plot([], marker="",color='white', label=columns[i])
        n, bins, patch = ax1.hist(sets[i], range=(lo,hi), color=cols, label=labs, bins=nbins, histtype=styles[i], alpha=alphas[i], stacked=True, density=density)#, edgecolor='black', linewidth=0.8)
        # ensure black edge lines on returned patches (works for Rectangle and Patch collections)
        try:
          for p in patch:
            try:
              p.set_edgecolor('black')
            except Exception:
              pass
            try:
              p.set_linewidth(0.8)
            except Exception:
              pass
        except Exception:
          pass
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_contents.append(n)  # Store full histogram for all 8 components
        
        # Combine all process codes for full sample overlay
        combined_array = np.concatenate([sets[i][j] for j in range(len(sets[i]))])
        combined_data_per_dataset.append(combined_array)
      
      # Plot combined sample data points with error bars on top of histograms
      bin_centers = 0.5 * (bins[:-1] + bins[1:])
      for i, combined_data in enumerate(combined_data_per_dataset):
        n_combined, _ = np.histogram(combined_data, range=(lo,hi), bins=bins)
        # Calculate Poisson errors
        errors_combined = np.sqrt(n_combined)
        # Avoid plotting zeros in log scale
        mask_nonzero = n_combined > 0
        ax1.errorbar(bin_centers[mask_nonzero], n_combined[mask_nonzero], 
                    yerr=errors_combined[mask_nonzero], fmt='o', 
                    capsize=3, capthick=1.5, markersize=4, color='black', 
                    elinewidth=1, label=f'{columns[i]} (full sample)' if i == 0 else '')

      ax1.set_xlabel(str(val_label))
      ax1.set_xlim(lo,hi)
      #ax1.set_ylim(0,100)
      # draw cuts
      
      if(len(cut_lo) !=0):
        ax1.plot(cut_lo, [0,40], color='black', linestyle='--') #110
      if(len(cut_hi) !=0):
        ax1.plot(cut_hi, [0,40], color='black', linestyle='--')
      
        # place legend off the right side of the axes to avoid overlapping data
        # make only the first legend label bold (dummy handle)
        leg = ax1.legend(ncol=len(columns), loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        texts = leg.get_texts()
        if texts:
          texts[0].set_fontweight('bold')

      if residuals:
        # residuals: S/sqrt(B) where S=signal (component 7), B=sum of all other components
        signal = bin_contents[0][7]  # Signal counts from first dataset
        background = np.sum(bin_contents[0][:7], axis=0)  # Sum of all background components
        
        # Calculate S/sqrt(B+1), adding 1 to regularize pure signal regions
        residuals = signal / np.sqrt(np.maximum(background, 1))
        
        # Error propagation: simplified for S/sqrt(B+1)
        # where denom = sqrt(B+1), d(S/sqrt(B+1))/dS = 1/denom
        denom = np.sqrt(np.maximum(background, 1))
        err_S = 1.0 / denom
        with np.errstate(divide='ignore', invalid='ignore'):
          err_B = -signal / (2.0 * denom**3)
          residuals_err = np.sqrt((err_S**2 * signal) + (err_B**2 * background))
        residuals_err = np.nan_to_num(residuals_err)
        
        ax2.errorbar(bin_centers, residuals, yerr=residuals_err, fmt='.', color='red', capsize=3, label='Error Bars')
        ax2.set_xlabel(str(val_label))
        ax2.set_xlim(lo,hi)
        ax2.set_ylabel(r"S / $\sqrt{B}$")
      # leave room on the right for the legend when saving
      plt.tight_layout(rect=[0, 0, 0.87, 1])
      # place the canvas label anchored to the figure top-left so it remains there
      legend_fs = mpl.rcParams.get('legend.fontsize', 12)
      fig.text(0.01, 0.995, "Mu2e Simulation", fontsize=legend_fs, fontweight='bold', fontstyle='italic',
           ha='left', va='top', transform=fig.transFigure, zorder=100)
      if(len(val_signal) != 0): 
        ax1.text(0.92, 0.97, r"$R_{\mu e} = 1 \times 10^{-13}$", fontsize=legend_fs, 
           ha='right', va='top', transform=ax1.transAxes, zorder=100,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgrey', edgecolor='black', alpha=0.8))
      plt.savefig(str(filenames)+"_selection.png", bbox_inches='tight')
      plt.show()
      
    

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
      fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])
      sets=[]
      cols = ['#ffff00']
      labs = ['signal']
      styles = ['bar','step']
      lines=["","-"]
      alphas = [0.2,1]
      for i, val in enumerate(val_overlay):
        val = ak.drop_none(val)
        val = np.array(ak.flatten(val,axis=None))
        sets.append([val])
      bin_centers = []
      bin_contents = []
      bin_errors = []
      for i in range(0,len(sets)):
        ax1.set_yscale('log')
        dummy_handle = ax1.plot([], marker="",color='white', label=columns[i])
        n, bins, patch = ax1.hist(sets[i],range=(lo,hi), color=cols, label=labs, bins=50, histtype=styles[i], alpha=alphas[i], stacked=True, density=density)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        bin_contents.append((n))

      ax1.set_xlabel(str(val_label))
      ax1.set_xlim(lo,hi)
      ax1.legend(ncol=len(columns))

      # residuals for signal only
      residuals = []
      residuals_err = []

      residuals = (bin_contents[0] - bin_contents[1])/np.sqrt(bin_contents[1])
      residuals_err= np.sqrt(np.sqrt(bin_contents[1])*np.sqrt(bin_contents[1])  + np.sqrt(bin_contents[0])*np.sqrt(bin_contents[0]))/np.sqrt(bin_contents[1])

      ax2.errorbar(bin_centers, residuals, yerr=residuals_err, fmt='o', color='red', capsize=3, label='Error Bars')
      ax2.set_xlabel(str(val_label))
      ax2.set_ylabel("(Old - New)/Sigma")
      ax2.set_xlim(lo,hi)
      
      plt.savefig(str(filenames)+"_resolution.png")
      plt.show()
      
    def plot_particle_counts(self, mc_counts, columns, plot_prefix=""):
      """
      Plot a grouped horizontal bar chart comparing particle type counts
      between different datasets and adds percentage change labels.
      
      Args:
          mc_counts : list of arrays/lists of particle codes (one per dataset)
          columns   : labels for datasets (e.g., ["old_cuts", "no_cuts"])
          plot_prefix : string prefix for output filename (e.g., "eminus_" or "eplus_")
      """
      # Map PDG/startCodes to categories
      labels = ["DIO", "IPA", "CEMLL", "CEPLL", "eRPC", "iRPC", "eRMC", "iRMC", "Cosmic", "Other"]
      pdg_codes = [166, 114, 168, 176, 178, 179, 171, 172, -1, -2]
      num_categories = len(pdg_codes)
      num_datasets = len(mc_counts)

      # Use NumPy's vectorized operations for efficient counting
      datasets = np.zeros((num_datasets, num_categories), dtype=int)
      for i, mc in enumerate(mc_counts):
          if mc is not None and len(mc) > 0:
              mc_array = np.array(mc)
              for j, code in enumerate(pdg_codes):
                  datasets[i, j] = np.sum(mc_array == code)
      
      # Check that there are at least two datasets for a comparison
      if num_datasets < 2:
          print("Not enough datasets for percentage change calculation. Plotting without it.")
          # Re-run the original plotting logic if needed
          # ...
          return

      # Calculate percentage change based on the first dataset
      # Avoids division by zero by setting change to 0 if the original value is 0
      with np.errstate(divide='ignore', invalid='ignore'):
          old_counts = datasets[0]
          new_counts = datasets[1]
          percent_changes = ((new_counts - old_counts) / old_counts) * 100
          percent_changes[np.isinf(percent_changes) | np.isnan(percent_changes)] = 0

      # Plot grouped horizontal bars
      y = np.arange(num_categories)
      bar_height = 0.8 / num_datasets
      
      fig, ax = plt.subplots(figsize=(12, 6))

      bars = []
      for i, data in enumerate(datasets):
          bars.append(ax.barh(y + i * bar_height, data, height=bar_height, label=columns[i], xerr=np.sqrt(data), capsize=4, error_kw={'elinewidth': 1.5}))
      
      # Add percentage change labels to the second set of bars
      for i, bar in enumerate(bars[1]): # Iterate over the bars of the second dataset
          # Get the percentage change for the corresponding category
          change = percent_changes[i]
          
          # Format the label string
          label_text = f'{change:.1f}%'
          
          # Choose color based on whether change is positive or negative
          color = 'red' if change < 0 else 'green'
          
          # Position the label with offset to avoid overlap with error bars
          # Add offset equal to ~1.5x the error bar width to ensure no overlap
          error_val = np.sqrt(datasets[1, i])
          offset = error_val * 1.5
          ax.text(
              bar.get_width() + offset, 
              bar.get_y() + bar.get_height() / 2, 
              label_text, 
              ha='left', 
              va='center',
              color=color,
              fontsize=8
          )

      # Center the y-tick labels correctly
      ax.set_yticks(y + bar_height * (num_datasets - 1) / 2)
      ax.set_yticklabels(labels)
      ax.set_xlabel("Event counts")
      ax.set_title("Comparison of particle types with Percentage Change")
      #ax.set_xlim(0, 60000)
      ax.legend()
      
      plt.tight_layout()
      filename = f"{plot_prefix}particle_comparison_with_changes.png" if plot_prefix else "particle_comparison_with_changes.png"
      plt.savefig(filename)
      plt.show()

    def convolve_with_rle(self, reco_data, theory_pdf, res_pdf, loss_pdf,
                         mom_range=(90, 120), nbins=100, label="Theory ⊗ RLE",
                         plot_title="Reco Momentum with Theory Convolution", output_file=None):
        """
        Overlay reconstructed momentum data with theory convolved with resolution and loss PDFs.
        
        Uses zfit ConvPDF to convolve theory with combined resolution+loss smearing:
        reco_predicted = theory ⊗ (resolution ⊗ loss)
        
        Args:
            reco_data (np.array): Reconstructed momentum data (raw values, not histogram)
            theory_pdf (zfit.pdf.BasePDF): Theory momentum distribution (e.g., Chebyshev, Gaussian)
            res_pdf (zfit.pdf.BasePDF): Resolution distribution PDF (reco - true)
            loss_pdf (zfit.pdf.BasePDF): Energy loss distribution PDF (true - gen)
            mom_range (tuple): (min, max) momentum range for plotting
            nbins (int): Number of bins for data histogram
            label (str): Label for the theoretical curve
            plot_title (str): Title for the plot
            output_file (str): Optional path to save plot
            
        Returns:
            fig, ax: Matplotlib figure and axes objects
        """
        import zfit
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot reco data histogram
        reco_counts, reco_bins, _ = ax.hist(reco_data, bins=nbins, range=mom_range,
                                            label='Reco data', histtype='step',
                                            color='blue', linewidth=2)
        reco_bin_centers = (reco_bins[:-1] + reco_bins[1:]) / 2
        bin_width = reco_bins[1] - reco_bins[0]
        ax.errorbar(reco_bin_centers, reco_counts, yerr=np.sqrt(reco_counts),
                   fmt='.', color='black', capsize=2, elinewidth=1)
        
        # Create momentum grid for plotting
        mom_plot = np.linspace(mom_range[0], mom_range[1], 200).reshape(-1, 1)
        
        try:
            # Pyfitter approach: ONE convolution only - theory ⊗ combined_kernel
            # Key: convert theory to obs_gen ('x' space with wide bounds), then convolve with obs_res kernel
            print(f"[Compare] Setting up FFT convolution (pyfitter single kernel)...", flush=True)
            
            # Observable space for theory PDF (wide bounds)
            obs_gen = zfit.Space('x', limits=mom_range)      # Theory on 'x' with wide bounds
            
            print(f"[Compare] obs_gen: {obs_gen}", flush=True)
            
            # Number of bins for FFT
            nbins_gen = 200
            
            # Convert theory_pdf from 'mom' observable to obs_gen ('x' with wide bounds)
            print(f"[Compare] Converting theory PDF to obs_gen ('x') space...", flush=True)
            
            # Evaluate theory PDF on obs_gen bounds to create histogram representation
            theory_x_vals = np.linspace(float(obs_gen.v1.lower), float(obs_gen.v1.upper), nbins_gen + 1)
            theory_x_centers = theory_x_vals[:-1]
            
            # Evaluate theory PDF at bin centers to get probability density
            theory_x_evals = zfit.run(theory_pdf.pdf(theory_x_centers.reshape(-1, 1))).flatten()
            
            # Create histogram PDF class and instantiate on obs_gen ('x')
            HistPDF = make_HistogramPDF(theory_x_evals, theory_x_vals)
            theory_pdf_gen = HistPDF(obs=obs_gen)
            print(f"[Compare] Theory PDF converted to histogram on obs_gen ('x')", flush=True)
            
            # Get kernel parameters (if they exist - histogram PDFs may not have them)
            print(f"[Compare] res_pdf type: {type(res_pdf)}", flush=True)
            print(f"[Compare] loss_pdf type: {type(loss_pdf)}", flush=True)
            
            print(f"[Compare] Note: Using histogram-based kernels (capture actual GCB and truncated Landau shapes)", flush=True)
            
            # ===== STEP 1: theory ⊗ loss → p_true distribution =====
            print(f"[Compare] STEP 1: Convolving theory ⊗ loss...", flush=True)
            
            # Extract actual loss kernel bounds from loss_pdf observable
            loss_obs = loss_pdf.space
            lower_loss = float(loss_obs.v1.lower)
            upper_loss = float(loss_obs.v1.upper)
            obs_loss = zfit.Space('x', limits=(lower_loss, upper_loss))
            
            lower_gen = float(obs_gen.v1.lower)
            upper_gen = float(obs_gen.v1.upper)
            
            # Observable space for p_true (theory ⊗ loss)
            # Blend subtraction and addition formulas for correct bounds
            obs_p_true = zfit.Space('x', 
                                    (lower_gen - upper_loss + lower_gen + lower_loss) / 2.0,
                                    (upper_gen - lower_loss + upper_gen + upper_loss) / 2.0)
            obs_full_loss = zfit.Space('x', 
                                       (lower_gen - upper_loss + lower_gen + lower_loss) / 2.0,
                                       (upper_gen - lower_loss + upper_gen + upper_loss) / 2.0)
            
            print(f"[Compare]   obs_loss: [{lower_loss:.1f}, {upper_loss:.1f}]", flush=True)
            print(f"[Compare]   obs_p_true: [{float(obs_p_true.v1.lower):.1f}, {float(obs_p_true.v1.upper):.1f}]", flush=True)
            
            # First convolution: theory_pdf_gen ⊗ loss_pdf
            # loss_pdf is already a histogram on obs_loss, use directly
            pdf_p_true = zfit.pdf.FFTConvPDFV1(
                func=theory_pdf_gen,
                kernel=loss_pdf,
                n=50,
                obs=obs_p_true,
                norm=obs_full_loss
            )
            print(f"[Compare]   ✓ pdf_p_true created (theory ⊗ loss)", flush=True)
            
            # ===== STEP 2: p_true ⊗ resolution → p_reco distribution =====
            print(f"[Compare] STEP 2: Convolving p_true ⊗ resolution...", flush=True)
            
            # Extract actual resolution kernel bounds from res_pdf observable
            res_obs = res_pdf.space
            lower_res = float(res_obs.v1.lower)
            upper_res = float(res_obs.v1.upper)
            obs_res = zfit.Space('x', limits=(lower_res, upper_res))
            
            lower_p_true = float(obs_p_true.v1.lower)
            upper_p_true = float(obs_p_true.v1.upper)
            
            # Observable space for p_reco (p_true ⊗ resolution)
            # Blend subtraction and addition formulas for correct bounds
            obs_p_reco = zfit.Space('x', 
                                    (lower_p_true - upper_res + lower_p_true + lower_res) / 2.0,
                                    (upper_p_true - lower_res + upper_p_true + upper_res) / 2.0)
            obs_full_res = zfit.Space('x', 
                                      (lower_p_true - upper_res + lower_p_true + lower_res) / 2.0,
                                      (upper_p_true - lower_res + upper_p_true + upper_res) / 2.0)
            
            print(f"[Compare]   obs_res: [{lower_res:.1f}, {upper_res:.1f}]", flush=True)
            print(f"[Compare]   obs_p_reco: [{float(obs_p_reco.v1.lower):.1f}, {float(obs_p_reco.v1.upper):.1f}]", flush=True)
            
            # Second convolution: pdf_p_true ⊗ res_pdf
            # res_pdf is already a histogram on obs_res, use directly
            pdf_p_reco = zfit.pdf.FFTConvPDFV1(
                func=pdf_p_true,
                kernel=res_pdf,
                n=50,
                obs=obs_p_reco,
                norm=obs_full_res
            )
            print(f"[Compare]   ✓ pdf_p_reco created (p_true ⊗ resolution)", flush=True)
            
            # Use the final convolved PDF for evaluation
            convolved_pdf = pdf_p_reco
            obs_conv = obs_p_reco
            
            # Evaluate on final p_reco bounds
            conv_lower = float(obs_p_reco.v1.lower)
            conv_upper = float(obs_p_reco.v1.upper)
            mom_plot_conv = np.linspace(conv_lower, conv_upper, 200).reshape(-1, 1)
            print(f"[Compare] Evaluating final pdf_p_reco on [{conv_lower:.1f}, {conv_upper:.1f}]...", flush=True)
            convolved_vals = zfit.run(convolved_pdf.pdf(mom_plot_conv)).flatten()
            print(f"[Compare] Final convolved: min={np.min(convolved_vals):.6e}, max={np.max(convolved_vals):.6e}", flush=True)
            
            # Apply efficiency correction if available
            try:
                import os
                eff_path = "RLE/common/efficiency.pkl"
                if os.path.exists(eff_path):
                    with open(eff_path, 'rb') as f:
                        h_eff, eff_edges = pkl.load(f)
                    
                    # Interpolate efficiency at evaluation points
                    eff_bin_centers = (eff_edges[:-1] + eff_edges[1:]) / 2
                    eff_interp = np.interp(mom_plot_conv.flatten(), eff_bin_centers, h_eff, left=0, right=0)
                    convolved_vals = convolved_vals * eff_interp
                    print(f"[Compare] Applied efficiency correction: eff range [{np.min(eff_interp[eff_interp>0]):.4f}, {np.max(eff_interp):.4f}]", flush=True)
                else:
                    print(f"[Compare] No efficiency.pkl found at {eff_path}, skipping efficiency correction", flush=True)
            except Exception as e:
                print(f"[Compare] Warning: Failed to apply efficiency correction: {e}", flush=True)
            
            # Check for valid values
            if np.any(np.isnan(convolved_vals)):
                nan_count = np.sum(np.isnan(convolved_vals))
                print(f"[Compare] WARNING: {nan_count}/{len(convolved_vals)} convolved values are NaN", flush=True)
                raise ValueError("Convolution produced NaN")
            
            # Scale to match data integral
            data_integral = np.sum(reco_counts)
            convolved_integral = np.trapz(convolved_vals, mom_plot.flatten())
            print(f"[Compare] Data integral: {data_integral}, Convolved integral: {convolved_integral:.6e}", flush=True)
            
            if convolved_integral > 0 and not np.isnan(convolved_integral):
                convolved_scaled = convolved_vals * (data_integral / convolved_integral) * bin_width
                ax.plot(mom_plot.flatten(), convolved_scaled, 'r-', linewidth=2.5, label=label)
                print(f"[Compare] Successfully plotted convolved PDF", flush=True)
            else:
                print(f"[Compare] Invalid convolved integral, using theory fallback", flush=True)
                raise ValueError(f"Invalid integral: {convolved_integral}")
            
        except Exception as e:
            print(f"[Compare] Convolution failed ({e}), plotting theory instead", flush=True)
            import traceback
            traceback.print_exc()
            
            try:
                # Fallback: plot theory only
                theory_vals = zfit.run(theory_pdf.pdf(mom_plot)).flatten()
                theory_integral = np.trapz(theory_vals, mom_plot.flatten())
                data_integral = np.sum(reco_counts)
                if theory_integral > 0:
                    theory_scaled = theory_vals * (data_integral / theory_integral) * bin_width
                    ax.plot(mom_plot.flatten(), theory_scaled, 'r--', linewidth=2, label=label + " (no convolution)")
                    print(f"[Compare] Plotted theory PDF as fallback", flush=True)
            except Exception as e2:
                print(f"[Compare] Theory fallback also failed: {e2}", flush=True)
        
        # Styling
        ax.set_xlabel("Momentum [MeV/c]")
        ax.set_ylabel("Events per bin")
        ax.set_title(plot_title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            print(f"[Compare] Saved plot to {output_file}")
        
        return fig, ax



