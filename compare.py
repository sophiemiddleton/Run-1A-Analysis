import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import awkward as ak

from pyutils.pyselect import Select
from pyutils.pyvector import Vector

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




 


  

