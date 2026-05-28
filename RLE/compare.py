import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import tensorflow as tf
import math

from pyutils.pyselect import Select
from pyutils.pyvector import Vector

import zfit
import zfit.z.numpy as znp

# 1. Define the custom PDF based on erf
class CustomErfPDF(zfit.pdf.ZPDF):
    """A custom PDF based on a scaled and shifted error function."""
    _PARAMS = ['mu', 'sigma']
    _N_OBS = 1

    def _unnormalized_pdf(self, x):
        """The unnormalized PDF is the derivative of the error function."""
        x = zfit.z.unstack_x(x)
        mu = self.params['mu']
        sigma = self.params['sigma']
        
        z = (x - mu) / (sigma * znp.sqrt(2.0))
        
        return znp.exp(-z**2)


class CustomLandau(zfit.pdf.ZPDF):
    # Specify the names of the parameters
    _PARAMS = ['mpv', 'width']
    _N_OBS = 1

    # Implement the unnormalized PDF calculation
    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x)
        
        mpv = self.params['mpv']
        width = self.params['width']

        z = (x - mpv) / width
        return znp.exp(-0.5 * (z + znp.exp(-z))) / width
        

class trunc_landau(zfit.pdf.ZPDF):
    _N_OBS = 1
    _PARAMS = ['loc','scale']

    bounds = [-5.1328125, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]
        
    P = [[6.26864481454444278646e-1, 5.10647753508714204745e-1, 1.98551443303285119497e-1, 4.71644854289800143386e-2, 7.71285919105951697285e-3,   8.93551020612017939395e-4,  6.97020145401946303751e-5,   4.17249760274638104772e-6,  7.73502439313710606153e-12],
         [6.31126317567898819465e-1, 5.28493759149515726917e-1, 3.28301410420682938866e-1, 1.31682639578153092699e-1, 3.86573798047656547423e-2,   7.77797337463414935830e-3,  9.97883658430364658707e-4,   6.05131104440018116255e-5],
         [6.50763682207511020789e-3, 5.73790055136022120436e-2, 2.22375662069496257066e-1, 4.92288611166073916396e-1, 6.74552077334695078716e-1,   5.75057550963763663751e-1,  2.85690710485234671432e-1,   6.73776735655426117231e-2,  3.80321995712675339999e-3, 1.09503400950148681072e-3, -9.00045301380982997382e-5], 
         [2.21762208692280384264e-1, 7.10041055270973473923e-1, 8.66556480457430718380e-1, 4.78718713740071686348e-1, 1.03670563650247405820e-1,   4.31699263023057628473e-3,  1.72029926636215817416e-3,  -2.76271972015177236271e-4,  1.89483904652983701680e-5],
         [2.62240126375351657026e-1, 3.37943593381366824691e-1, 1.53537606095123787618e-1, 3.01423783265555668011e-2, 2.66982581491576132363e-3,  -1.57344124519315009970e-5,  3.46237168332264544791e-7,   2.54512306953704347532e-8],
         [1.63531240868022603476e-1, 1.42818648212508067982e-1, 4.95816076364679661943e-2, 8.59234710489723831273e-3, 5.76649181954629544285e-4,  -5.66279925274108366994e-7],
         [9.55242261334771588094e-2, 6.66529732353979943139e-2, 1.80958840194356287100e-2, 2.34205449064047793618e-3, 1.16859089123286557482e-4,  -1.48761065213531458940e-7,  4.37245276130361710865e-9,  -8.10479404400603805292e-11],
         [3.83643820409470770350e-2, 1.97555000044256883088e-2, 3.71748668368617282698e-3, 3.04022677703754827113e-4, 8.76328889784070114569e-6,  -3.34900379044743745961e-9,  5.36581791174380716937e-11, -5.50656207669255770963e-13],
         [1.12656323880287532947e-2, 2.87311140580416132088e-3, 2.61788674390925516376e-4, 9.74096895307400300508e-6, 1.19317564431052244154e-7,  -6.99543778035110375565e-12, 4.33383971045699197233e-14, -1.75185581239955717728e-16],
         [2.83847488747490686627e-3, 4.95641151588714788287e-4, 2.79159792287747766415e-5, 5.93951761884139733619e-7, 3.89602689555407749477e-9,  -4.86595415551823027835e-14, 9.68524606019510324447e-17],
         [6.85767880395157523315e-4, 4.08288098461672797376e-5, 8.10640732723079320426e-7, 6.10891161505083972565e-9, 1.37951861368789813737e-11, -1.25906441382637535543e-17],
         ]

    Q = [[1., 8.15124079722976906223e-1, 3.16755852188961901369e-1, 7.52819418000330690962e-2, 1.23053506566779662890e-2, 1.42615273721494498141e-3, 1.11211928184477279204e-4, 6.65899898061789485757e-6],
         [1., 8.47781139548258655981e-1, 5.21797290075642096762e-1, 2.10939174293308469446e-1, 6.14856955543769263502e-2, 1.24427885618560158811e-2, 1.58973907730896566627e-3, 9.66647686344466292608e-5],
         [1., 1.07919389927659014373e0,  2.56142472873207168042e0,  1.68357271228504881003e0,  2.23924151033591770613e0,  9.05629695159584880257e-1, 8.94372028246671579022e-1, 1.98616842716090037437e-1, 1.70142519339469434183e-1, 1.46288923980509020713e-2, 1.26171654901120724762e-2],
         [1., 2.18155995697310361937e0,  2.53173077603836285217e0,  1.91802065831309251416e0,  9.94481663032480077373e-1, 3.72037148486473195054e-1, 8.85828240211801048938e-2, 1.41354784778520560313e-2],
         [1., 1.61596691542333069131e0,  1.31560197919990191004e0,  6.37865139714920275881e-1, 1.99051021258743986875e-1, 3.73788085017437528274e-2, 3.72580876403774116752e-3],
         [1., 1.41478104966077351483e0,  9.41180365857002724714e-1, 3.65084346985789448244e-1, 8.77396986274371571301e-2, 1.24233749817860139205e-2, 8.57476298543168142524e-4],
         [1., 1.21670723402658089612e0,  6.58224466688607822769e-1, 2.00828142796698077403e-1, 3.64962053761472303153e-2, 3.76034152661165826061e-3, 1.74723754509505656326e-4],
         [1., 9.09290785092251223006e-1, 3.49404120360701349529e-1, 7.23730835206014275634e-2, 8.47875744543245845354e-3, 5.28021165718081084884e-4, 1.33941126695887244822e-5],
         [1., 4.94430267268436822392e-1, 1.00370783567964448346e-1, 1.05989564733662652696e-2, 6.04942184472254239897e-4, 1.72741008294864428917e-5, 1.85398104367945191152e-7],
         [1., 3.01847536766892219351e-1, 3.63152433272831196527e-2, 2.20938897517130866817e-3, 7.05424834024833384294e-5, 1.09010608366510938768e-6, 6.08711307451776092405e-9],
         [1., 1.23722380864018634550e-1, 6.05800403141772433527e-3, 1.47809654123655473551e-4, 1.84909364620926802201e-6, 1.08158235309005492372e-8, 2.16335841791921214702e-11]
         ]
    
    def polyval(self, p, t):
        tot = p[-1]*tf.ones_like(t)
        for i in range(len(p)-1):
            tot = tot*t + p[-2-i]
        return tot
    
    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x)
        loc   = self.params['loc']
        scale = self.params['scale']
        y = (loc-x)/scale
        result = tf.zeros_like(y)

        for i in range(11):
            t = self.bounds[i+1]-y if i < 2 else y-self.bounds[i]
            sigma = tf.exp(-y * math.pi / 2 - 1.45158270528945486473)
            s = tf.math.exp(-sigma)*tf.math.sqrt(sigma) if i < 2 else 1.
            result = result + s * tf.where(((y >= self.bounds[i]) & (y < self.bounds[i+1])), self.polyval(p=self.P[i], t=t) / self.polyval(p=self.Q[i], t=t), 0)
            
        return result
        
        
        
class Compare():
    """Class to conduct comparisons between cut or data sets
    """
    def __init__(self ):
      """
      """
      
      # Custom prefix for log messages from this processor
      self.print_prefix = "[Compare] "
      print(f"{self.print_prefix}Initialised")

    def plot_variable(self, val_overlay, val_label, filenames, lo, hi, cut_lo, cut_hi, mc_count, columns=[], nbins = 50, density=True):
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
      rpc = ["e+", "e-"]
      code = [173,174]
      fig, ax1 = plt.subplots(1,1)
      #cols = ['black']
      labs = ['flat']
      styles = ['step']
      lines=["","-","--"]
      alphas = [0.2,1,1]
      text_contents = []
      for i, val in enumerate(val_overlay):
        val = ak.drop_none(val)

        val_test = val.mask[mc_count[i] == int(code[i])]
        val_test = np.array(ak.flatten(val_test,axis=None))

        mean_val = np.mean(val_test)
        std_dev = np.std(val_test)
        text_contents.append(str(rpc[i])+ f"Mean: {mean_val:.2f}\nStd Dev: {std_dev:.2f}")
        sets.append([val_test])
      for i in range(0,len(sets)):
        ax1.set_yscale('log')
        dummy_handle = ax1.plot([], marker="",color='white', label=columns[i])
        n, bins, patch = ax1.hist(sets[i],range=(lo,hi), label=labs, bins=nbins, histtype='step', stacked=True, density=density)


      ax1.set_xlabel(str(val_label))
      ax1.set_xlim(lo,hi)
 
      ax1.legend(ncol=len(columns))#,loc='upper center')
      for i in range(0,len(text_contents)):
        plt.text(0.1, 0.95-i*0.15, text_contents[i], 
                 transform=plt.gca().transAxes,
                 fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

      plt.savefig(str(filenames)+"_selection.pdf")
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
      fig, (ax1) = plt.subplots(1,1)
      sets=[]
      cols = ['black']
      labs = ['flat']
      styles = ['step']
      lines=["-",""]
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
      
    def plot_particle_counts(self, mc_counts, columns):
      """
      Plot a grouped horizontal bar chart comparing particle type counts
      between different datasets and adds percentage change labels.
      
      Args:
          mc_counts : list of arrays/lists of particle codes (one per dataset)
          columns   : labels for datasets (e.g., ["old_cuts", "no_cuts"])
      """
      # Map PDG/startCodes to categories
      labels = ["FlateMinus","FlatePlus"]
      pdg_codes = [173,174]
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
          bars.append(ax.barh(y + i * bar_height, data, height=bar_height, label=columns[i]))
      
      # Add percentage change labels to the second set of bars
      for i, bar in enumerate(bars[1]): # Iterate over the bars of the second dataset
          # Get the percentage change for the corresponding category
          change = percent_changes[i]
          
          # Format the label string
          label_text = f'{change:.1f}%'
          
          # Choose color based on whether change is positive or negative
          color = 'red' if change < 0 else 'green'
          
          # Position the label
          # Get the y-position and width (x-value) of the bar
          ax.text(
              bar.get_width(), 
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
      plt.savefig("particle_comparison_with_changes.pdf")
      plt.show()
      
    def plot_cut_eff(self, numerator_array, denominator_array, bin_centers, title="Ratio of Arrays", name="all", x_label="true momentums", y_label="Efficiency"):
      """
      Calculates the element-wise ratio of two arrays and plots the result.
      
      Args:
          numerator_array (np.ndarray): The array for the numerator.
          denominator_array (np.ndarray): The array for the denominator.
          title (str): The title of the plot.
          x_label (str): The label for the x-axis.
          y_label (str): The label for the y-axis.
      """
      # 1. Ensure arrays have the same shape
      if numerator_array.shape != denominator_array.shape:
          raise ValueError("Input arrays must have the same shape.")

      # 2. Handle potential division by zero
      # Use np.divide with 'where' to perform division only where denominator is not zero.
      # Specify the `out` array to hold the result and set values to 0 where denominator is zero.
      ratio = np.divide(numerator_array, denominator_array, out=np.zeros_like(numerator_array, dtype=float), where=denominator_array != 0)

      # 3. Create the plot
      fig, ax = plt.subplots(figsize=(10, 6))
      
      #ax.plot(ratio, marker='o', linestyle='-', color='b')
      #plt.scatter(, marker="-")
      plt.plot(bin_centers, ratio, marker='o', linestyle='-', label='Connected Points')
      ax.set_title(title)
      ax.set_xlabel(x_label)
      ax.set_ylabel(y_label)
      ax.grid(True)
      
      # Optional: Highlight where denominator was zero
      zero_indices = np.where(denominator_array == 0)[0]
      #if zero_indices.size > 0:
      #    ax.plot(zero_indices, ratio[zero_indices], 'rx', label='Denominator was zero')
      #    ax.legend()
      
      
      plt.savefig("eff_"+str(name)+".pdf")
      plt.show()
      
    def plot_2D(self, xs, ys):
      
      for i, x in enumerate(xs):
        y = ys[i]
        x_sync = []
        y_sync = []
        for j, element in enumerate(x):
          for k, subelement in enumerate (element):
            for l, subsubelement in enumerate (subelement):
              if(x[j][k][l] != None):
                for m, y_els in enumerate (y[j][k]):
                  if(y[j][k][m] != None):
                    
                    x_sync.append(x[j][k][l])
                    y_sync.append(y[j][k][m])
              

        # Plot the 2D histogram
        fig, ax = plt.subplots(figsize=(8, 6))
        h = ax.hist2d(y_sync, x_sync, bins=50, cmin = 1, cmap='viridis')

        # Add labels and a color bar
        ax.set_xlabel("True Momentum at TrkEnt [MeV/c]")
        ax.set_ylabel("rmax")
       
        fig.colorbar(h[3], ax=ax, label='Counts in bin') # Changed from h to h[3] to match hist2d output

        # Display the plot
        plt.show()

    def fit_momentum(self, data_list, start, end, opt, label):
        """
        Fits a simple Polynomial shape to the reconstructed momentum data
        using an extended unbinned maximum likelihood fit.
        """
        # Create figure with two subplots: main plot and ratio plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        colors = [ "blue","green"]
        labels = [ "Flateminus","Flateplus"]
        
        # Store text box y-positions to avoid overlap
        text_y_pos = [0.8, 0.5]
        text_x_pos = [0.8, 0.5]
        norm = 0.
        for i, data in enumerate(data_list):
            mom_mag_skim = ak.nan_to_none(data)
            mom_mag_skim = ak.drop_none(mom_mag_skim)

            # Define the observable space for the fit
            obs_mom = zfit.Space('x', limits=(start, end))
            mom_np = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))
            mom_zfit = zfit.Data.from_numpy(array=mom_np, obs=obs_mom)
            
            # Define parameters for the Polynomial shape and yield
            
            N_Flat = zfit.Parameter('N_Flat', 5000, 10, 150000)
            
            # Create parameters for the coefficients
            if opt == "poly":

              c1 = zfit.Parameter("c1", 0.1, -2, 2)
              c2 = zfit.Parameter("c2", 0.1, -2, 2)
              c3 = zfit.Parameter("c3", 0.1, -2, 2)
              c4 = zfit.Parameter("c4", 0.1, -2, 2)
              c5 = zfit.Parameter("c5", 0.1, -2, 2)
              coeffs = [c1, c2, c3, c4, c5] #--> 3 seems good
              text_y_pos = [0.98, 0.65]
              text_x_pos = [0.05, 0.05]
              # Create a Chebyshev polynomial PDF
              poly_model = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=coeffs, extended=N_Flat)

            if opt == "erf":
              mu = zfit.Parameter("mu", 103, 100, 110)
              sigma = zfit.Parameter("sigma", 7, 1, 20)
              poly_model = CustomErfPDF(mu=mu, sigma=sigma, obs=obs_mom, extended=N_Flat)

            if opt == "landau":
              #loc  = zfit.Parameter("loc",   0.0, -5.0, 5.0)
              #scale = zfit.Parameter("scale", 1.0,  0.0, 5.0)
              #poly_model = trunc_landau(obs=obs_mom, loc=loc, scale=scale, extended=N_Flat)
              mpv = zfit.Parameter('mpv', 0, 0, 0.45)
              width = zfit.Parameter('width', 0.1, 0.05, 0.15)
              text_y_pos = [0.98, 0.8]
              text_x_pos = [0.05, 0.05]
              #alpha = zfit.Parameter('alpha', 1.0, 0.1, 5)
              #n = zfit.Parameter('n', 1.0, 0.1, 10)
              #cb_pdf = zfit.pdf.CrystalBall(mu=0, sigma=sigma, alpha=alpha, n=n, obs=obs_mom)

              
              poly_model = CustomLandau(mpv=mpv, width=width, obs=obs_mom, extended=N_Flat)
              #poly_model = zfit.pdf.FFTConvPDFV1(func=landau_pdf, kernel=cb_pdf, obs=obs_mom, extended=N_Flat)
            if opt == "logN":
              mu = zfit.Parameter('mu', 0, -1, 1)
              sigma = zfit.Parameter('sigma', 0.1, 0.1,0.7)
              poly_model = zfit.pdf.LogNormal(mu=mu, sigma=sigma, obs=obs_mom, extended=N_Flat)


            if opt == "dscb":
              mu = zfit.Parameter("mu", 0, -1,1)
              sigma = zfit.Parameter("sigma", 0.5, 0.1, 5.0)
              alphal = zfit.Parameter("alphal", 0.5, 0.1, 2.0)
              nl = zfit.Parameter("nl", 2.0, 1.0, 20.0)
              alphar = zfit.Parameter("alphar", 0.5, 0.1, 2.0)
              nr = zfit.Parameter("nr", 2.0, 1.0, 20.0)
              text_y_pos = [0.98, 0.65]
              text_x_pos = [0.05, 0.05]
              ax1.set_yscale('log')
              poly_model = zfit.pdf.DoubleCB(mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr, obs=obs_mom, extended=N_Flat)

      
            # Create the extended unbinned negative log-likelihood loss
            nll = zfit.loss.ExtendedUnbinnedNLL(model=poly_model , data=mom_zfit)
            
            # Minimize the loss and get the result
            minimizer = zfit.minimize.Minuit()
            result = minimizer.minimize(loss=nll)
            hesse_errors = result.hesse()
            print(result)
            
            # Print fitted Chebyshev coefficients if this was a polynomial fit
            if opt == "poly":
                print(f"\n{'='*70}", flush=True)
                print(f"FITTED CHEBYSHEV POLYNOMIAL: {labels[i]}", flush=True)
                print(f"{'='*70}", flush=True)
                print(f"Momentum range: [{start}, {end}] MeV", flush=True)
                print(f"Yield (N_Flat): {result.params[N_Flat]['value']:.0f}", flush=True)
                print(f"\nPolynomial Coefficients:", flush=True)
                for param in [c1, c2, c3, c4, c5]:
                    param_name = param.name
                    param_value = result.params[param]['value']
                    param_error = result.params[param]['hesse']['error'] if 'hesse' in result.params[param] else 0
                    print(f"  {param_name}: {param_value:+.8f} ± {param_error:.8f}", flush=True)
                print(f"{'='*70}\n", flush=True)
            
            # --- Plotting the fit result ---
            
            fit_range = (obs_mom.lower[0, 0], obs_mom.upper[0, 0])
            n_bins = 100
            bin_width = (fit_range[1] - fit_range[0]) / n_bins
            
            # --- Main plot ---
            
            mom_plot = np.linspace(fit_range[0], fit_range[1], 200).reshape(-1, 1)

            poly_model_curve = zfit.run(poly_model.pdf(mom_plot) * result.params[N_Flat]['value'] * bin_width)
            ax1.plot(mom_plot.flatten(), poly_model_curve.flatten(), color=colors[i], linestyle="--", label=str(labels[i])+' Fit')
            ax1.grid(True)
            
            data_hist, data_bins, _ = ax1.hist(mom_np, color=colors[i], bins=n_bins, range=fit_range, label=labels[i], histtype='step')
            data_bin_center = (data_bins[:-1] + data_bins[1:]) / 2
            ax1.errorbar(data_bin_center, data_hist, yerr=np.sqrt(data_hist), fmt='.', color=colors[i], capsize=2)
            
            ax1.set_xlabel(str(label))
            ax1.set_ylabel('# of events per bin')
            #ax1.set_yscale('log')
            ax1.legend()
            #ax1.set_title('Polynomial Fit to Momentum Data (Extended Unbinned)')
            
            # --- Add text box with fit parameters ---
            if opt == "poly":
              param_text = (
              f"Fit parameters for {labels[i]}:\n"
              f"$N_{{Flat}} = {result.params[N_Flat]['value']:.0f} \\pm {hesse_errors[N_Flat]['error']:.2f}$\n"
              f"$c_{{1}} = {result.params[c1]['value']:.3f} \\pm {hesse_errors[c1]['error']:.4f}$\n"
              f"$c_{{2}} = {result.params[c2]['value']:.3f} \\pm {hesse_errors[c2]['error']:.4f}$\n"
              f"$c_{{3}} = {result.params[c3]['value']:.3f} \\pm {hesse_errors[c3]['error']:.4f}$\n"
              f"$c_{{4}} = {result.params[c4]['value']:.3f} \\pm {hesse_errors[c4]['error']:.4f}$\n"
              f"$c_{{5}} = {result.params[c5]['value']:.3f} \\pm {hesse_errors[c5]['error']:.4f}$\n"
              )
            if opt == "erf":
              param_text = (
              f"Fit parameters for {labels[i]}:\n"
              f"$N_{{Flat}} = {result.params[N_Flat]['value']:.0f} \\pm {hesse_errors[N_Flat]['error']:.2f}$\n"
              f"$\\mu = {result.params[mu]['value']:.3f} \\pm {hesse_errors[mu]['error']:.4f}$\n"
              f"$\\sigma = {result.params[sigma]['value']:.3f} \\pm {hesse_errors[sigma]['error']:.4f}$\n"
              )
            if opt == "logN":
              param_text = (
              f"Fit parameters for {labels[i]}:\n"
              f"$N_{{Flat}} = {result.params[N_Flat]['value']:.1f} \\pm {hesse_errors[N_Flat]['error']:.2f}$\n"
              f"$\\mu = {result.params[mu]['value']:.3f} \\pm {hesse_errors[mu]['error']:.4f}$\n"
              f"$\\sigma = {result.params[sigma]['value']:.3f} \\pm {hesse_errors[sigma]['error']:.4f}$\n"
              )
              
            if opt == "landau":
              param_text = (
              f"Fit parameters for {labels[i]}:\n"
              f"$N_{{Flat}} = {result.params[N_Flat]['value']:.1f} \\pm {hesse_errors[N_Flat]['error']:.2f}$\n"
              f"$mpv = {result.params[mpv]['value']:.3f} \\pm {hesse_errors[mpv]['error']:.4f}$\n"
              f"$width = {result.params[width]['value']:.3f} \\pm {hesse_errors[width]['error']:.4f}$\n"
              )
              
            if opt == "dscb":
              param_text = (
              f"Fit parameters for {labels[i]}:\n"
              f"$N_{{Flat}} = {result.params[N_Flat]['value']:.0f} \\pm {hesse_errors[N_Flat]['error']:.2f}$\n"
              f"$\\mu = {result.params[mu]['value']:.3f} \\pm {hesse_errors[mu]['error']:.4f}$\n"
              f"$\\sigma = {result.params[sigma]['value']:.3f} \\pm {hesse_errors[sigma]['error']:.4f}$\n"
              f"$\\alpha_{{l}} = {result.params[alphal]['value']:.3f} \\pm {hesse_errors[alphal]['error']:.4f}$\n"
              f"$n_{{l}} = {result.params[nl]['value']:.3f} \\pm {hesse_errors[nl]['error']:.4f}$\n"
              f"$\\alpha_{{r}} = {result.params[alphar]['value']:.3f} \\pm {hesse_errors[alphar]['error']:.4f}$\n"
              f"$n_{{r}} = {result.params[nr]['value']:.3f} \\pm {hesse_errors[nr]['error']:.4f}$\n"
              )

            props = dict(boxstyle='round', facecolor=colors[i], alpha=0.3)
            
            # Position the text box in the upper left corner of the subplot
            # with an offset for each iteration
            ax1.text(text_x_pos[i], text_y_pos[i], param_text, transform=ax1.transAxes,
                     fontsize=10, verticalalignment='top', bbox=props)
            
            # --- Ratio plot ---
            
            data_bin_center_2d = data_bin_center.reshape(-1, 1)
            fit_at_bin_center = zfit.run(poly_model.pdf(data_bin_center_2d) * result.params[N_Flat]['value'] * bin_width)
            ratio = data_hist / fit_at_bin_center
            
            ax2.errorbar(data_bin_center, ratio, yerr=np.sqrt(data_hist) / fit_at_bin_center, fmt='.', color=colors[i], capsize=2)
            ax2.axhline(1, color='gray', linestyle='--')
            ax2.set_ylabel('Ratio (Data/Fit)')
            #ax2.set_xlabel('Reconstructed Momentum [MeV/c]')
            ax2.set_ylim(0.5, 1.5)
            ax2.grid(True)
            #mean = result.params[mu]['value']
            #meanr_err = hesse_errors[mu]['error']
            #sigma = result.params[sigma]['value']
            #sigma_err = hesse_errors[sigma]['error']
            norm = result.params[N_Flat]['value']
        plt.tight_layout()
        plt.savefig("Flatfit_"+str(opt)+".pdf")
        plt.show()
        return  norm

    def overlay_fit(self, c1, c2, c3, c4, c5, data_list, mc_count):
        """
        Fits a simple Polynomial shape to the reconstructed momentum data
        using an extended unbinned maximum likelihood fit.
        """
        # Create figure with two subplots: main plot and ratio plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        colors = ["black"]
        labels = ["MDS2c"]
        
        # Store text box y-positions to avoid overlap
        text_y_pos = [0.8] 

        for i, data in enumerate(data_list):
            mom_mag_skim = ak.nan_to_none(data)
            mom_mag_skim = ak.drop_none(mom_mag_skim)
            
            true_Flat = mom_mag_skim.mask[(mc_count[i] == 173) ]
            true_Flat = ak.to_numpy((ak.flatten(true_Flat,axis=None)))
            print(true_Flat)
            print(mc_count)

            # Define the observable space for the fit
            obs_mom = zfit.Space('x', limits=(95, 105))
            mom_np = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))
            mom_zfit = zfit.Data.from_numpy(array=true_Flat, obs=obs_mom)
            
            # Define parameters for the Polynomial shape and yield
            N_Flat = zfit.Parameter('N_Flat', 5000, 100, 15000)
            
            # Create parameters for the coefficients
            c1 = zfit.Parameter("c1", c1,floating=False)
            c2 = zfit.Parameter("c2", c2, floating=False)
            #c3 = zfit.Parameter("c3", c3, floating=False)
            #c4 = zfit.Parameter("c4", c4, floating=False)
            #c5 = zfit.Parameter("c5", c5, floating=False)
            coeffs = [c1, c2]#, c3]#, c4, c5]

            # Create a Chebyshev polynomial PDF
            poly_model = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=coeffs, extended=N_Flat)


            
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

            poly_model_curve = zfit.run(poly_model.pdf(mom_plot) * result.params[N_Flat]['value'] * bin_width)
            ax1.plot(mom_plot.flatten(), poly_model_curve.flatten(), color=colors[i], linestyle="--", label=str(labels[i])+' Fitted Polynomial')
            ax1.grid(True)
            ax1.set_yscale('log')
            data_hist, data_bins, _ = ax1.hist(mom_np, color=colors[i], bins=n_bins, range=fit_range, label=labels[i], histtype='step')
            true_hist, true_bins, _ = ax1.hist(true_Flat, color="orange", bins=n_bins, range=fit_range, label="Flat", histtype='bar')
            data_bin_center = (data_bins[:-1] + data_bins[1:]) / 2
            ax1.errorbar(data_bin_center, data_hist, yerr=np.sqrt(data_hist), fmt='.', color=colors[i], capsize=2)
            
            ax1.set_xlabel('Reconstructed Momentum [MeV/c]')
            ax1.set_ylabel('# of events per bin')
            ax1.legend()
            ax1.set_title('Polynomial Fit to Momentum Data (Extended Unbinned)')
            
            # --- Add text box with fit parameters ---
            param_text = (
                f"Fit parameters for {labels[i]}:\n"
                f"$N_{{Flat}} = {result.params[N_Flat]['value']:.0f} \\pm {hesse_errors[N_Flat]['error']:.2f}$"
            )
            
            props = dict(boxstyle='round', facecolor=colors[i], alpha=0.3)
            
            # Position the text box in the upper left corner of the subplot
            # with an offset for each iteration
            ax1.text(0.4, text_y_pos[i], param_text, transform=ax1.transAxes,
                     fontsize=10, verticalalignment='top', bbox=props)
            
            # --- Ratio plot ---
            
            data_bin_center_2d = data_bin_center.reshape(-1, 1)
            fit_at_bin_center = zfit.run(poly_model.pdf(data_bin_center_2d) * result.params[N_Flat]['value'] * bin_width)
            ratio = true_hist / fit_at_bin_center
            
            ax2.errorbar(data_bin_center, ratio, yerr=np.sqrt(data_hist) / fit_at_bin_center, fmt='.', color=colors[i], capsize=2)
            ax2.axhline(1, color='gray', linestyle='--')
            ax2.set_ylabel('Ratio (Flat/Fit)')
            ax2.set_xlabel('Reconstructed Momentum [MeV/c]')
            ax2.set_ylim(0.5, 1.5)
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig("Flatfit.pdf")
        plt.show()
