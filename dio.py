import zfit
import numpy as np
import awkward as ak
import dill as pickle
import tensorflow as tf
import matplotlib.pyplot as plt

default_resolution_params = {
    'su2020' : {
        'delta_p_0': (-0.5527, -5, 5),
        'a_0': (0.3765, 0, 20),
        'b_0': (4.494, 1, 10),
        'alpha_1': (1.173, 0, 15),
        'alpha_2': (0.4416, 0, 10),
        'N_1': (1.728, 0, 100),
        'N_2': (6.507, 0, 100)
    },
}

class MomResolution(zfit.pdf.ZPDF):
    _N_OBS = 1
    _PARAMS = ['delta_p_0', 'a_0', 'b_0', 'alpha_1', 'alpha_2', 'N_1', 'N_2']

    def _unnormalized_pdf(self, x):
        z = zfit.z
        x = zfit.z.unstack_x(x)

        delta_p_0 = self.params['delta_p_0']
        a_0       = self.params['a_0']
        b_0       = self.params['b_0']
        alpha_1   = self.params['alpha_1']
        alpha_2   = self.params['alpha_2']
        N_1       = self.params['N_1']
        N_2       = self.params['N_2']

        shifted = x - delta_p_0

        B_1 = alpha_1 - N_1 / (a_0 * b_0 * (1 - z.exp(-b_0 * alpha_1)))
        A_1 = z.exp(a_0 * (-b_0 * alpha_1 - z.exp(-b_0 * alpha_1))) * (alpha_1 - B_1)**N_1

        B_2 = -alpha_2 - N_2 / (a_0 * b_0 * (1 - z.exp(b_0 * alpha_2)))
        A_2 = z.exp(a_0 * (b_0 * alpha_2 - z.exp(b_0 * alpha_2))) * (B_2 + alpha_2)**N_2

        core = z.exp(a_0 * (b_0 * shifted - z.exp(b_0 * shifted)))

        pdf = tf.where(shifted < -alpha_1,
                       A_1 / (- B_1 - shifted)**N_1,
                       tf.where(shifted < alpha_2,
                                core,
                                A_2 / (B_2 + shifted)**N_2))

        return pdf

def Unbinned_fit_resolution_function(data_np, fit_range, mom_resolution = 0):
    obs_mom_diff = zfit.Space('mom_diff', limits=fit_range)
    
    # build the zfit PDF for the momentum resolution
    N = zfit.Parameter('N', 10000, 0, 1e10)
    zpars_res = {'N': N}

    if mom_resolution == 0:
        for parameter in default_resolution_params['su2020']:
            zpars_res[parameter] = zfit.Parameter(parameter + '_res', 
                                                  default_resolution_params['su2020'][parameter][0], 
                                                  default_resolution_params['su2020'][parameter][1],
                                                  default_resolution_params['su2020'][parameter][2])
        PDF = MomResolution(obs=obs_mom_diff, extended = N, 
                            delta_p_0=zpars_res['delta_p_0'],
                                                      a_0=zpars_res['a_0'], 
                                                      b_0=zpars_res['b_0'], 
                                                      alpha_1=zpars_res['alpha_1'],
                                                      alpha_2=zpars_res['alpha_2'], 
                                                      N_1=zpars_res['N_1'], 
                                                      N_2=zpars_res['N_2'])
    
    # convert numpy array to zfit Data
    data_zfit =  zfit.Data.from_numpy(obs=obs_mom_diff, array=data_np)

    # do the fit
    loss = zfit.loss.ExtendedUnbinnedNLL(model = PDF, data = data_zfit)
    minimizer = zfit.minimize.Minuit(
        gradient=True,    # use Minuit’s own gradient
        mode=2,           # full Hesse
        maxiter=2000      # more calls
    )
    result = minimizer.minimize(loss, params=zpars_res.values())

    return result, PDF, N

def plot_fit_result(data_np, fit_range, PDF, N, log_scale=False):
    n_bins = 50
    scale = 1 / n_bins * (fit_range[1] - fit_range[0])
    data_hist, data_binedge = np.histogram(data_np, bins=n_bins, range=fit_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])
    ax1.hist(data_np, color='black', bins=n_bins, range=fit_range, histtype='step')
    ax1.errorbar(data_bincenter, data_hist, yerr=np.sqrt(data_hist), color='None', ecolor='black', capsize=3)

    ax1.plot(data_bincenter, (PDF.pdf(data_bincenter, norm_range=fit_range) * N* scale).numpy(), '-r')

    # plot the residuals
    residuals = (data_hist - PDF.pdf(data_bincenter, norm_range=fit_range).numpy() * N * scale) / np.sqrt(data_hist)
    ax2.errorbar(data_bincenter, residuals, yerr=np.ones_like(residuals), fmt='o', color='black', markersize=3, capsize=3)
    ax2.set_xlim(ax1.get_xlim())

    if log_scale:
        ax1.set_yscale('log')


def Unbinned_fit_efficiency(data_mc_np, fit_range, degree = 4):
    obs_mom = zfit.Space('mom', limits=fit_range)
    
    # build the zfit PDF for the efficiency
    N = zfit.Parameter('N', 10000, 0, 1e10)
    zpars_eff = {'N': N}

    # Chebychev polynomial coefficients
    coeffs = [zfit.Parameter(f'c_{i}', 0.1, -1, 1) for i in range(degree + 1)]
    for i, coeff in enumerate(coeffs):
        zpars_eff[f'c_{i}'] = coeff
    
    PDF = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=coeffs, extended=N)

    # convert numpy array to zfit Data
    data_zfit =  zfit.Data.from_numpy(obs=obs_mom, array=data_mc_np)

    # do the fit
    loss = zfit.loss.ExtendedUnbinnedNLL(model=PDF, data=data_zfit)
    minimizer = zfit.minimize.Minuit(
        gradient=True,    # use Minuit’s own gradient
        mode=2,           # full Hesse
        tol=1e-10,        # tighter EDM goal
        maxiter=2000      # more calls
    )
    result = minimizer.minimize(loss, params=zpars_eff.values())

    return result, PDF, N