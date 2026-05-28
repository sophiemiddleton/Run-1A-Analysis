# helper.py provides helper functions for the simultaneous fits and convolutions

import zfit
import numpy as np
import tensorflow as tf
import pickle as pkl
from RLE.landau_pdf import trunc_landau
from pyutils.pylogger import Logger


# Module-level logger
logger = Logger(print_prefix='[helper] ', verbosity=1)

# =============================================================================
# Helper Module for Theoretical and Experimental PDF Generation
# =============================================================================

def load_lineshape(lineshape_in, name, binedges=None):
    """
    Loads theoretical lineshapes from various file formats or returns objects directly.
    
    Supported formats:
    - .pkl: Pickled numpy histogram or raw data (converts to histogram if needed).
    - .txt/.tbl: Two-column space-separated (momentum, density) files.
    """
    if isinstance(lineshape_in, str):
        # Handle Pickle files
        if lineshape_in.endswith('.pkl'):
            with open(lineshape_in, 'rb') as f:
                lineshape = pkl.load(f)
            logger.log(f'Loaded lineshape for {name} as pickle!', 'info')
            
            if isinstance(lineshape, tuple):
                return lineshape
            else:
                return np.histogram(lineshape, binedges)
        
        # Handle text/table files
        else:
            vals = []
            cents = []
            with open(lineshape_in, 'r') as f:
                for line in f:
                    parts = line.split()
                    if not parts: continue
                    cents.append(float(parts[0]))
                    vals.append(float(parts[1]))
            
            logger.log(f'Loaded lineshape for {name} as .txt/tbl file!', 'info')
            
            # Calculate bin edges from centers
            cents = np.array(cents)
            edges = [(a + b) / 2 for a, b in zip(cents[:-1], cents[1:])]
            low = cents[0] - (edges[0] - cents[0])
            edges.insert(0, low)
            high = cents[-1] + (cents[-1] - edges[-1])
            edges.append(high)
            
            return (np.array(vals), np.array(edges))
    
    return lineshape_in


def gen_theo_exp(fitpars_in, lineshapes_in) -> dict:
    """
    Combines theoretical lineshapes (optionally with efficiency) into a single 
    histogram-based PDF object for convolution.
    """
    # Case 1: Single theoretical component
    if len(lineshapes_in) == 1:
        prob, edges = load_lineshape(list(lineshapes_in.values())[0], 
                                     list(lineshapes_in.keys())[0])
    
    # Case 2: Multi-component (Theo * Eff)
    elif sorted(lineshapes_in.keys()) == ['eff', 'theo']:
        p_theo, e_theo = load_lineshape(lineshapes_in['theo'], 'theo')
        e_theo = np.round(e_theo, decimals=2)
        
        p_eff, e_eff = load_lineshape(lineshapes_in['eff'], 'eff')
        e_eff = np.round(e_eff, decimals=2)
        
        # Rebin/Align efficiency and theory onto a unique edge set
        e_all = np.unique(np.append(e_theo, e_eff))
        e_all = e_all[(e_all - e_theo[0] >= -0.01) & (e_all - e_theo[-1] <= 0.01)]
        
        b_theo = 0
        b_eff = 0
        p_mult = []
        
        for e in e_all[:-1]:
            if b_theo < len(p_theo) - 1 and abs(e - e_theo[b_theo + 1]) < 0.01:
                b_theo += 1
            if b_eff < len(p_eff) - 1 and abs(e - e_eff[b_eff + 1]) < 0.01:
                b_eff += 1
            p_mult.append(p_theo[b_theo] * p_eff[b_eff])
            
        prob = np.array(p_mult)
        edges = e_all
    
    else:
        logger.log('Unsupported lineshape combination', 'error')
        exit()

    # Consolidate experimental parameters and info dictionary
    fitpars_out = {'info': {}, 'lineshape': (prob, edges)}
    for comp in fitpars_in.keys():
        fitpars_comp = fitpars_in[comp].copy()
        fitpars_out['info'][comp] = fitpars_comp.pop('info', None)
        fitpars_out.update(fitpars_comp)

    return fitpars_out


def make_HistogramPDF(prob, edges):
    """
    Factory function to create a custom zfit Histogram PDF from numpy arrays.
    """
    class myHistogramPDF(zfit.pdf.ZPDF):
        _N_OBS = 1
        _PARAMS = []
        _PROB = tf.reshape(tf.constant(prob, dtype=tf.float64), [len(prob), 1])
        _LOW  = tf.reshape(tf.constant(edges[:-1], dtype=tf.float64), [len(prob), 1])
        _HIGH = tf.reshape(tf.constant(edges[1:], dtype=tf.float64), [len(prob), 1])
        
        def _unnormalized_pdf(self, x):
            x = zfit.z.unstack_x(x)
            # Binary search logic inside TF: check which bin x falls into
            within_bounds = (x >= self._LOW) & (x < self._HIGH)
            return tf.reduce_sum(tf.where(within_bounds, self._PROB, 0), axis=0)
            
    return myHistogramPDF


def doConv(true_pdf, obs_gen, obs_res, name, info, zpars):
    """
    Performs FFT convolution between the theoretical lineshape and experimental response.
    Supports piecewise convolution over momentum bins (p_bins).
    """
    # Define observation and normalization spaces for convolution
    obs_conv = zfit.Space('x', float(obs_gen.v1.lower - obs_res.v1.lower), 
                               float(obs_gen.v1.upper - obs_res.v1.upper))
    obs_full = zfit.Space('x', float(obs_gen.v1.lower + obs_res.v1.lower), 
                               float(obs_gen.v1.upper + obs_res.v1.upper))

    binwidth_eval = 0.1
    nbins_gen = int((obs_gen.v1.upper - obs_gen.v1.lower) / binwidth_eval)
    nbins_res = int((obs_res.v1.upper - obs_res.v1.lower) / binwidth_eval)
    
    pdfs = []
    fracs = []

    # Filter momentum bins that fall within the fit range
    p_bins = [pb for pb in info['p_bins'] 
              if pb[1] > float(obs_gen.v1.lower) + 0.0001 
              and pb[0] < float(obs_gen.v1.upper) - 0.0001]
    
    pdf_type = info['pdf']
    
    for ip, pbin in enumerate(p_bins):
        # Determine slice boundaries
        plow  = float(obs_gen.v1.lower) if ip == 0 else pbin[0]
        phigh = float(obs_gen.v1.upper) if ip == len(p_bins) - 1 else pbin[1]
        obs_slice = zfit.Space('x', plow, phigh)
        
        # Prepare the theoretical PDF slice
        if isinstance(true_pdf, tuple):
            prob, edges = true_pdf
            pfilt = (edges >= plow) & (edges < phigh)
            efilt = (edges >= plow) & (edges <= phigh)
            
            my_hist_cls = make_HistogramPDF(prob[pfilt[:-1]], edges[efilt])
            norm = sum(prob[pfilt[:-1]] * (edges[efilt][1:] - edges[efilt][:-1]))
            fracs.append(norm)
            true_pdf_slice = my_hist_cls(obs=obs_gen)
            
        elif len(p_bins) == 1:
            true_pdf_slice = true_pdf
        else:
            norm = float(true_pdf.integrate(limits=obs_slice, norm=False))
            fracs.append(norm)
            true_pdf_slice = zfit.pdf.TruncatedPDF(true_pdf, limits=obs_slice, obs=obs_gen)

        # Prepare the resolution/experimental kernel kernel
        if pdf_type == "landau":
            res_pdf = trunc_landau(obs=obs_res, 
                                   loc=zpars[f'loc{ip}_{name}'], 
                                   scale=zpars[f'scale{ip}_{name}'])
        elif pdf_type == "gcb":
            res_pdf = zfit.pdf.GeneralizedCB(obs=obs_res, 
                                             mu=zpars[f'mu{ip}_{name}'], 
                                             sigmal=zpars[f'sigmaL{ip}_{name}'], 
                                             sigmar=zpars[f'sigmaR{ip}_{name}'], 
                                             alphal=zpars[f'alphaL{ip}_{name}'], 
                                             alphar=zpars[f'alphaR{ip}_{name}'], 
                                             nl=zpars[f'nL{ip}_{name}'], 
                                             nr=zpars[f'nR{ip}_{name}'])
        else:
            logger.log(f'ERROR: {pdf_type} is not supported for experimental effects. Exiting...', 'error')
            exit()

        # Perform FFT Convolution
        func   = true_pdf_slice if (nbins_gen >= nbins_res) else res_pdf
        kernel = res_pdf        if (nbins_gen >= nbins_res) else true_pdf_slice
        nbins_kern = min(nbins_res, nbins_gen)
        
        conv = zfit.pdf.FFTConvPDFV1(func, kernel, n=nbins_kern, obs=obs_conv, norm=obs_full)
        pdfs.append(conv)

    # Sum all slices if using multiple p_bins
    if len(pdfs) > 1:
        total_frac = sum(fracs)
        pdf_sum = zfit.pdf.SumPDF(pdfs, fracs=[f/total_frac for f in fracs], 
                                  obs=obs_conv, norm=obs_full)
    else:
        pdf_sum = pdfs[0]
    
    return pdf_sum