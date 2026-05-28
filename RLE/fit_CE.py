########################################################################################
# Standalone fits to the high-stats CE sample                                          #
# TODO this could be replaced by running regular py-fitter on the high stats CE sample #
########################################################################################


import matplotlib.pyplot as plt
import hist as hist
import mplhep
import numpy as np
import zfit
import sys
import awkward as ak
import pickle as pkl
import math
import tensorflow as tf
import time
import gc
sys.path.append('/exp/mu2e/app/users/sdittmer/LikelihoodAnalysis/py-fitter')
from analyze import Analyze
from pyutils.pyprocess import Processor, Skeleton
from pyutils.pyvector import Vector
from pyutils.pyselect import Select
from landau_pdf import trunc_landau

# from https://github.com/Mu2e/Offline/blob/main/Mu2eUtilities/src/ConversionSpectrum.cc
eMax = 104.97
alpha = 1./137.035999139
me = 0.511
def LeadingLog(E):
    val = (1./eMax)*(alpha/(2*math.pi))*(math.log(4*E*E/me/me)-2.)*((E*E+eMax*eMax)/eMax/(eMax-E))
    if val < 0.0: val = 0.0
    return val
def binned_spectrum_CeLL(binwidth=0.1):
    nbins = math.floor(eMax/binwidth)
    if binwidth*nbins < eMax: nbins += 1
    upedge = binwidth*nbins
    edges = np.linspace(0., upedge, nbins+1)
    centers = (edges[:-1]+edges[1:])/2.
    vectorize_LL = np.vectorize(LeadingLog)
    values = vectorize_LL(centers)
    values[-1] = (1-np.sum(values[:-1])*binwidth)/binwidth
    return values,edges

def make_HistogramPDF(prob,edges):
    class myHistogramPDF(zfit.pdf.ZPDF):
        _N_OBS = 1
        _PARAMS = []
        _PROB = tf.reshape(tf.constant(prob),      [len(prob),1])
        _LOW  = tf.reshape(tf.constant(edges[:-1]),[len(prob),1])
        _HIGH = tf.reshape(tf.constant(edges[1:]), [len(prob),1])
        
        def _unnormalized_pdf(self, x):
            x = zfit.z.unstack_x(x)
            return tf.reduce_sum(tf.where((x >= self._LOW) & (x < self._HIGH),self._PROB,0),0)
    return myHistogramPDF

from collections.abc import Iterable
class MySumPDF(zfit.pdf.BaseFunctor):
    def __init__(
        self,
        pdfs: Iterable[zfit.core.interfaces.ZfitPDF],
        fracs: Iterable[float],
        obs: zfit.util.ztyping.ObsTypeInput = None,
        extended: zfit.util.ztyping.ExtendedInputType = None,
        norm: zfit.util.ztyping.NormInputType = None,
        name: str = "MySumPDF",
        label = None,
    ):
        self.fracs = fracs
        self.pdfs = pdfs
        super().__init__(pdfs=pdfs, obs=obs, params={}, name=name, extended=extended, norm=norm, label=label)

    def _unnormalized_pdf(self, x):
        pdfs = self.pdfs
        fracs = self.fracs
        probs = [pdf.pdf(x) * frac for pdf, frac in zip(pdfs, fracs)]
        prob = sum(probs)  # to keep the broadcasting ability
        return zfit.z.convert_to_tensor(prob)

analyse = Analyze(verbosity=0)

processor = Processor(use_remote=False, verbosity=0)
            
# Process the files using multithreading
my_branches = { 
    "crv" : [
        "crvcoincs.time"
    ],
    "trk" : [
        "trk.nactive", 
        "trk.pdg", 
        "trkqual.result"
    ],
    "trkfit" : [
        "trksegs",
        "trksegsmc",
        "trksegpars_lh"
    ],
    "trkmc" : [
        "trkmcsim"
    ]
}

npar = 1

binwidth_eval = 0.1
binwidth_plot = 0.5

#fittypes = ['DCB','GCB','KDE']
#fitcat = 'analytic'
fittypes = ['landau_gcb_fix','landau_gcb_constrain']
fitcat = 'theo_eff_loss_res'

def doGCBFit(data_in,obs_mom):
    N = zfit.Parameter('N',  len(data_in), len(data_in)-100., len(data_in)+100., step_size=0.1)
    mu     = zfit.Parameter("mu",     104.1,  102.0, 106.0,  step_size=0.001)  
    sigmaL = zfit.Parameter("sigmaL", 0.3,    0.0,   2.0,    step_size=0.001)
    sigmaR = zfit.Parameter("sigmaR", 0.3,    0.0,   2.0,    step_size=0.001)
    alphaL = zfit.Parameter("alphaL", 0.7,    0.0,   2.0,    step_size=0.001)
    alphaR = zfit.Parameter("alphaR", 2.4,    0.0,   5.0,    step_size=0.001)
    nL     = zfit.Parameter("nL",     2.0,    0.0,   5.0,    step_size=0.001)
    nR     = zfit.Parameter("nR",     2.0,    0.0,   5.0,    step_size=0.001)
    npar   = 8
    
    # Define PDF
    acb = zfit.pdf.GeneralizedCB(obs=obs_mom, mu=mu, sigmal=sigmaL, sigmar=sigmaR, alphal=alphaL, alphar=alphaR, nl=nL, nr=nR, extended=N)
    # Create the negative log likelihood
    nll = zfit.loss.ExtendedUnbinnedNLL(model=acb, data=data_in)  # loss
    # Load and instantiate a minimizer
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss=nll)
    try:
        errors, _ = result.errors(method='minuit_minos')
    except:
        errors = None
        print('WARNING! Invalid fit, postfit parameters may not be optimal')

    return acb,result,errors

def doConv(true_pdf, obs_true, res_type, inputs, params_tot, constraints, nll, ax):

    bound = 1. if res_type == "res" else 10.
    obs_res = zfit.Space('x', -bound, bound)
    x_res = np.linspace(obs_res.v1.lower, obs_res.v1.upper, 280)

    x_true = np.linspace(obs_true.v1.lower, obs_true.v1.upper, 280)
    nbins_true = int((obs_true.v1.upper - obs_true.v1.lower)/binwidth_eval)

    obs_conv = zfit.Space('x',float(obs_true.v1.lower-obs_res.v1.lower),float(obs_true.v1.upper-obs_res.v1.upper))
    x_conv = np.linspace(obs_conv.v1.lower, obs_conv.v1.upper, 280)

    obs_full = zfit.Space('x',float(obs_true.v1.lower+obs_res.v1.lower),float(obs_true.v1.upper+obs_res.v1.upper))

    params = {}
    for pardict in inputs["params"].values():
        params.update(pardict)

    zpars = {}
    for p in params.keys():
        if inputs['treat_params'] == 'constrain':
            zpars[p] = zfit.Parameter(p+"_"+res_type, params[p][0], params[p][0]+5*params[p][1], params[p][0]+5*params[p][2],step_size=0.0001)
            params_tot.append(zpars[p])
            constraints.append(zfit.constraint.GaussianConstraint(zpars[p],observation=params[p][0],uncertainty=max(abs(params[p][1]),abs(params[p][2]))))
        elif inputs['treat_params'] == 'fix':
            zpars[p] = zfit.Parameter(p+'_'+res_type, params[p][0], params[p][0]-0.005, params[p][0]+0.005, floating=False)
        elif inputs['treat_params'] == 'simul':
            zpars[p] = zfit.ComposedParameter(p+'_'+res_type, lambda x : 1*x, params=params[p])
        else:
            zpars[p] = zfit.Parameter(p+'_'+res_type, params[p][0], params[p][1], params[p][2])
            params_tot.append(zpars[p])
            
    pdfs = []
    fracs = []

    for ip,pbin in enumerate(list(inputs["params"].keys())):
        # Get res pdf
        if inputs["pdf"] == "landau":
            res_pdf = trunc_landau(obs=obs_res, loc=zpars[f'loc{ip}'], scale=zpars[f'scale{ip}'])
        elif inputs["pdf"] == "gcb":
            res_pdf = zfit.pdf.GeneralizedCB(obs=obs_res, mu=zpars[f'mu{ip}'], sigmal=zpars[f'sigmaL{ip}'], sigmar=zpars[f'sigmaR{ip}'], alphal=zpars[f'alphaL{ip}'], alphar=zpars[f'alphaR{ip}'], nl=zpars[f'nL{ip}'], nr=zpars[f'nR{ip}'])

        # If needed, set up simultaneous fit
        if constrain == "simul":
            # Get slice of flat p data
            true_mom, reco_mom = inputs["simul_source"]
            true_slice = true_mom[(true_mom>=p_bins[ip]) & (true_mom<p_bins[ip+1])]
            reco_slice = reco_mom[(true_mom>=p_bins[ip]) & (true_mom<p_bins[ip+1])]
            res_slice = flat_reco_slice - flat_true_slice
            del true_mom, reco_mom, true_slice, reco_slice
            data_res = zfit.Data(data=res_slice, obs=obs_res)
            N_slice = zfit.Parameter(f'N_{res_type}{ip}', len(res_slice), 0.8*len(res_slice), 1.2*len(res_slice), step_size=0.1)
            del res_slice
            res_ext = res_pdf.create_extended(N_slice)
            params_tot.append(N_slice)

            nll_tmp = zfit.res.ExtendedUnbinnedNLL(model=res_ext, data=data_res)  # loss
            if nll == None:
                nll = nll_tmp
            else:
                nll = nll+nll_tmp
            del data_res
            del nll_tmp

        # Do convolution
        plow  = float(obs_true.v1.lower) if ip == 0 else pbin[0]
        phigh = float(obs_true.v1.upper) if ip == len(inputs["params"])-1 else pbin[1]
        obs_slice = zfit.Space('x',plow,phigh)
        if isinstance(true_pdf,list):
            prob = true_pdf[0]
            edges = true_pdf[1]
            pfilt = (edges>=plow)&(edges<phigh)
            efilt = (edges>=plow)&(edges<=phigh)
            my_hist = make_HistogramPDF(prob[pfilt[:-1]],edges[efilt])
            norm = sum(prob[pfilt[:-1]]*(edges[efilt][1:]-edges[efilt][:-1]))
            fracs.append(norm)
            true_pdf_slice = my_hist(obs=obs_true)
        elif len(inputs["params"]) == 1:
            true_pdf_slice = true_pdf
        else:
            norm = float(true_pdf.integrate(limits=obs_slice, norm=False))
            fracs.append(norm)
            true_pdf_slice = zfit.pdf.TruncatedPDF(true_pdf,limits=obs_slice,obs=obs_true)
        conv = zfit.pdf.FFTConvPDFV1(true_pdf_slice, res_pdf, n=nbins_true, obs=obs_conv, norm=obs_full)
        pdfs.append(conv)

        irow = 1 if res_type == "loss" else 2
        ax[irow][0].plot(x_true, true_pdf_slice.pdf(x_true)*float(fracs[-1]))
        ax[irow][1].plot(x_res, res_pdf.pdf(x_res))
        if len(pdfs) == 1:
            ax[irow][2].plot(x_conv, pdfs[0].pdf(x_conv)*fracs[0])
        else:
            conv_sum = zfit.pdf.SumPDF(pdfs, fracs=[f/sum(fracs) for f in fracs], obs=obs_conv, norm=obs_full, extended=sum(fracs))
            #conv_sum = MySumPDF(pdfs, fracs=[f/sum(fracs) for f in fracs], obs=obs_conv, extended=sum(fracs))
            ax[irow][2].plot(x_conv, conv_sum.ext_pdf(x_conv))

    if len(pdfs) > 1:
        pdf_sum = zfit.pdf.SumPDF(pdfs, fracs=[f/sum(fracs) for f in fracs], obs=obs_conv)
        #pdf_sum = MySumPDF(pdfs, fracs=[f/sum(fracs) for f in fracs], obs=obs_conv)
    else:
        pdf_sum = pdfs[0]
    del pdfs
    del fracs
    del zpars
    
    return pdf_sum, obs_conv

# TODO make this more general
# For now, assuming we are constructing PDF as theory lineshape * efficiency X loss X resolution 
# Theory lineshape and efficiency are both np histograms (i.e. bin contents and bin edges)
# Loss and resolution are both binned in true momentum
def doConvFit(data_in,obs_mom,inputs_dict,figname):

    fig,ax = plt.subplots(3,3,figsize=(15.6,14.4))

    N = zfit.Parameter('N',  len(data_in), len(data_in)-100., len(data_in)+100., step_size=0.1)
    pars_tot = [N]
    
    # Get theory lineshape
    obs_theo = zfit.Space('x',84.,116.)
    x_theo = np.linspace(obs_theo.v1.lower, obs_theo.v1.upper, 280)
    nbins_theo = int((obs_theo.v1.upper - obs_theo.v1.lower)/binwidth_eval)

    c_theo, e_theo = inputs_dict["theo"]["params"]
    ax[0][0].stairs(c_theo,e_theo)
    
    # Get efficiency
    c_eff, e_eff = inputs_dict["eff"]["params"]
    ax[0][1].stairs(c_eff,e_eff)

    # Construct PDF for theo * eff
    prob = []
    edges = []
    if all([any([abs(a-b)<0.0001 for b in e_theo]) for a in e_eff]):
        effbin = -1
        for i,e in enumerate(e_theo[:-1]):
            if abs(e-e_eff[effbin+1])<0.0001: effbin += 1
            if effbin >= 0:
                prob.append(c_theo[i]*c_eff[effbin])
                edges.append(e)
        edges.append(e_theo[-1])
    else:
        print("Warning! Eff binedges are not a subset of theo binedges.")
        print(f"Eff  binning: {e_eff}")
        print(f"Theo binning: {e_theo}")
        exit()
        
    ax[0][2].stairs(prob,edges)
    
    constraints = []
    nll_simul = None

    
    
    conv_loss_pdf, obs_conv_loss = doConv([np.array(prob),np.array(edges)],  obs_theo,      'loss', inputs_dict['loss'], pars_tot, constraints, nll_simul, ax)
    gc.collect()
    conv_res_pdf,  _             = doConv(conv_loss_pdf,                     obs_conv_loss, 'res',  inputs_dict['res'],  pars_tot, constraints, nll_simul, ax)
    pdf_tot = conv_res_pdf.create_extended(N)
    plt.savefig(f"{figname}_val.png")
    plt.clf()
    gc.collect()
    
    # Create the negative log likelihood
    print("Fitting...")
    nll = zfit.loss.ExtendedUnbinnedNLL(model=pdf_tot, data=data_in, constraints=constraints)  # loss
    if constrain == "simul" : nll = nll+nll_simul
    # Load and instantiate a minimizer
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss=nll, params=pars_tot)
    try:
        errors, _ = result.errors(method='minuit_minos')
    except:
        errors = None
        print('WARNING! Invalid fit, postfit parameters may not be optimal')

    return pdf_tot,result,errors

def doKDEFit(data_in,kdetype,bandwidth=None):
    N = zfit.Parameter('N',  len(data_in), len(data_in)-100., len(data_in)+100., step_size=0.1)
    # Define KDE
    if kdetype == 'grid':
        # Bandwidth options are ‘silverman’, ‘scott’, ‘adaptive_zfit’, ‘adaptive_geom’, or scalar
        kde = zfit.pdf.KDE1DimGrid(data_in, num_grid_points=256, binning_method='linear', extended=N, bandwidth=bandwidth)
    if kdetype == 'isj':
        kde = zfit.pdf.KDE1DimISJ(data_in, num_grid_points=256, binning_method='linear', extended=N) #Automatic bandwidth
    return kde

fitpars = dict((fittype,{}) for fittype in fittypes)

dict_flat = pkl.load(open('/exp/mu2e/data/users/sdittmer/SignalShape/skimmed_flat_mom_v2.pkl','rb'))

for version in ['best']:#, 'perfect']: #TODO
    for fittype in fittypes: fitpars[fittype][version] = {}
    if version == 'best':
        data_raw = processor.process_data(file_name = "/exp/mu2e/data/users/sdittmer/SignalShape/ntuple_CeMLeadingLog.root", branches = my_branches)
        data_sel = analyse.execute(data_raw, "/exp/mu2e/data/users/sdittmer/SignalShape/ntuple_CeMLeadingLog.root")
    else:
        data_raw = processor.process_data(file_name = "/exp/mu2e/data/users/sdittmer/SignalShape/ntuple_CeMLeadingLog_perfect.root", branches = my_branches)
        data_sel = analyse.execute(data_raw, "/exp/mu2e/data/users/sdittmer/SignalShape/ntuple_CeMLeadingLog_perfect.root")

    # Clean up
    del data_raw
    gc.collect()

    for sid, plane in enumerate(['entrance','middle','exit']):
        # select only track at plane to fit to
        selector = Select()
        at_plane_reco = selector.select_surface(data_sel['trkfit'], sid=sid, branch_name="trksegs")
        vector = Vector()
        trk_mom = vector.get_mag(data_sel['trkfit']["trksegs"].mask[(at_plane_reco)],'mom')
        trk_mom = ak.nan_to_none(trk_mom)
        trk_mom = ak.drop_none(trk_mom)
        trk_mom = np.array(ak.flatten(trk_mom,axis=None))
        print(f'Number of tracks: {len(trk_mom)}')
        del selector
        del at_plane_reco

        obs_mom = zfit.Space('x',95.,105.)
        x_mom = np.linspace(obs_mom.v1.lower, obs_mom.v1.upper, 280)
        data = zfit.Data(data=trk_mom, obs=obs_mom)
        nbins_plot = int((obs_mom.v1.upper - obs_mom.v1.lower)/binwidth_plot)
        
        fitted_dist = {}
        times = {}
        for fittype in fittypes:
            
            outstring = f'# Fitting {version} {plane} {fittype} #'
            print('#'*len(outstring))
            print(outstring)
            print('#'*len(outstring))

            start_time = time.time()

            if fittype == 'GCB':
                fitted_dist[fittype], fit_result, fit_errors = doGCBFit(data,obs_mom)
                names = [par.name for par in fit_result.params]
                vals  = fit_result.values
                if fit_errors is not None:
                    errdo = [fit_errors[par]['lower'] for par in fit_result.params]
                    errup = [fit_errors[par]['upper'] for par in fit_result.params]
                else:
                    errdo = [0.005] * len(names)
                    errup = [0.005] * len(names)
                print(fit_result)
                fitpars[fittype][version][sid] = dict(zip(names,zip(vals,errdo,errup)))
                fitpars[fittype][version][sid].pop('N')
            elif fittype == 'KDE':
                fitted_dist['KDE'] = doKDEFit(data,'grid','adaptive_zfit')
                fitpars['KDE'][version][sid] = trk_mom
            else: 
                losstype, restype, constrain = fittype.split('_')
                inputs_dict = {'theo' : {'params' : binned_spectrum_CeLL()},
                               'eff'  : {'params' : pkl.load(open('../common/efficiency.pkl','rb'))},
                               'loss' : {'pdf' : losstype, 'treat_params' : constrain},
                               'res'  : {'pdf' : restype, 'treat_params' : constrain}}

                if constrain == 'simul':
                    inputs_dict['loss']['simul_source'] = dict_flat[plane]['gen'],dict_flat[plane]['mc']
                    inputs_dict['res']['simul_source']  = dict_flat[plane]['mc'], dict_flat[plane]['reco']
                    p_bins = [95., 97., 99., 101., 103., 105.] #TODO drop hard code
                    fitpars_res = {}
                    fitpars_loss = {}
                    for ip in range(len(p_bins)-1):
                        fitpars_res[(p_bins[ip],p_bins[ip+1])] = {
                            f"mu{ip}"     : (0.0, -3.0, 0.5),
                            f"sigmaL{ip}" : (0.5,  0.0, 2.0),
                            f"sigmaR{ip}" : (0.5,  0.0, 2.0),
                            f"alphaL{ip}" : (0.5,  0.0, 3.0),
                            f"alphaR{ip}" : (0.5,  0.0, 3.0),
                            f"nL{ip}"     : (2.0,  0.0, 12.0),
                            f"nR{ip}"     : (2.0,  0.0, 12.0)}
                        fitpars_loss[(p_bins[ip],p_bins[ip+1])] = {
                            f"loc{ip}"   : (0.0, -5.0, 5.0),
                            f"scale{ip}" : (1.0,  0.0, 5.0)}
                    inputs_dict['loss']['params'] = fitpars_loss
                    inputs_dict['res']['params']  = fitpars_res
                else:
                    inputs_dict['loss']['params'] = pkl.load(open(f'../common/fitpars_flat_loss_{plane}_{losstype}_unbinned.pkl','rb'))
                    inputs_dict['res']['params'] = pkl.load(open(f'../common/fitpars_flat_res_{plane}_{restype}.pkl','rb'))
                fitted_dist[fittype], fit_result, fit_errors = doConvFit(data,obs_mom,inputs_dict,f"{plane}_{fittype}")
                print(fit_result)
                del inputs_dict
                gc.collect()

            end_time = time.time()
            times[fittype] = end_time - start_time
            
        print('Plotting...')
                            
        # Now plot
        data_hist = data.to_binned(int(nbins_plot))
        fig,ax = plt.subplots(1,1,figsize=(5.2,4.8))
        mplhep.histplot(data_hist,ax=ax)
        
        counts = np.array(data_hist.values())
        weights2 = np.where(counts==0.,1.0,counts)
        
        for fittype in fitted_dist.keys():
            ax.plot(x_mom, fitted_dist[fittype].ext_pdf(x_mom)*binwidth_plot, label=f"{fittype.split('_')[-1]}")
            # Manually calculate reduced chi-square
            curve = fitted_dist[fittype].ext_pdf(np.array(*data_hist.binning.centers))*binwidth_plot
            chisq = np.sum(np.square(counts-curve)/weights2)
            redchi = chisq/(len(counts)-npar)
            print(f'{version} {plane} {fittype} redchi: {redchi} duration: {times[fittype]}')

        ax.set_xlabel('Momentum [MeV/c]')
        ax.set_ylabel('Number of Tracks')
        ax.set_yscale('log')
        ax.set_ylim([1.0,None])
        plt.legend()
        plt.savefig(f'plots_pyutils/{version}_{plane}_{fitcat}.png')
        plt.clf()

for fittype in fittypes:
    with open(f'../common/fitpars_{fittype}.pkl','wb') as f:
        pkl.dump(fitpars[fittype],f)
