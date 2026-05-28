import matplotlib.pyplot as plt
import hist as hist
import mplhep
import numpy as np
import zfit
import sys
import uproot
import awkward as ak
import pickle as pkl
from collections import OrderedDict
import gc

sys.path.append('../py-fitter')
from analyze import Analyze
from pyutils.pyprocess import Processor, Skeleton
from pyutils.pyvector import Vector
from pyutils.pyselect import Select

#######################
### User parameters ###
#######################

acbtype     = "gcb"                        # Distribution (GeneralizedCB = gcb or DoubleCB = dcb) to use for fitting 
landau_loss = True                         # Overrides above and uses Landau for loss if True
conv_resloss = True                        # Instead of fitting resolution+loss, compare to res X loss convolution
binwidth_eval = 0.1                        # Binwidth (=sampling frequency) for digitization of PDFs before convolution
p_bins = [95., 97., 99., 101., 103., 105.] # Measure loss / resolution / etc. separately in these bins of 'true' momentum
#p_bins = [95.,105.]
planes = ['entrance','middle','exit'] 

############################
### Define fit functions ###
############################
npar = 1

def doDCBFit(data_in,obs,ip):
    mu     = zfit.Parameter(f"mu{ip}",     0.0, -3.0, 0.5, step_size=0.001)  
    sigma  = zfit.Parameter(f"sigma{ip}",  0.5,  0.0, 2.0, step_size=0.001)
    alphaL = zfit.Parameter(f"alphaL{ip}", 0.5,  0.0, 3.0, step_size=0.001)
    alphaR = zfit.Parameter(f"alphaR{ip}", 0.5,  0.0, 3.0, step_size=0.001)
    nL     = zfit.Parameter(f"nL{ip}",     2.0,  0.0, 12.0, step_size=0.001)
    nR     = zfit.Parameter(f"nR{ip}",     2.0,  0.0, 12.0, step_size=0.001)
    npar = 6
    
    # Define PDF
    acb = zfit.pdf.DoubleCB(obs=obs, mu=mu, sigma=sigma, alphal=alphaL, alphar=alphaR, nl=nL, nr=nR)
    # Create the negative log likelihood
    nll = zfit.loss.UnbinnedNLL(model=acb, data=data_in)  # loss
    # Load and instantiate a minimizer
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss=nll)

    try:
        errors, _ = result.errors(method='minuit_minos')
    except:
        errors = None
        print('WARNING! Invalid fit, postfit parameters may not be optimal')

    return acb,result,errors

def doGCBFit(data_in,obs,ip):
    mu     = zfit.Parameter(f"mu{ip}",     0.0, -3.0, 0.5, step_size=0.001)  
    sigmaL = zfit.Parameter(f"sigmaL{ip}", 0.5,  0.0, 2.0, step_size=0.001)
    sigmaR = zfit.Parameter(f"sigmaR{ip}", 0.5,  0.0, 2.0, step_size=0.001)
    alphaL = zfit.Parameter(f"alphaL{ip}", 0.5,  0.0, 3.0, step_size=0.001)
    alphaR = zfit.Parameter(f"alphaR{ip}", 0.5,  0.0, 3.0, step_size=0.001)
    nL     = zfit.Parameter(f"nL{ip}",     2.0,  0.0, 12.0, step_size=0.001)
    nR     = zfit.Parameter(f"nR{ip}",     2.0,  0.0, 12.0, step_size=0.001)
    npar = 7
    
    # Define PDF
    acb = zfit.pdf.GeneralizedCB(obs=obs, mu=mu, sigmal=sigmaL, sigmar=sigmaR, alphal=alphaL, alphar=alphaR, nl=nL, nr=nR)
    # Create the negative log likelihood
    nll = zfit.loss.UnbinnedNLL(model=acb, data=data_in)  # loss
    # Load and instantiate a minimizer
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss=nll)

    try:
        errors, _ = result.errors(method='minuit_minos')
    except:
        errors = None
        print('WARNING! Invalid fit, postfit parameters may not be optimal')

    return acb,result,errors

from landau_pdf import trunc_landau
def doLandauFit(data_in,obs,ip):

    loc   = zfit.Parameter(f"loc{ip}",   0.0, -5.0, 5.0, step_size=0.001)  
    scale = zfit.Parameter(f"scale{ip}", 1.0,  0.0, 5.0, step_size=0.001)  
    npar = 2
    
    # Define PDF
    landau_pdf = trunc_landau(obs=obs, loc=loc, scale=scale)
    # Create the negative log likelihood
    nll = zfit.loss.UnbinnedNLL(model=landau_pdf, data=data_in)  # loss
    # Load and instantiate a minimizer
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss=nll)
        
    try:
        errors, _ = result.errors(method='minuit_minos')
    except:
        errors = None
        print('WARNING! Invalid fit, postfit parameters may not be optimal')

    return landau_pdf,result,errors

def doConvFit(data_in,obs,respars,losspars):
    zpars = {}
    constraints = []
    obs_res  = zfit.Space('x',-1,1)
    obs_loss = zfit.Space('x',-5,5)
    obs_full = zfit.Space('x',-6,6)
    nbins_res = int((obs_res.v1.upper - obs_res.v1.lower)/binwidth_eval)
    ip = list(respars.keys())[0][-1]

    # Get resolution part
    for p in respars.keys():
        zpars[p[:-1]+'_r'] = zfit.Parameter(p[:-1]+'_r', respars[p][0], respars[p][0]+5*respars[p][1], respars[p][0]+5*respars[p][2],step_size=0.0001)
        constraints.append(zfit.constraint.GaussianConstraint(zpars[p[:-1]+'_r'],observation=respars[p][0],uncertainty=max(abs(respars[p][1]),abs(respars[p][2]))))
    if len(respars.keys()) == 7:
        res_pdf = zfit.pdf.GeneralizedCB(obs=obs_res, mu=zpars['mu_r'], sigmal=zpars['sigmaL_r'], sigmar=zpars['sigmaR_r'], alphal=zpars['alphaL_r'], alphar=zpars['alphaR_r'], nl=zpars['nL_r'], nr=zpars['nR_r'])
    else:
        res_pdf = zfit.pdf.DoubleCB(obs=obs_res, mu=zpars['mu_r'], sigma=zpars['sigma_r'], alphal=zpars['alphaL_r'], alphar=zpars['alphaR_r'], nl=zpars['nL_r'], nr=zpars['nR_r'])

    # Get loss part
    for p in losspars.keys():
        zpars[p[:-1]+'_l'] = zfit.Parameter(p[:-1]+'_l', losspars[p][0], losspars[p][0]+5*losspars[p][1], losspars[p][0]+5*losspars[p][2],step_size=0.0001)
        constraints.append(zfit.constraint.GaussianConstraint(zpars[p[:-1]+'_l'],observation=losspars[p][0],uncertainty=max(abs(losspars[p][1]),abs(losspars[p][2]))))
    if len(losspars.keys()) == 7:
        loss_pdf = zfit.pdf.GeneralizedCB(obs=obs_loss, mu=zpars['mu_l'], sigmal=zpars['sigmaL_l'], sigmar=zpars['sigmaR_l'], alphal=zpars['alphaL_l'], alphar=zpars['alphaR_l'], nl=zpars['nL_l'], nr=zpars['nR_l'])
    elif len(losspars.keys()) == 6:
        loss_pdf = zfit.pdf.DoubleCB(obs=obs_loss, mu=zpars['mu_l'], sigma=zpars['sigma_l'], alphal=zpars['alphaL_l'], alphar=zpars['alphaR_l'], nl=zpars['nL_l'], nr=zpars['nR_l'])
    else:
        loss_pdf = trunc_landau(obs=obs_loss, loc=zpars['loc_l'], scale=zpars['scale_l'])

    # Convolve
    npar = len(respars.keys())+len(losspars.keys())
    resloss_pdf = zfit.pdf.FFTConvPDFV1(loss_pdf, res_pdf, n=nbins_res, obs=obs_loss, norm=obs_full)

    # Create the negative log likelihood
    nll = zfit.loss.UnbinnedNLL(model=resloss_pdf, data=data_in, constraints=constraints)  # loss

    # Load and instantiate a minimizer
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss=nll)
        
    try:
        errors, _ = result.errors(method='minuit_minos')
    except:
        errors = None
        print('WARNING! Invalid fit, postfit parameters may not be optimal')

    return resloss_pdf,result,errors

##################################
### Loop over resolution types ###
##################################

dict_flat = pkl.load(open('/exp/mu2e/data/users/sdittmer/SignalShape/skimmed_flat_mom_v2.pkl','rb'))

fitpars = {}

for restype in ['res','loss','resloss']:
    convs = []
    fracs = []
    qconvs = []
    qfracs = []
    
    obs_gen = zfit.Space('x',p_bins[0],p_bins[-1])
    obs_res = zfit.Space('x',-1,1) if restype == 'res' else zfit.Space('x',-5,5)
    obs_mom = zfit.Space('x',float(obs_gen.v1.lower+obs_res.v1.lower),float(obs_gen.v1.upper+obs_res.v1.upper))
    x_gen = np.linspace(obs_gen.v1.lower, obs_gen.v1.upper, 280)
    x_res = np.linspace(obs_res.v1.lower, obs_res.v1.upper, 280)
    x_mom = np.linspace(obs_mom.v1.lower, obs_mom.v1.upper, 280)
    nbins_gen = int((obs_gen.v1.upper - obs_gen.v1.lower)/binwidth_eval)
    nbins_res = int((obs_res.v1.upper - obs_res.v1.lower)/binwidth_eval)
    nbins_mom = int((obs_mom.v1.upper - obs_mom.v1.lower)/binwidth_eval)

    fig_val, ax_val = plt.subplots(3,3,figsize=(15.6,13.2))
    append = acbtype
    if restype == "resloss" and conv_resloss:
        append = acbtype+"Xlandau" if landau_loss else acbtype+"X"+acbtype
    if restype == "loss" and landau_loss:
        parnames = ["loc","scale"]
        fig_par, ax_par = plt.subplots(2,1,figsize=(10.4,4.8))
        figidx = [0,1]
        append = "landau"
    elif acbtype == "gcb":
        parnames = ["mu","sigmaL","sigmaR","alphaL","alphaR","nL","nR"]
        fig_par, ax_par = plt.subplots(3,3,figsize=(15.6,15.6))
        ax_par[0][0].axis('off')
        ax_par[2][0].axis('off')
        figidx = [1,2,3,4,5,7,8]
    else:
        parnames = ["mu","sigma","alphaL","alphaR","nL","nR"]
        fig_par, ax_par = plt.subplots(3,2,figsize=(15.6,9.6))
        figidx = [0,1,2,3,4,5]

    if len(p_bins) == 2:
        append = append+"_unbinned"
        
    fitpars[restype] = {}
        
    for sid, plane in enumerate(planes):
        if restype == 'res':
            mom_in = dict_flat[plane]['mc']
            mom_out = dict_flat[plane]['reco']
        elif restype == 'loss':
            mom_in = dict_flat[plane]['gen']
            mom_out = dict_flat[plane]['mc']
        else:
            mom_in = dict_flat[plane]['gen']
            mom_out = dict_flat[plane]['reco']
        
        pReco = 'pReco' if 'res' in restype else 'pSurf'
        pTrue = 'pProd' if 'loss' in restype else 'pSurf'

        gc.collect()
  
        convs = []
        fracs = []
        qconvs = []
        qfracs = []

        obs_gen = zfit.Space('x',p_bins[0],p_bins[-1])
        obs_res = zfit.Space('x',-1,1) if restype == 'res' else zfit.Space('x',-5,5)
        obs_mom = zfit.Space('x',float(obs_gen.v1.lower+obs_res.v1.lower),float(obs_gen.v1.upper+obs_res.v1.upper))
        x_gen = np.linspace(obs_gen.v1.lower, obs_gen.v1.upper, 280)
        x_res = np.linspace(obs_res.v1.lower, obs_res.v1.upper, 280)
        x_mom = np.linspace(obs_mom.v1.lower, obs_mom.v1.upper, 280)
        nbins_gen = int((obs_gen.v1.upper - obs_gen.v1.lower)/binwidth_eval)
        nbins_res = int((obs_res.v1.upper - obs_res.v1.lower)/binwidth_eval)
        nbins_mom = int((obs_mom.v1.upper - obs_mom.v1.lower)/binwidth_eval)

        fig_res, ax_res = plt.subplots(2,3,figsize=(15.6,9.6))

        fitpars[restype][plane] = OrderedDict()
        
        # Loop over true momentum bins (e.g. mom_in)
        for ibin in range(len(p_bins)-1):
            in_slice  = mom_in [(mom_in>=p_bins[ibin]) & (mom_in<p_bins[ibin+1])]
            out_slice = mom_out[(mom_in>=p_bins[ibin]) & (mom_in<p_bins[ibin+1])]
            res_slice = out_slice - in_slice
            data = zfit.Data(data=res_slice, obs=obs_res)
            if restype == "resloss" and conv_resloss:
                fitted_dist, fit_result, fit_errors = doConvFit(data,obs_res,fitpars["res"][plane][(p_bins[ibin],p_bins[ibin+1])],fitpars["loss"][plane][(p_bins[ibin],p_bins[ibin+1])])
            if restype == "loss" and landau_loss:
                fitted_dist, fit_result, fit_errors = doLandauFit(data,obs_res,ibin)
            elif acbtype == "gcb":
                fitted_dist, fit_result, fit_errors = doGCBFit(data,obs_res,ibin)
            else:
                fitted_dist, fit_result, fit_errors = doDCBFit(data,obs_res,ibin)

            # Now plot
            data_hist = data.to_binned(40)
            mplhep.histplot(data_hist,ax=ax_res[0][0],label=f'{pTrue} [{p_bins[ibin]},{p_bins[ibin+1]}]')
            mplhep.histplot(data_hist,ax=ax_res[int((ibin+1)/3)][(ibin+1)%3])
            ax_res[int((ibin+1)/3)][(ibin+1)%3].plot(x_res, fitted_dist.pdf(x_res)*(obs_res.v1.upper-obs_res.v1.lower)/40*len(res_slice))
            ax_res[int((ibin+1)/3)][(ibin+1)%3].set_title(f'{pTrue} [{p_bins[ibin]},{p_bins[ibin+1]}]')
            ax_res[int((ibin+1)/3)][(ibin+1)%3].set_xlabel(f'{pReco} - {pTrue}')
            ax_res[int((ibin+1)/3)][(ibin+1)%3].set_ylabel('Number of Tracks')
            
            counts = np.array(data_hist.values())
            weights2 = np.where(counts==0.,1.0,counts)
            curve = fitted_dist.pdf(np.array(*data_hist.binning.centers))*(obs_res.v1.upper-obs_res.v1.lower)/40.*len(res_slice)
            chisq = np.sum(np.square(counts-curve)/weights2)
            redchi = chisq/(len(counts)-npar)
            print(f'{restype} resolution {plane} ({p_bins[ibin]},{p_bins[ibin+1]}) redchi: {redchi}')

            names = [par.name for par in fit_result.params]
            if fit_errors is not None:
                errdo = [fit_errors[par]['lower'] for par in fit_result.params]
                errup = [fit_errors[par]['upper'] for par in fit_result.params]
            else:
                errdo = [-0.005] * len(names)
                errup = [0.005] * len(names)
            vals  = fit_result.values
            print(fit_result)
            fitpars[restype][plane][(p_bins[ibin],p_bins[ibin+1])] = dict(zip(names,zip(vals,errdo,errup)))
        
            # Validate that mom_in X res = mom_out
            h_gen_slice = hist.Hist(hist.axis.Regular(bins=nbins_gen, start=obs_gen.v1.lower, stop=obs_gen.v1.upper, name="x"))
            h_gen_slice.fill(x=in_slice)
            lineshape = zfit.pdf.UnbinnedFromBinnedPDF(zfit.pdf.HistogramPDF(h_gen_slice),obs=obs_gen)
            conv = zfit.pdf.FFTConvPDFV1(lineshape, fitted_dist, n=nbins_res, obs=obs_mom)
            convs.append(conv)
            fracs.append(len(in_slice))
            
            ax_val[sid][0].plot(x_gen, lineshape.pdf(x_gen)*len(in_slice))
            ax_val[sid][1].plot(x_res, fitted_dist.pdf(x_res)*(obs_res.v1.upper-obs_res.v1.lower)/40*len(res_slice),label=f'{pTrue} [{p_bins[ibin]},{p_bins[ibin+1]}]')
            if len(convs) == 1:
                ax_val[sid][2].plot(x_mom, conv.pdf(x_mom)*(obs_mom.v1.upper-obs_mom.v1.lower)/40.*len(in_slice))
            else:
                tot = sum(fracs)
                conv_total = zfit.pdf.SumPDF(convs,fracs=[f/tot for f in fracs],obs=obs_mom,extended=tot)
                ax_val[sid][2].plot(x_mom, conv_total.ext_pdf(x_mom)*(obs_mom.v1.upper-obs_mom.v1.lower)/40.)

        data_reco = zfit.Data(data=mom_out, obs=obs_mom)
        mplhep.histplot(data_reco.to_binned(40),ax=ax_val[sid][2])    
            
        ax_res[0][0].set_xlabel(f'{pReco} - {pTrue}')
        ax_res[0][0].set_ylabel('Number of Tracks')
        ax_res[0][0].legend()
        fig_res.savefig(f'flat_{plane}_{restype}_{append}.png')

        ax_val[sid][0].set_xlabel(f'{pTrue}')
        ax_val[sid][1].set_xlabel(f'{pReco} - {pTrue}')
        ax_val[sid][2].set_xlabel(f'{pReco}')
        ax_val[sid][1].set_title(plane)

        if not (restype == "resloss" and conv_resloss):
            for ipar, par in enumerate(parnames):
                nom   = [ d[f"{par}{ip}"][0] for ip,d in enumerate(fitpars[restype][plane].values())]
                errdo = [-d[f"{par}{ip}"][1] for ip,d in enumerate(fitpars[restype][plane].values())]
                errup = [ d[f"{par}{ip}"][2] for ip,d in enumerate(fitpars[restype][plane].values())]
                idx = range(len(nom))
                ifig = figidx[ipar]
                if restype == "loss" and landau_loss:
                    ax_par[ifig].errorbar(idx,nom,xerr=None,yerr=[errdo,errup],label=planes[sid],linestyle='none',marker='o')
                    ax_par[ifig].set_xticks(idx, labels=[f'({int(k[0])},{int(k[1])})' for k in fitpars[restype][plane].keys()])
                    ax_par[ifig].set_ylabel(par)
                else:
                    ax_par[int(ifig/3)][int(ifig%3)].errorbar(idx,nom,xerr=None,yerr=[errdo,errup],label=planes[sid],linestyle='none',marker='o')
                    ax_par[int(ifig/3)][int(ifig%3)].set_xticks(idx, labels=[f'({int(k[0])},{int(k[1])})' for k in fitpars[restype][plane].keys()])
                    ax_par[int(ifig/3)][int(ifig%3)].set_ylabel(par)

        # Save fitpars
        print(fitpars[restype][plane])
        with open(f'../common/fitpars_flat_{restype}_{plane}_{append}.pkl','wb') as f:
            pkl.dump(fitpars[restype][plane],f)

    ax_val[0][1].legend()
    fig_val.savefig(f'flat_{restype}_val_{append}.png')
    if not (restype == "resloss" and conv_resloss):
        fig_par.legend(loc='upper left')
        fig_par.savefig(f'params_{restype}_{append}.png')
    
