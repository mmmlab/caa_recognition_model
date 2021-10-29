# configure for compatibility with Python 3
from __future__ import (absolute_import, division, print_function)
# standard library imports
import os
import shelve
from collections import namedtuple
# scientific library imports
import numpy as np
import pylab as pl
from numpy import array
from scipy import stats
from scipy.special import gammaln
import scipy.optimize as opt
from scipy.integrate import trapz
# third party imports
import yaml

EXPT1_FILENAME = 'caa_model/data/neha_data_revised.yml';
EXPT2_FILENAME = 'caa_model/data/mengxue_data.yml';

Phi = stats.norm.cdf
invPhi = stats.norm.ppf

def multinom_LL(obs,n,probs):
    """
    computes the log likelihood for a multinomial distribution 
    """
    if np.any(probs<0):
        return -np.inf
    x = np.array(obs)
    p = np.array(probs)
    if any(p==0):
        p += 1.0/(2*n)
    #res = gammaln(n+1)-np.sum(gammaln(x+1))+np.sum(x*np.log(p))
    res = stats.multinomial.logpmf(x,n,p)
    return res

################################################################################
## Define SDT ROC Model
def sdt_roc_NLL(roc,params):
    mu = params[0] # signal mean (or distance between means)
    sigma = params[1] # signal sd (or ratio of signal to noise sd)
    crits = list(params[2:]) # criterion locations
    bin_edges = [-np.inf]+crits+[np.inf]
    # compute expected target and lure classification probabilities
    targ_probs = []
    lure_probs = []
    for i in range(1,len(bin_edges)):
        hi = bin_edges[i]
        lo = bin_edges[i-1]
        lure_prob = Phi(hi)-Phi(lo)
        targ_prob = Phi(hi,mu,sigma) - Phi(lo,mu,sigma)
        targ_probs.append(targ_prob)
        lure_probs.append(lure_prob)
    # use these to compute a mutinomial (log) likelihood for the targets ...
    targ_LL = multinom_LL(roc.hit_counts,roc.targ_count,targ_probs)
    # ... and for the lures
    lure_LL = multinom_LL(roc.fa_counts,roc.noise_count,lure_probs)
    LL = targ_LL + lure_LL
    return -LL

def fit_sdt_model(roc):
    mu_0 = roc.dprime
    sigma_0 = 1
    crits_0 = np.flipud(roc.criteria)
    params_init = [mu_0,sigma_0]+list(crits_0)
    lo_bounds = [0,0] + [-2]*len(crits_0)
    hi_bounds = [10,10] + [20]*len(crits_0)
    param_bounds = opt.Bounds(lo_bounds,hi_bounds)
    init_NLL = sdt_roc_NLL(roc,params_init)
    print('NLL for inital parameter estimates (SDT) = %2.2f'%init_NLL)
    objective = lambda theta:sdt_roc_NLL(roc,theta)
    # compute preliminary global optimization
    #prelim_fit = opt.differential_evolution(objective,param_bounds)
    #params_fit = opt.basinhopping(objective,params_init)
    # compute local optimization
    params_fit = opt.minimize(objective,params_init,bounds=param_bounds,method='Nelder-Mead')
    # params = opt.fmin(objective,prelim_fit.x)
    params = params_fit.x
    return params

def compute_sdt_model(params):
    mu = params[0] # signal mean (or distance between means)
    sigma = params[1] # signal sd (or ratio of signal to noise sd)
    crits = list(params[2:]) # criterion locations
    fa_rates = stats.norm.sf(crits)
    hit_rates = stats.norm.sf(crits,mu,sigma)
    return fa_rates,hit_rates


def plot_sdt_model(params):
    mu = params[0] # signal mean (or distance between means)
    sigma = params[1] # signal sd (or ratio of signal to noise sd)
    crits = list(params[2:]) # criterion locations
    crit_ax = pl.linspace(-2,2*sigma+mu,100)
    fa_rates = stats.norm.sf(crit_ax)
    hit_rates = stats.norm.sf(crit_ax,mu,sigma)
    pl.plot(fa_rates,hit_rates,'b-')

################################################################################
## Define 2HTM ROC Model

def htm_roc_NLL(roc,params):
    params = np.clip(params,0,1)
    p_old = params[0] # prob. of classifying a target as old (excluding guesses)
    p_new = params[1] # prob. of classifying a lure as new (excluding guesses)
    biases = np.array(params[2:]) # probs. of guessing 'old' under different conf levels.
    # compute expected target and lure classification probabilities
    cum_targ_probs = p_old + biases*(1-p_old)
    cum_lure_probs = biases*(1-p_new)

    targ_probs = np.diff(cum_targ_probs,prepend=0,append=1)
    lure_probs = np.diff(cum_lure_probs,prepend=0,append=1)

    # targ_probs = []
    # lure_probs = []
    # for i,bias in enumerate(biases):
    #     # ordered from lowest bias to highest
    #     # or, equivalently, from most conservative to most liberal criterion
    #     # or, equivalently, from highest to lowest confidence
    #     cum_targ_prob = p_old + bias*(1-p_old)
    #     cum_lure_prob = bias*(1-p_new)
    #     if i==0:
    #         targ_prob = cum_targ_prob
    #         lure_prob = cum_lure_prob
    #     else:
    #         targ_prob = cum_targ_prob - pl.sum(targ_probs)
    #         lure_prob = cum_lure_prob - pl.sum(lure_probs)
    #     targ_probs.append(targ_prob)
    #     lure_probs.append(lure_prob)

    #     lure_probs[-1] += p_new

    targ_probs = np.flipud(targ_probs)
    lure_probs = np.flipud(lure_probs)

    # use these to compute a mutinomial (log) likelihood for the targets ...
    targ_LL = multinom_LL(roc.hit_counts,roc.targ_count,targ_probs)
    # ... and for the lures
    lure_LL = multinom_LL(roc.fa_counts,roc.noise_count,lure_probs)
    LL = targ_LL + lure_LL
    return -LL


# def htm_roc_NLL(roc,params):
#     params = np.clip(params,0,1)
#     p_old = params[0] # prob. of classifying a target as old (excluding guesses)
#     p_new = params[1] # prob. of classifying a lure as new (excluding guesses)
#     biases = list(params[2:])+[1] # probs. of guessing 'old' under different conf levels.
#     # compute expected target and lure classification probabilities
#     targ_probs = []
#     lure_probs = []
#     for i,bias in enumerate(biases):
#         cum_targ_prob = p_old + bias*(1-p_old)
#         cum_lure_prob = bias*(1-p_new)
#         if i==0:
#             targ_prob = cum_targ_prob
#             lure_prob = cum_lure_prob
#         else:
#             targ_prob = cum_targ_prob - pl.sum(targ_probs)
#             lure_prob = cum_lure_prob - pl.sum(lure_probs)
#         targ_probs.append(targ_prob)
#         lure_probs.append(lure_prob)

#     targ_probs = np.flipud(targ_probs)
#     lure_probs = np.flipud(lure_probs)

#     # use these to compute a mutinomial (log) likelihood for the targets ...
#     targ_LL = multinom_LL(roc.hit_counts,roc.targ_count,targ_probs)
#     # ... and for the lures
#     lure_LL = multinom_LL(roc.fa_counts,roc.noise_count,lure_probs)
#     LL = targ_LL + lure_LL
#     return -LL

def fit_htm_model(roc):
    # fit a line through the far and hr points to estimate initial params
    m,b = np.polyfit(roc.fa_rates,roc.hit_rates,deg=1)
    p_old_0 = b # i.e., minimum hit rate should be equal to the y intercept
    p_new_0 = 1-((1-b)/m) # i.e., minimum CR rate should be 1 - value of far when hr=1
    # biases_0 = (roc.hit_rates-p_old_0)/(1-p_old_0)
    bias_est_fa = roc.fa_rates/(1-p_new_0)
    bias_est_hit = (roc.hit_rates-p_old_0)/(1-p_old_0)
    biases_0 = 0.5*(bias_est_fa+bias_est_hit)
    params_init = [p_old_0,p_new_0]+list(biases_0)
    param_bounds = opt.Bounds([0]*len(params_init),[1]*len(params_init))
    init_NLL = htm_roc_NLL(roc,params_init)
    print('NLL for inital parameter estimates (2HTM) = %2.2f'%init_NLL)
    objective = lambda theta:htm_roc_NLL(roc,theta)
    # compute preliminary global optimization
    # prelim_fit = opt.differential_evolution(objective,param_bounds)
    # compute local optimization
    params_fit = opt.minimize(objective,params_init,bounds=param_bounds,method='L-BFGS-B')
    # params_fit = opt.basinhopping(objective,res.x)
    
    # params = res.x
    params = params_fit.x
    print('Current function value = %2.4f'%htm_roc_NLL(roc,params))
    return params

def compute_htm_model(params):
    p_old = params[0] # prob. of classifying a target as old (excluding guesses)
    p_new = params[1] # prob. of classifying a lure as new (excluding guesses)
    biases = list(params[2:])# probs. of guessing 'old' under different conf levels.
    hit_rates = p_old + biases*(1-p_old)
    fa_rates = biases*(1-p_new)
    return fa_rates,hit_rates

def plot_htm_model(params):
    p_old = params[0] # prob. of classifying a target as old (excluding guesses)
    p_new = params[1] # prob. of classifying a lure as new (excluding guesses)
    biases = list(params[2:])# probs. of guessing 'old' under different conf levels.
    bias_ax = pl.linspace(0,1,100)
    hit_rates = p_old + bias_ax*(1-p_old)
    fa_rates = bias_ax*(1-p_new)
    pl.plot(fa_rates,hit_rates,'r-')

################################################################################
## Define ROC class
class ROC(object):
    def __init__(self,hit_counts,fa_counts,targ_count,noise_count):
        self.levels = np.arange(len(hit_counts))
        self.hit_counts = hit_counts
        self.fa_counts = fa_counts
        self.targ_count = targ_count
        self.noise_count = noise_count
        self._sdt_params = None
        self._htm_params = None
        self._ax = None

    @property
    def hit_rates(self):
        # compute cumulative hit rates (from level 1 up, since the minimum
        # response is a "sure no")
        hits = np.flipud(self.hit_counts[1:])
        cum_hits = np.cumsum(hits)
        return cum_hits/self.targ_count

    @property
    def fa_rates(self):
        # compute cumulative fa rates (from level 1 up)
        fas = np.flipud(self.fa_counts[1:])
        cum_fas = np.cumsum(fas)
        return cum_fas/self.noise_count
    
    @property
    def nr_trials(self):
        return self.targ_count + self.noise_count

    @property
    def auc(self):
        # compute area under ROC curve
        hr = [0]+self.hit_rates.tolist()+[1]
        far = [0]+self.fa_rates.tolist()+[1]
        AUC = trapz(hr,far)
        return AUC
    
    @property
    def dprime(self):
        return 2*stats.norm.ppf(self.auc)

    @property
    def criteria(self):
        if self._sdt_params is None:
            return -0.5*(invPhi(self.hit_rates)+invPhi(self.fa_rates))
        else:
            mu = self._sdt_params[0]
            sigma = self._sdt_params[1]
            return -0.5*(invPhi(self.hit_rates,mu,sigma)+invPhi(self.fa_rates))
    
    def get_htm_params(self,recompute=False):
        if (self._htm_params is None) or recompute:
            self._htm_params = fit_htm_model(self)
        return self._htm_params
    
    def get_htm_nll(self,recompute=False):
        return htm_roc_NLL(self,self.get_htm_params(recompute))

    def get_sdt_params(self,recompute=False):
        if (self._sdt_params is None) or recompute:
            self._sdt_params = fit_sdt_model(self)
        return self._sdt_params
    
    def get_sdt_nll(self,recompute=False):
        return sdt_roc_NLL(self,self.get_sdt_params(recompute))

    def plot_roc_axes(self):
        pl.figure(figsize=(4,4))
        # plot diagonal
        pl.plot([0,1],[0,1],color='gray',linestyle='dashed')
        pl.axis([-0.01,1.01,-0.01,1.01])
        pl.xlabel('False-alarm rate')
        pl.ylabel('Hit rate')
        self._ax = pl.gca()

    def plot_htm_fit(self,plot_biases=True):
        params = self.get_htm_params()
        p_old = params[0] # prob. of classifying a target as old (excluding guesses)
        p_new = params[1] # prob. of classifying a lure as new (excluding guesses)
        biases = list(params[2:])# probs. of guessing 'old' under different conf levels.
        bias_ax = pl.linspace(0,1,100)
        hit_rates = p_old + bias_ax*(1-p_old)
        fa_rates = bias_ax*(1-p_new)
        pl.plot(fa_rates,hit_rates,'b-')
        if(plot_biases):
            C = np.array(biases)
            HR = p_old+C*(1-p_old)
            FAR = C*(1-p_new)
            pl.plot(FAR,HR,'bs',mfc='none',ms=11)



    def plot_sdt_fit(self):
        params = self.get_sdt_params()
        mu = params[0] # signal mean (or distance between means)
        sigma = params[1] # signal sd (or ratio of signal to noise sd)
        crits = list(params[2:]) # criterion locations
        crit_ax = pl.linspace(-2,2*sigma+mu,100)
        fa_rates = stats.norm.sf(crit_ax)
        hit_rates = stats.norm.sf(crit_ax,mu,sigma)
        pl.plot(fa_rates,hit_rates,'k-')

    def plot(self,recompute=True):
        if (self._ax is None) or recompute:
            self.plot_roc_axes()
        # plot HTM fit
        self.plot_htm_fit()
        # plot SDT fit
        self.plot_sdt_fit()
        # plot ROC data
        hr = self.hit_rates
        far = self.fa_rates
        pl.plot(far,hr,'ko')
        # add nll for SDT model
        pl.text(0.25,0.15,'SDT NLL = %2.2f'%self.get_sdt_nll(),color='black')
        # add nll for HTM model
        pl.text(0.25,0.05,'2HTM NLL = %2.2f'%self.get_htm_nll(),color='blue')



def get_trial_data(filename=EXPT1_FILENAME,subid=None):
    # open yaml file
    ifile = open(filename,'r')
    # read in data string
    filestr = ifile.read()
    # close file
    ifile.close()
    # parse data string into object (list of dicts)
    data = yaml.load(filestr,Loader=yaml.CLoader)
    if subid is not None:
        subj_data = [trial for trial in data if str(trial['subject'])==subid]
        if subj_data==[]:
            1/0
        return subj_data

    return data


def get_conf_crit(trial,use_raw_conf=True):
    """
    converts confidence to ROC criterion for an individual trial.
    
    it does this by treating high-confidence "old" judgments as most 
    conservative criteria, low-confidence "old" judgments and "new judgments,
    respectively, as intermediate criteria, and high-confidence "new" judgments
    as the most liberal criteria.
    """
    if(use_raw_conf):
        conf_key = 'raw.confidence'
        neutral_point = 7
    else:
        conf_key = 'confidence'
        neutral_point = 3
    
    # look up old/new 'detection' judgment
    detected = trial['judgment'] in ['hit','FA']
    # look up confidence rating
    adjusted_confidence = trial[conf_key]
    # compute criterion
    # criterion = neutral_point + adjusted_confidence * ((2*detected)-1)
    criterion = neutral_point + adjusted_confidence if detected else\
                neutral_point - adjusted_confidence - 1

    return criterion
    
def get_rt_crit(trial,rt_quantiles,use_normed=True):
    """
    converts confidence to ROC criterion for an individual trial.
    
    it does this by treating high-confidence "old" judgments as most 
    conservative criteria, low-confidence "old" judgments and "new judgments,
    respectively, as intermediate criteria, and high-confidence "new" judgments
    as the most liberal criteria.
    """
    neutral_point = len(rt_quantiles)
    trial_rt = trial['rt.normed'] if use_normed else trial['rt.secs']
    # compute rt "rank"
    rt_rank = len(rt_quantiles) - pl.find(rt_quantiles>=trial_rt).min() - 1
    # look up old/new 'detection' judgment
    detected = trial['judgment'] in ['hit','FA']
    # compute criterion
    # criterion = neutral_point + rt_rank * ((2*detected)-1)
    criterion = neutral_point + rt_rank if detected else\
                neutral_point - rt_rank - 1

    return criterion
    

def get_conf_roc(trial_data,use_raw_conf=False):
    """
    
    """
    conf_levels = np.array([get_conf_crit(trial,use_raw_conf) for trial in trial_data])
    is_target = np.array([trial['judgment'] in ['hit','miss'] for trial in trial_data])
    # compute roc coords
    nr_targ_trials = np.sum(is_target)
    nr_noise_trials = np.sum(is_target==False)
    possible_levels = range(conf_levels.max()+1)

    hit_counts = []
    fa_counts = []
    # count number of positive responses at each confidence level
    # organized from low-confidence [0] to high-confidence [k]
    # or, equivalently, from liberal to conservative criterion
    for conf in possible_levels:
        yes_trials = conf_levels==conf
        nr_targs = np.sum(yes_trials*is_target)
        nr_lures = np.sum(yes_trials*np.logical_not(is_target))
        hit_counts.append(nr_targs)
        fa_counts.append(nr_lures)

    roc_obj = ROC(hit_counts,fa_counts,nr_targ_trials,nr_noise_trials)
    return roc_obj


def get_rt_roc(trial_data,nr_quantiles=3,use_normed=True):
    """
    
    """
    if use_normed:
        trial_rts = np.array([trial['rt.normed'] for trial in trial_data])
    else:
        trial_rts = np.array([trial['rt.secs'] for trial in trial_data])
    quantile_ranks = (pl.arange(nr_quantiles)+1)/nr_quantiles
    quantiles = pl.quantile(trial_rts,quantile_ranks)
    rt_levels = np.array([get_rt_crit(trial,quantiles,use_normed) for trial in trial_data])
    is_target = np.array([trial['judgment'] in ['hit','miss'] for trial in trial_data])
    # compute roc coords
    nr_targ_trials = np.sum(is_target)
    nr_noise_trials = np.sum(is_target==False)
    possible_levels = range(rt_levels.max()+1)

    hit_counts = []
    fa_counts = []
    for level in possible_levels:
        yes_trials = rt_levels==level
        nr_targs = np.sum(yes_trials*is_target)
        nr_lures = np.sum(yes_trials*np.logical_not(is_target))
        hit_counts.append(nr_targs)
        fa_counts.append(nr_lures)

    roc_obj = ROC(hit_counts,fa_counts,nr_targ_trials,nr_noise_trials)
    return roc_obj

def recovery_test_2htm(params,nr_targs,nr_lures):
    """
    simulate a 2HTM ROC based on the specified parameters and test the ability 
    of the fitting function to recover these parameters
    """
    p_old = params[0] # prob. of classifying a target as old (excluding guesses)
    p_new = params[1] # prob. of classifying a lure as new (excluding guesses)
    biases = list(params[2:])+[1] # probs. of guessing 'old' under different conf levels.
    # generate positive (target) trials
    nr_targ_ht = stats.binom.rvs(nr_targs,p_old)
    nr_targ_guess = nr_targs - nr_targ_ht
    # generate negative (lure) trials
    nr_lure_ht = stats.binom.rvs(nr_lures,p_new)
    nr_lure_guess = nr_lures - nr_lure_ht
    # create random 'guesses' for targets and lures
    targ_guess = pl.rand(nr_targ_guess)
    lure_guess = pl.rand(nr_lure_guess)
    # classify the guesses by confidence
    targ_confs = np.zeros(nr_targ_guess)
    lure_confs = np.zeros(nr_lure_guess)
    for i in range(1,len(biases)):
        hi = biases[i]
        lo = biases[i-1]
        targ_idxs = np.logical_and(targ_guess>lo,targ_guess<=hi)
        lure_idxs = np.logical_and(lure_guess>lo,lure_guess<=hi)
        targ_confs[targ_idxs] = i
        lure_confs[lure_idxs] = i
    # count number of positive responses at each confidence level
    # organized from low-confidence [0] to high-confidence [k]
    # or, equivalently, from high bias to low bias
    # or, equivalently, from liberal to conservative criterion
    hit_confs,hit_counts = np.unique(targ_confs,return_counts=True)
    fa_confs,fa_counts = np.unique(lure_confs,return_counts=True)
    # add in high-threshold (non-guess) trials
    hit_counts[-1] += nr_targ_ht    # non-guess hits
    fa_counts[0] += nr_lure_ht      # non-guess correct rejections
    # construct and return ROC object
    roc_obj = ROC(hit_counts,fa_counts,nr_targs,nr_lures)
    recovered_params = roc_obj.get_htm_params()
    return roc_obj,recovered_params


    
# Script
def computeAndPlotROC(filename=EXPT1_FILENAME,subid=None,use_normed_rts=True):
    trial_data = get_trial_data(filename,subid)
    conf_roc = get_conf_roc(trial_data,False)
    rt_roc = get_rt_roc(trial_data,3,use_normed_rts)
    savepath = 'caa_model/plots/'
    # make sure that the savepath exists
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    conf_roc.plot()
    pl.title('Confidence ROCs')
    pl.gca().set_aspect('equal')
    filename = '%sconf_roc_%d'%(savepath,(len(conf_roc.criteria)+1)/2)
    if subid is not None:
        filename+='_s%s'%subid
    pl.savefig(filename+'.png',transparent=True,dpi=300,bbox_inches='tight',\
                pad_inches=0.05)

    rt_roc.plot()
    pl.title('Response Time ROCs')
    pl.gca().set_aspect('equal')
    filename = '%srt_roc_%d'%(savepath,(len(rt_roc.criteria)+1)/2)
    if subid is not None:
        filename+='_s%s'%subid
    pl.savefig(filename+'.png',transparent=True,dpi=300,bbox_inches='tight',\
                pad_inches=0.05)
    
    return conf_roc,rt_roc

# Script
# make plots for Expt 1
expt1_rocs = computeAndPlotROC()

# # make (individual) plots for Expt 2
# subids = ['%s'%num for num in [1,3,5,7]]
# subj_rocs = []
# for subid in subids:
#     rocs = computeAndPlotROC(EXPT2_FILENAME,subid,False)
#     subj_rocs.append(rocs)


# define and recover parameters from test 2HTM model ROC
nr_targs = 3849
nr_lures = 3886
true_params = [0.36,0.16,0.07,0.32,0.34,0.53,0.8]
model_roc,params_est = recovery_test_2htm(true_params,nr_targs,nr_lures)

