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
import scipy.optimize as opt
from scipy.integrate import trapz
# third party imports
import yaml

YAML_FILENAME = 'caa_model/data/neha_data_revised.yml';

Phi = stats.norm.cdf
invPhi = stats.norm.ppf

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
    targ_LL = stats.multinomial.logpmf(roc.hit_counts,roc.targ_count,targ_probs)
    # ... and for the lures
    lure_LL = stats.multinomial.logpmf(roc.fa_counts,roc.noise_count,lure_probs)
    LL = targ_LL + lure_LL
    return -LL

def fit_sdt_model(roc):
    mu_0 = roc.dprime
    sigma_0 = 1
    crits_0 = np.flipud(roc.criteria)
    params_init = [mu_0,sigma_0]+list(crits_0)
    init_NLL = sdt_roc_NLL(roc,params_init)
    print('NLL for inital parameter estimates = %2.2f'%init_NLL)
    objective = lambda theta:sdt_roc_NLL(roc,theta)
    params = opt.fmin(objective,params_init)
    return params

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
    p_old = np.clip(params[0],0,1) # prob. of classifying a target as old (excluding guesses)
    p_new = np.clip(params[1],0,1) # prob. of classifying a lure as new (excluding guesses)
    biases = list(params[2:])+[1] # probs. of guessing 'old' under different conf levels.
    # compute expected target and lure classification probabilities
    targ_probs = []
    lure_probs = []
    for i,bias in enumerate(biases):
        cum_targ_prob = p_old + bias*(1-p_old)
        cum_lure_prob = bias*(1-p_new)
        if i==0:
            targ_prob = cum_targ_prob
            lure_prob = cum_lure_prob
        else:
            targ_prob = cum_targ_prob - pl.sum(targ_probs)
            lure_prob = cum_lure_prob - pl.sum(lure_probs)
        targ_probs.append(targ_prob)
        lure_probs.append(lure_prob)

    targ_probs = list(reversed(targ_probs))
    lure_probs = list(reversed(lure_probs))

    # use these to compute a mutinomial (log) likelihood for the targets ...
    targ_LL = stats.multinomial.logpmf(roc.hit_counts,roc.targ_count,targ_probs)
    # ... and for the lures
    lure_LL = stats.multinomial.logpmf(roc.fa_counts,roc.noise_count,lure_probs)
    LL = targ_LL + lure_LL
    return -LL

def fit_htm_model(roc):
    # fit a line through the far and hr points to estimate initial params
    m,b = np.polyfit(roc.fa_rates,roc.hit_rates,deg=1)
    p_old_0 = b # i.e., minimum hit rate should be equal to the y intercept
    p_new_0 = 1-((1-b)/m) # i.e., minimum CR rate should be 1 - value of far when hr=1
    biases_0 = (roc.hit_rates-p_old_0)/(1-p_old_0)
    params_init = [p_old_0,p_new_0]+list(biases_0)
    param_bounds = opt.Bounds(0,1)
    init_NLL = htm_roc_NLL(roc,params_init)
    print('NLL for inital parameter estimates = %2.2f'%init_NLL)
    objective = lambda theta:htm_roc_NLL(roc,theta)
    res = opt.minimize(objective,params_init,bounds=param_bounds,method='Nelder-Mead')
    params = res.x
    print('Current function value = %2.4f'%htm_roc_NLL(roc,params))
    return params

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
        return -0.5*(invPhi(self.hit_rates)+invPhi(self.fa_rates))
    
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

    def plot_htm_fit(self):
        params = self.get_htm_params()
        p_old = params[0] # prob. of classifying a target as old (excluding guesses)
        p_new = params[1] # prob. of classifying a lure as new (excluding guesses)
        biases = list(params[2:])# probs. of guessing 'old' under different conf levels.
        bias_ax = pl.linspace(0,1,100)
        hit_rates = p_old + bias_ax*(1-p_old)
        fa_rates = bias_ax*(1-p_new)
        pl.plot(fa_rates,hit_rates,'b-')

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



def get_trial_data():
    # open yaml file
    ifile = open(YAML_FILENAME,'r')
    # read in data string
    filestr = ifile.read()
    # close file
    ifile.close()
    # parse data string into object (list of dicts)
    neha_data = yaml.load(filestr,Loader=yaml.CLoader)

    return neha_data


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
    
def get_rt_crit(trial,rt_quantiles):
    """
    converts confidence to ROC criterion for an individual trial.
    
    it does this by treating high-confidence "old" judgments as most 
    conservative criteria, low-confidence "old" judgments and "new judgments,
    respectively, as intermediate criteria, and high-confidence "new" judgments
    as the most liberal criteria.
    """
    neutral_point = len(rt_quantiles)
    trial_rt = trial['rt.normed']
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

    targ_counts = []
    lure_counts = []
    for conf in possible_levels:
        yes_trials = conf_levels==conf
        nr_targs = np.sum(yes_trials*is_target)
        nr_lures = np.sum(yes_trials*np.logical_not(is_target))
        targ_counts.append(nr_targs)
        lure_counts.append(nr_lures)

    roc_obj = ROC(targ_counts,lure_counts,nr_targ_trials,nr_noise_trials)
    return roc_obj


def get_rt_roc(trial_data,nr_quantiles=3):
    """
    
    """
    trial_rts = np.array([trial['rt.normed'] for trial in trial_data])
    quantile_ranks = (pl.arange(nr_quantiles)+1)/nr_quantiles
    quantiles = pl.quantile(trial_rts,quantile_ranks)
    rt_levels = np.array([get_rt_crit(trial,quantiles) for trial in trial_data])
    is_target = np.array([trial['judgment'] in ['hit','miss'] for trial in trial_data])
    # compute roc coords
    nr_targ_trials = np.sum(is_target)
    nr_noise_trials = np.sum(is_target==False)
    possible_levels = range(rt_levels.max()+1)

    targ_counts = []
    lure_counts = []
    for level in possible_levels:
        yes_trials = rt_levels==level
        nr_targs = np.sum(yes_trials*is_target)
        nr_lures = np.sum(yes_trials*np.logical_not(is_target))
        targ_counts.append(nr_targs)
        lure_counts.append(nr_lures)

    roc_obj = ROC(targ_counts,lure_counts,nr_targ_trials,nr_noise_trials)
    return roc_obj

    
# Script
trial_data = get_trial_data()
conf_roc = get_conf_roc(trial_data,False)
rt_roc = get_rt_roc(trial_data,3)
savepath = 'caa_model/plots/'
# make sure that the savepath exists
if not os.path.exists(savepath):
    os.makedirs(savepath)

conf_roc.plot()
pl.title('Confidence ROCs')
pl.gca().set_aspect('equal')
filename = '%sconf_roc_%d'%(savepath,(len(conf_roc.criteria)+1)/2)
pl.savefig(filename+'.png',transparent=True,dpi=300,bbox_inches='tight',\
            pad_inches=0.05)

rt_roc.plot()
pl.title('Response Time ROCs')
pl.gca().set_aspect('equal')
filename = '%srt_roc_%d'%(savepath,(len(rt_roc.criteria)+1)/2)
pl.savefig(filename+'.png',transparent=True,dpi=300,bbox_inches='tight',\
            pad_inches=0.05)