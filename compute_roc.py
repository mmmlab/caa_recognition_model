# configure for compatibility with Python 3
from __future__ import (absolute_import, division, print_function)
# standard library imports
import shelve
from collections import namedtuple
# scientific library imports
import numpy as np
import pylab as pl
from numpy import array
from scipy import stats
from scipy.integrate import trapz
# third party imports
import yaml

YAML_FILENAME = 'caa_model/data/neha_data_revised.yml';

Phi = stats.norm.cdf
invPhi = stats.norm.ppf


class ROC(object):
    def __init__(self,hit_counts,fa_counts,targ_count,noise_count):
        self.levels = np.arange(len(hit_counts))
        self.hit_counts = hit_counts
        self.fa_counts = fa_counts
        self.targ_count = targ_count
        self.noise_count = noise_count

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

    def plot(self):
        pl.figure(figsize=(4,4))
        # plot diagonal
        pl.plot([0,1],[0,1],'k:')
        # plot ROC points
        hr = self.hit_rates
        far = self.fa_rates
        pl.plot(far,hr,'bo')

def sdt_roc_loglike(roc,params):
    mu = params[0] # signal mean (or distance between means)
    sigma = params[1] # signal sd (or ratio of signal to noise sd)
    crits = params[2:] # criterion locations
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
    return LL


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

    