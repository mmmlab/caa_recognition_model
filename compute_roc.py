# configure for compatibility with Python 3
from __future__ import (absolute_import, division, print_function)
# standard library imports
import shelve
from collections import namedtuple
# scientific library imports
import numpy as np
import pylab as pl
from numpy import array
from scipy.integrate import trapz
# third party imports
import yaml

YAML_FILENAME = 'caa_model/data/neha_data_revised.yml';

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
    adjusted_confidence = trial[conf_key]+1
    # compute criterion
    criterion = neutral_point + adjusted_confidence * ((2*detected)-1)

    return criterion
    
def get_rt_crit(trial,rt_quantiles):
    """
    converts confidence to ROC criterion for an individual trial.
    
    it does this by treating high-confidence "old" judgments as most 
    conservative criteria, low-confidence "old" judgments and "new judgments,
    respectively, as intermediate criteria, and high-confidence "new" judgments
    as the most liberal criteria.
    """
    neutral_point = len(rt_quantiles)+1
    trial_rt = trial['rt.normed']
    # compute rt "rank"
    rt_rank = len(rt_quantiles) - pl.find(rt_quantiles>trial_rt).min()
    # look up old/new 'detection' judgment
    detected = trial['judgment'] in ['hit','FA']
    # compute criterion
    criterion = neutral_point + rt_rank * ((2*detected)-1)


    return criterion
    

def get_conf_roc(use_raw_conf=True):
    """
    
    """
    # open yaml file
    ifile = open(YAML_FILENAME,'r');
    # read in data string
    filestr = ifile.read();
    # close file
    ifile.close();
    # parse data string into object (list of dicts)
    neha_data = yaml.load(filestr,Loader=yaml.CLoader);
    conf_crits = np.array([get_conf_crit(trial,use_raw_conf) for trial in neha_data])
    is_target = np.array([trial['judgment'] in ['hit','miss'] for trial in neha_data])
    # compute roc coords
    nr_targ_trials = np.sum(is_target)
    nr_noise_trials = np.sum(is_target==False)
    possible_crits = range(conf_crits.max())

    FA_rates = []
    hit_rates = []
    for crit in possible_crits:
        yes_trials = conf_crits>=crit
        nr_hits = np.sum(yes_trials*is_target)
        nr_FAs = np.sum(yes_trials*np.logical_not(is_target))
        hit_rate = nr_hits/nr_targ_trials
        FA_rate = nr_FAs/nr_noise_trials
        FA_rates.append(FA_rate)
        hit_rates.append(hit_rate)
    # compute AUC
    # add 0,1 endpoints and flip arrays
    far = pl.flipud([1]+FA_rates+[0])
    hr = pl.flipud([1]+hit_rates+[0])
    auc = trapz(hr,far)
    nr_trials = len(neha_data)

    return hit_rates,FA_rates,auc,nr_trials

def get_rt_roc(nr_quantiles=3):
    """
    
    """
    # open yaml file
    ifile = open(YAML_FILENAME,'r');
    # read in data string
    filestr = ifile.read();
    # close file
    ifile.close();
    # parse data string into object (list of dicts)
    neha_data = yaml.load(filestr,Loader=yaml.CLoader);

    trial_rts = np.array([trial['rt.normed'] for trial in neha_data])
    quantile_ranks = (pl.arange(nr_quantiles)+1)/nr_quantiles
    quantiles = pl.quantile(trial_rts,quantile_ranks)
    rt_crits = np.array([get_rt_crit(trial,quantiles) for trial in neha_data])
    is_target = np.array([trial['judgment'] in ['hit','miss'] for trial in neha_data])
    # compute roc coords
    nr_targ_trials = np.sum(is_target)
    nr_noise_trials = np.sum(is_target==False)
    possible_crits = range(rt_crits.max())
    1/0

    FA_rates = []
    hit_rates = []
    for crit in possible_crits:
        yes_trials = rt_crits>crit
        nr_hits = np.sum(yes_trials*is_target)
        nr_FAs = np.sum(yes_trials*np.logical_not(is_target))
        hit_rate = nr_hits/nr_targ_trials
        FA_rate = nr_FAs/nr_noise_trials
        FA_rates.append(FA_rate)
        hit_rates.append(hit_rate)

    # compute AUC
    # add 0,1 endpoints and flip arrays
    far = pl.flipud([1]+FA_rates+[0])
    hr = pl.flipud([1]+hit_rates+[0])
    auc = trapz(hr,far)
    nr_trials = len(neha_data)


    return hit_rates,FA_rates,auc,nr_trials

    