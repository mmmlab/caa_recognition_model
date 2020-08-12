#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding: utf-8

"""
Created on Thu Aug  6 08:49:03 2020
"""

# configure for compatibility with Python 3
from __future__ import (absolute_import, division, print_function)
# standard library imports
import shelve
from collections import namedtuple
# scientific library imports
import pylab as pl
import numpy as np
from scipy import stats
from scipy import optimize
# local imports
#import caa_model.fftw_test as fftw
#from caa_model.multinomial_funcs import multinom_loglike,chi_square_gof
import matplotlib.pyplot as plt
import math


# Constants 
LURE_MEAN = 0
LURE_SD = 1

# Sample parameters used in Rotello & Zeng (2008)
T_MU = 1.25     # the mean strength for target words
T_SD = 1.25     # the SD of strength for target words
CRIT_O = 0.625  # the old/new criterion
CRIT_MU = 1.7   # the mean of the (variable) remember/know criteria
CRIT_SD = 1     # the SD of the (variable) remember/know criteria
E_A = 0.2       # the y-offset parameter from the mapping function
E_B = 0.12      # the slope parameter from the mapping function
E_C = -0.06      # the exponential parameter from the mapping function
CONF1 = 0.1     # the confidence criterion for the second confidence level
CONF2 = 0.5    # not actually defined in R&Z, who only used two confidence levels
SAMPLE_PARAMS = (T_MU,T_SD,CRIT_MU,CRIT_SD,CRIT_O,E_A,E_B,E_C,CONF1,CONF2)

EPS = 1e-10 # a very small value (used for numerical stability)

DPMParams = namedtuple('DPMParams',['T_MU','T_SD','CRIT_O','CRIT_MU','CRIT_SD','E_A','E_B',
                    'E_C','CONF1','CONF2'])
param_bounds = DPMParams((-2.0,2.0),(EPS,2.0),(-2.0,2.0),(-2.0,2.0),(EPS,2.0),(0,2000),
    (-2000,2000),(-20.0,0.0),(0.0,2.0),(0.0,2.0))


params_est = DPMParams(1.85896148e-01,  4.92577868e-02,  1.31514433e+00,  1.53815593e+00,
        1.89261476e-01,  1.16369299e-02,  1.61317476e+03, -9.99264542e+00,
        1.74521898e-03,  8.27258488e-03)


data_path = 'caa_model/data/'; # this is the base path for the data files

# Mengxue: Read in new Vincentized RT data
db = shelve.open(data_path+'neha_data.dat','r');
data = db['empirical_results']; 
#db.close();

# Mengxue: Combine all the RTs and confidences. RT changed from sec to milliseconds.
rts = np.concatenate((data.rem_hit.rt,data.know_hit.rt, data.rem_fa.rt,
    data.know_fa.rt, data.CR.rt, data.miss.rt))*1000
confs = np.concatenate((data.rem_hit.conf,data.know_hit.conf, data.rem_fa.conf,
    data.know_fa.conf, data.CR.conf, data.miss.conf))

rh_l = len(data.rem_hit.rt)
kh_l = len(data.know_hit.rt)
rf_l = len(data.rem_fa.rt)
kf_l = len(data.know_fa.rt)
cr_l = len(data.CR.rt)
mi_l = len(data.miss.rt)

is_target = pl.zeros(pl.shape(rts),dtype=bool)
is_old = pl.zeros(pl.shape(rts),dtype=bool)
is_remember = pl.zeros(pl.shape(rts),dtype=bool)

is_target[:(rh_l+kh_l)] = True # Set remember hit and know hit = True in is_target
is_target[-mi_l:] = True # Set miss = True in is_target

is_old[:(rh_l+kh_l+rf_l+kf_l)] = True # Set remember hit, know hit, remember FA, know FA = True in is_old

is_remember[:rh_l] = True # Set remember hit in is_remember
is_remember[(rh_l + kh_l):(rh_l + kh_l+rf_l)]= True # Set remember FA in is_remember

# Mengxue: Combine all the trial outcomes (real data)
val = rts,confs,is_target,is_old,is_remember 


# Functions
def simulate_vc_model(params,n):
    """
    Simulate trials from the variable-criterion model with the provided 
    parameters.

    Arguments:
        params: a tuple or list of the model parameters (see SAMPLE_PARAMS above
        for parameter definitions and order
        n: the number of target trials and lure trials (these are assumed to be
        equal, as in the experiment, giving a total of 2n simulated trials)

    Returns:
        rts: an array of 2n simulated response times, one for each trial, in ms
        confs: an array indicating the confidence rating for each trial
        is_target: a logical array indicating which trials used a target word
        is_old: a logical array indicating which trials were lableled 'old' 
        (i.e., rather than 'new')
        is_remember: a logical array indicating which trials were labeled 
        'remember' (i.e., rather than 'know')  
    """
    # unpack parameters
    t_mu,t_sd,crit_mu,crit_sd,crit_o,e_a,e_b,e_c,conf1,conf2 = params
    # 1. Simulate strengths and criteria for each of n trials
    targets = stats.norm.rvs(t_mu,t_sd,size=n)
    lures = stats.norm.rvs(LURE_MEAN,LURE_SD,size=n)
    # stack the two sets of trials (lures and targets) into a single array
    s = pl.hstack([targets,lures])
    # create a logical array telling us which trials are target trials
    is_target = pl.zeros(pl.shape(s),dtype=bool)
    is_target[:n] = True
    crit_r = stats.norm.rvs(crit_mu,crit_sd,size=2*n)
    # 2. Compute the distance of each strength from the old/new criterion
    o_dists = s-crit_o
    # 3. Convert these into RTs for each trial
    rts = e_a + e_b * pl.exp(e_c*abs(o_dists))
    # 4. Compute the old/new and remember/know categories
    is_old = s > crit_o
    is_remember = s > crit_r
    # 5. Compute confidence levels
    confs = pl.zeros(pl.shape(rts))
    confs[s>conf1] = 1
    confs[s>conf2] = 2
    # return the simulated data
    return rts,confs,is_target,is_old,is_remember


def vc_model_NLL(val,params):
    """
    Compute and return the negative log-likelihood for the variable-criterion
    model with the provided parameters.

    Arguments:
        params: a tuple or list of the model parameters (see SAMPLE_PARAMS above
        for parameter definitions and order
        data: a tuple or list of arrays describing the trial outcomes. The tuple
        should include rts, confs, is_target, is_old, and is_remember as defined
        and in the same format described in the simulate_vc_model function.

    Returns:
        NLL: the negative log-likelihood of the model parameters given the data 
    """
    # unpack data
    rts,confs,is_target,is_old,is_remember = val
    # unpack parameters
    t_mu,t_sd,crit_mu,crit_sd,crit_o,e_a,e_b,e_c,conf1,conf2 = params
    # convert RTs into "distances" by inverting exponential function
    o_dists = pl.log((rts-e_a)/e_b)/e_c
    # ... we also need to make the distances for the 'new' judgments negative
    o_dists[pl.logical_not(is_old)] *= -1
    # convert the distances into word "strengths"
    s = o_dists + crit_o
    # this is the "fragile" part. If any of the strengths are smaller than the 
    # corresponding confidence criteria, then the likelihood will be zero
    conf_crits = pl.zeros(pl.shape(confs))
    conf_crits[confs==1] = conf1
    conf_crits[confs==2] = conf2
    # check for any violations of the confidence boundaries
    violation_found = pl.any(pl.logical_and((s<conf_crits),is_old))
    if violation_found:
        print(pl.logical_and((s<conf_crits),is_old).sum())
        return pl.inf  #i.e., likelihood = 0

    # compute the distribution parameters associated with each item
    s_means = pl.zeros(pl.shape(s))
    s_sds = pl.zeros(pl.shape(s))
    # ... for targets
    s_means[is_target] = t_mu
    s_sds[is_target] = t_sd
    # ... for lures
    s_means[pl.logical_not(is_target)] = LURE_MEAN
    s_sds[pl.logical_not(is_target)] = LURE_SD
    # and use these to compute the log-likelihood associated with each strength
    s_LLs = stats.norm.logpdf(s,s_means,s_sds)
    # for 'old' responses, you also have to compute the (log) probability that
    # the remember/know criterion would be greater than (for 'know' responses) 
    # or less than (for 'remember' responses) the word strength for that trial.
    # Categories:
    ## 'new': just set the likelihood to 1 (log-likelihood=0)
    rk_LLs = pl.zeros(pl.shape(s))
    ## 'old-remember': p(crit_r < s)
    rk_LLs[is_remember] = stats.norm.logcdf(s[is_remember],crit_mu,crit_sd)
    ## 'old-know': p(crit_r > s)
    is_know = pl.logical_not(is_remember)
    rk_LLs[is_know] = stats.norm.logsf(s[is_know],crit_mu,crit_sd)
    
    # Compute the overall log-likelihood (across all trials)
    LL = s_LLs.sum()+rk_LLs.sum()
    
    # Mengxue: exclude the LL that has a value of NaN
    if math.isnan(LL):
        return pl.inf
    print('LL', -LL)

    return -LL


def find_ml_params_all():
    """
    does a global maximum-likelihood parameter search, constrained by the bounds
    listed in param_bounds, and returns the result. Each RT distribution (i.e.,
    for each judgment category and confidence level) is represented using the
    number of quantiles specified by the 'quantiles' parameter.
    """
    obj_func = lambda x:vc_model_NLL(val,x);
    return optimize.differential_evolution(obj_func,param_bounds);



# Mengxue: Generate 200000 simulated trials
rts2,confs2,is_target2,is_old2,is_remember2 = simulate_vc_model(params_est,n=100000)

high_confs2 = confs2 == 2
med_confs2 = confs2 == 1
low_confs2 = confs2 == 0
rh_high2 = rts2[pl.logical_and(pl.logical_and( pl.logical_and(is_target2,is_old2),is_remember2),high_confs2)]*0.001 # target - remember hit with high confidence
rh_med2 = rts2[pl.logical_and(pl.logical_and( pl.logical_and(is_target2,is_old2),is_remember2),med_confs2)]*0.001 # target - remember hit with med confidence
rh_low2 = rts2[pl.logical_and(pl.logical_and( pl.logical_and(is_target2,is_old2),is_remember2),low_confs2)]*0.001 # target - remember hit with low confidence

kh_high2 = rts2[pl.logical_and(pl.logical_and( pl.logical_and(is_target2,is_old2),pl.logical_not(is_remember2)),high_confs2)]*0.001 # target - know hit with high confidence
kh_med2 = rts2[pl.logical_and(pl.logical_and( pl.logical_and(is_target2,is_old2),pl.logical_not(is_remember2)),med_confs2)]*0.001 # target - know hit with med confidence
kh_low2 = rts2[pl.logical_and(pl.logical_and( pl.logical_and(is_target2,is_old2),pl.logical_not(is_remember2)),low_confs2)]*0.001 # target - know hit with low confidence

new2 = rts2[pl.logical_and(is_target2,pl.logical_not(is_old2))]*0.001 # target - new judgment (miss)

# Mengxue: Real data from Neha's experiment
high_confs = confs == 2
med_confs = confs == 1
low_confs = confs == 0

rh_high = rts[pl.logical_and(pl.logical_and( pl.logical_and(is_target,is_old),is_remember),high_confs)]*0.001
rh_med = rts[pl.logical_and(pl.logical_and( pl.logical_and(is_target,is_old),is_remember),med_confs)]*0.001
rh_low = rts[pl.logical_and(pl.logical_and( pl.logical_and(is_target,is_old),is_remember),low_confs)]*0.001

kh_high = rts[pl.logical_and(pl.logical_and( pl.logical_and(is_target,is_old),np.logical_not(is_remember)),high_confs)]*0.001
kh_med = rts[pl.logical_and(pl.logical_and( pl.logical_and(is_target,is_old),np.logical_not(is_remember)),med_confs)]*0.001
kh_low = rts[pl.logical_and(pl.logical_and( pl.logical_and(is_target,is_old),np.logical_not(is_remember)),low_confs)]*0.001

new = rts[pl.logical_and(is_target,pl.logical_not(is_old))]*0.001


# Mengxue: Plot the figures. 
plt.figure()
plt.hist(rh_high,bins=100,density=True, alpha = 0.5, label = 'actual') # actual data
plt.hist(rh_high2,bins=100,density=True, alpha = 0.5, label = 'predict') # simulated data
plt.xlabel("Time", size=14)
plt.ylabel("Density", size=14)
plt.title("RT distribution for target - rem, conf = 2")
plt.legend(loc='upper right')
plt.savefig("target_rem_conf2.png")

plt.figure()
plt.hist(rh_med,bins=100,density=True, alpha = 0.5, label = 'actual')
plt.hist(rh_med2,bins=100,density=True, alpha = 0.5, label = 'predict')
plt.xlabel("Time", size=14)
plt.ylabel("Density", size=14)
plt.title("RT distribution for target - rem, conf = 1")
plt.legend(loc='upper right')
plt.savefig("target_rem_conf1.png")

plt.figure()
plt.hist(rh_low,bins=100,density=True, alpha = 0.5, label = 'actual')
plt.hist(rh_low2,bins=100,density=True, alpha = 0.5, label = 'predict')
plt.xlabel("Time", size=14)
plt.ylabel("Density", size=14)
plt.title("RT distribution for target - rem, conf = 0")
plt.legend(loc='upper right')
plt.savefig("target_rem_conf0.png")

plt.figure()
plt.hist(kh_high,bins=100,density=True, alpha = 0.5, label = 'actual')
plt.hist(kh_high2,bins=100,density=True, alpha = 0.5, label = 'predict')
plt.xlabel("Time", size=14)
plt.ylabel("Density", size=14)
plt.title("RT distribution for target - know, conf = 2")
plt.legend(loc='upper right')
plt.savefig("target_know_conf2.png")

plt.figure()
plt.hist(kh_med,bins=100,density=True, alpha = 0.5, label = 'actual')
plt.hist(kh_med2,bins=100,density=True, alpha = 0.5, label = 'predict')
plt.xlabel("Time", size=14)
plt.ylabel("Density", size=14)
plt.title("RT distribution for target - know, conf = 1")
plt.legend(loc='upper right')
plt.savefig("target_know_conf1.png")

plt.figure()
plt.hist(kh_low,bins=100,density=True, alpha = 0.5, label = 'actual')
plt.hist(kh_low2,bins=100,density=True, alpha = 0.5, label = 'predict')
plt.xlabel("Time", size=14)
plt.ylabel("Density", size=14)
plt.title("RT distribution for target - know, conf = 0")
plt.legend(loc='upper right')
plt.savefig("target_know_conf0.png")

plt.figure()
plt.hist(new,bins=100,density=True, alpha = 0.5, label = 'actual')
plt.hist(new2,bins=100,density=True, alpha = 0.5, label = 'predict')
plt.xlabel("Time", size=14)
plt.ylabel("Density", size=14)
plt.title("RT distribution for target - new")
plt.legend(loc='upper right')
plt.savefig("target_new.png")




