## Implementation of variable-criterion word recognition model as described in
## Rotello & Zeng (2008)

# standard library imports
import shelve
from collections import namedtuple
# pylab stack imports
import pylab as pl
import numpy as np
from scipy import stats
# local imports


# Constants 
LURE_MEAN = 0
LURE_SD = 1

# Sample parameters used in Rotello & Zeng (2008)
T_MU = 1.25     # the mean strength for target words
T_SD = 1.25     # the SD of strength for target words
CRIT_O = 0.625  # the old/new criterion
CRIT_MU = 1.7   # the mean of the (variable) remember/know criteria
CRIT_SD = 1     # the SD of the (variable) remember/know criteria
E_A = 700       # the y-offset parameter from the mapping function
E_B = 450       # the slope parameter from the mapping function
E_C = -0.6      # the exponential parameter from the mapping function
CONF1 = 1.2     # the confidence criterion for the second confidence level
CONF2 = 2.0     # not actually defined in R&Z, who only used two confidence levels
SAMPLE_PARAMS = (T_MU,T_SD,CRIT_MU,CRIT_SD,CRIT_O,E_A,E_B,E_C,CONF1,CONF2)

# empirical results structure
ERStruct = namedtuple('ERStruct',['know_hit','rem_hit','know_fa','rem_fa',
                                  'CR','miss']);
# reaction time & confidence structure
RTConf = namedtuple('RTConf',['rt','conf']);

# Functions
array_not = pl.logical_not

def array_and(*args):
    """
    Computes an elementwise logical 'and' operation across multiple arrays
    """
    return pl.prod(args,0).astype('bool')

def format_as_empirical(data):
    rts_ms,confs,is_target,is_old,is_remember = data
    rts = rts_ms/1000
    # Convert simulated data to 'empirical data' format
    # ...compute logical indices for each response category
    kh_idx = array_and(is_target,is_old,array_not(is_remember))
    rh_idx = array_and(is_target,is_old,is_remember)
    kfa_idx = array_and(array_not(is_target),is_old,array_not(is_remember))
    rfa_idx = array_and(array_not(is_target),is_old,is_remember)
    cr_idx = array_and(array_not(is_target),array_not(is_old))
    miss_idx = array_and(is_target,array_not(is_old))
    category_indices = [kh_idx,rh_idx,kfa_idx,rfa_idx,cr_idx,miss_idx]
    # ...use these indices to create lists of responses, separated by category
    categories = []
    for idxs in category_indices:
        category = RTConf(pl.array(rts[idxs]),pl.array(confs[idxs]))
        categories.append(category)
    # ...group these lists into an ERStruct
    empirical_format = ERStruct(*categories)
    return empirical_format

def format_as_simulation(data):
    rts = pl.concatenate((data.rem_hit.rt,data.know_hit.rt, data.rem_fa.rt,
        data.know_fa.rt, data.CR.rt, data.miss.rt))*1000
    confs = pl.concatenate((data.rem_hit.conf,data.know_hit.conf, data.rem_fa.conf,
        data.know_fa.conf, data.CR.conf, data.miss.conf))
    # compute response counts for each category
    rh_l = len(data.rem_hit.rt)
    kh_l = len(data.know_hit.rt)
    rf_l = len(data.rem_fa.rt)
    kf_l = len(data.know_fa.rt)
    cr_l = len(data.CR.rt)
    mi_l = len(data.miss.rt)
    # convert category membership information into boolean arrays
    is_target = pl.zeros(pl.shape(rts),dtype=bool)
    is_old = pl.zeros(pl.shape(rts),dtype=bool)
    is_remember = pl.zeros(pl.shape(rts),dtype=bool)
    is_target[:(rh_l+kh_l)] = True # Set remember hit and know hit = True in is_target
    is_target[-mi_l:] = True # Set miss = True in is_target
    is_old[:(rh_l+kh_l+rf_l+kf_l)] = True # Set remember hit, know hit, remember FA, know FA = True in is_old
    is_remember[:rh_l] = True # Set remember hit in is_remember
    is_remember[(rh_l + kh_l):(rh_l + kh_l+rf_l)]= True # Set remember FA in is_remember
    # Combine all the trial outcomes
    simulation_format = rts,confs,is_target,is_old,is_remember
    return simulation_format

# vectorized wrappers around existing functions
def tnorm_logpdf1(x,a,b,mu,sigma):
    # standardize a and b (this is what truncnorm expects)
    a_std = (a-mu)/sigma
    b_std = (b-mu)/sigma
    func = np.vectorize(stats.truncnorm.logpdf)
    return func(x,a_std,b_std,mu,sigma)

def tnorm_logcdf1(x,a,b,mu,sigma):
    # standardize a and b (this is what truncnorm expects)
    a_std = (a-mu)/sigma
    b_std = (b-mu)/sigma
    func = np.vectorize(stats.truncnorm.logcdf)
    return func(x,a_std,b_std,mu,sigma)

def tnorm_logsf1(x,a,b,mu,sigma):
    # standardize a and b (this is what truncnorm expects)
    a_std = (a-mu)/sigma
    b_std = (b-mu)/sigma
    func = np.vectorize(stats.truncnorm.logsf)
    return func(x,a_std,b_std,mu,sigma)

def tnorm_logpdf(x,a,b,mu,sigma):
    lnscale = pl.log(stats.norm.cdf(b,mu,sigma) - stats.norm.cdf(a,mu,sigma))
    lnpdf = stats.norm.logpdf(x,mu,sigma)-lnscale
    return lnpdf

def tnorm_logcdf(x,a,b,mu,sigma):
    lnscale = pl.log(stats.norm.cdf(b,mu,sigma) - stats.norm.cdf(a,mu,sigma))
    un_cdf = (stats.norm.cdf(x,mu,sigma)-stats.norm.cdf(a,mu,sigma)).clip(0,1)
    lncdf = pl.log(un_cdf)-lnscale
    return lncdf

def tnorm_logsf(x,a,b,mu,sigma):
    lnscale = pl.log(stats.norm.cdf(b,mu,sigma) - stats.norm.cdf(a,mu,sigma))
    un_sf = (stats.norm.sf(x,mu,sigma)-stats.norm.sf(b,mu,sigma)).clip(0,1)
    lnsf = pl.log(un_sf)-lnscale
    return lnsf

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
    confs = pl.zeros(pl.shape(rts),dtype=int)
    confs[s>conf1] = 1
    confs[s>conf2] = 2
    # return the simulated data
    simulated_data = rts,confs,is_target,is_old,is_remember
    return format_as_empirical(simulated_data)


def vc_model_NLL(data,params):
    """
    Compute and return the negative log-likelihood for the variable-criterion
    model with the provided parameters.

    Arguments:
        params: a tuple or list of the model parameters (see SAMPLE_PARAMS above
        for parameter definitions and order
        data: a tuple or ERStruct of arrays describing the trial outcomes. If 
        using a tuple, it should include rts, confs, is_target, is_old, and 
        is_remember as defined and in the same format described in the 
        simulate_vc_model function.

    Returns:
        NLL: the negative log-likelihood of the model parameters given the data 
    """
    # unpack data
    if(type(data)==tuple):
        rts,confs,is_target,is_old,is_remember = data
    else:
        rts,confs,is_target,is_old,is_remember = format_as_simulation(data)
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
        return pl.inf  #i.e., likelihood = 0

    # compute the distribution parameters associated with each item
    s_means = pl.zeros(pl.shape(s))
    s_sds = pl.zeros(pl.shape(s))
    # ... for lures
    s_means[pl.logical_not(is_target)] = LURE_MEAN
    s_sds[pl.logical_not(is_target)] = LURE_SD
    # ... for targets
    s_means[is_target] = t_mu
    s_sds[is_target] = t_sd
    # Use these to compute the response category probabilities. To do this, we 
    # will also need to define the truncation limits a and b. The default is to 
    # have an unbounded domain
    s_a = pl.zeros(pl.shape(s)); s_a[:] = -pl.inf
    s_b = pl.zeros(pl.shape(s)); s_b[:] = pl.inf
    # ... for 'new' responses
    s_b[pl.logical_not(is_old)] = crit_o
    # ... for targets
    s_a[is_old] = crit_o
    # ... also set bounds for specific confidence levels
    s_b[array_and(is_old,confs==0)] = conf1 # upper-bound for conf = 0
    s_a[array_and(is_old,confs==1)] = conf1 # lower-bound for conf = 1
    s_b[array_and(is_old,confs==1)] = conf2 # upper-bound for conf = 1
    s_a[array_and(is_old,confs==2)] = conf2 # lower-bound for conf = 2
    # compute category log-likelihoods (e.g., for p(old|target), p(new|lure),etc.)
    cat_LLs = pl.log(stats.norm.cdf(s_b,s_means,s_sds)-stats.norm.cdf(s_a,s_means,s_sds))
    # compute the log-likelihood of of the observed RTs/strengths given the 
    # category and confidence level--e.g., p(RT|old,conf,target). We can do this 
    # by using truncated normal distributions whose truncation boundaries are 
    # defined by the category/confidence bounds.
    s_LLs = tnorm_logpdf(s,s_a,s_b,s_means,s_sds)
    # for 'old' responses, you also have to compute the (log) probability that
    # the remember/know criterion would be greater than (for 'know' responses) 
    # or less than (for 'remember' responses) the word strength for that trial.
    # Categories:
    ## 'new': just set the likelihood to 1 (log-likelihood=0)
    rk_LLs = pl.zeros(pl.shape(s))
    ## 'old-remember': p(crit_r < s |old,conf)
    rk_LLs[is_remember] = tnorm_logcdf(s[is_remember],s_a[is_remember],
        s_b[is_remember],crit_mu,crit_sd)
    ## 'old-know': p(crit_r > s |old,conf)
    is_know = pl.logical_not(is_remember)
    rk_LLs[is_know] = tnorm_logsf(s[is_know],s_a[is_know],s_b[is_know],
        crit_mu,crit_sd)
    # Compute the overall log-likelihood (across all trials)
    LL = s_LLs.sum()+rk_LLs.sum()+cat_LLs.sum()
    print(-LL)
    return -LL

# Changes (8/14/2020):
# Changed the code above so that we: 
# 1. First compute the probability of each response category given the 
#   word status as target/lure--i.e., p(old|target), p(new|target);
# 2. Then compute the probability of of the observed RTs given the category
#   and confidence level--e.g., p(RT|old,conf). We can do this by using 
#   truncated normal distributions whose truncation boundaries are defined by 
#   the category/confidence bounds.
# 3. Finally, for 'old' judgements, compute the remember/know probability 
#   conditioned on the RT (2) and category/confidence.

#################################
# model-fitting code
#################################
import scipy.optimize as opt

def find_ml_params(data,param_bounds):
    """
    does a global maximum-likelihood parameter search, constrained by the bounds
    listed in param_bounds, and returns the result. Each RT distribution (i.e.,
    for each judgment category and confidence level) is represented using the
    number of quantiles specified by the 'quantiles' parameter.
    """
    obj_func = lambda x:vc_model_NLL(data,x)
    return opt.differential_evolution(obj_func,param_bounds)

def find_ml_params_lm(data,params_init):
    """
    computes MLE of params using a local (fast) and unconstrained optimization
    algorithm. Each RT distribution (i.e., for each judgment category and
    confidence level) is represented using the number of quantiles specified by
    the 'quantiles' parameter.
    """
    obj_func = lambda x:vc_model_NLL(data,x)
    # computes mle of params using a local (fast) optimization algorithm
    return opt.fmin(obj_func,params_init)

EPS = 1e-10 # a very small value (used for numerical stability)

DPMParams = namedtuple('DPMParams',['T_MU','T_SD','CRIT_O','CRIT_MU','CRIT_SD','E_A','E_B',
                    'E_C','CONF1','CONF2'])
PARAM_BOUNDS = DPMParams((0,2.0),(EPS,10.0),(-2.0,10.0),(-2.0,10.0),(EPS,10.0),(0,2000),
    (-2000,2000),(-20.0,0.0),(0.0,10.0),(0.0,10.0))

sample_data = simulate_vc_model(SAMPLE_PARAMS,1000)
params_est = find_ml_params_lm(sample_data,SAMPLE_PARAMS)