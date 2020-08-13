## Implementation of variable-criterion word recognition model as described in
## Rotello & Zeng (2008)

# standard library imports
import shelve
from collections import namedtuple
# pylab stack imports
import pylab as pl
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
    rts = (e_a + e_b * pl.exp(e_c*abs(o_dists)))/1000
    # 4. Compute the old/new and remember/know categories
    is_old = s > crit_o
    is_remember = s > crit_r
    # 5. Compute confidence levels
    confs = pl.zeros(pl.shape(rts),dtype=int)
    confs[s>conf1] = 1
    confs[s>conf2] = 2
    # 6. Convert simulated data to 'empirical' format
    kh_idx = array_and(is_target,is_old,array_not(is_remember))
    rh_idx = array_and(is_target,is_old,is_remember)
    kfa_idx = array_and(array_not(is_target),is_old,array_not(is_remember))
    rfa_idx = array_and(array_not(is_target),is_old,is_remember)
    cr_idx = array_and(array_not(is_target),array_not(is_old))
    miss_idx = array_and(is_target,array_not(is_old))

    category_indices = [kh_idx,rh_idx,kfa_idx,rfa_idx,cr_idx,miss_idx]
    categories = []
    for idxs in category_indices:
        category = [RTConf(rt,conf) for rt,conf in zip(rts[idxs],confs[idxs])]
        categories.append(category)
    
    simulated_results = ERStruct(*categories)
    # return the simulated data
    return simulated_results

    

def vc_model_NLL(data,params):
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
    rts,confs,is_target,is_old,is_remember = data
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
    return -LL

