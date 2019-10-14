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
import caa_model.fftw_test as fftw
from caa_model.multinomial_funcs import multinom_loglike,chi_square_gof
import matplotlib.pyplot as plt
data_path = 'caa_model/data/' # this is the base path for the data files

## Start by reading in the data.
# the reason to do this first is that, in order to be efficient,
# we don't want to represent any more of the time axis than we have to.

# Read in new Vincentized RT data
db = shelve.open(data_path+'neha_data.dat','r')
DATA = db['empirical_results'] 
db.close()

USE_FFTW    = True
INF_PROXY   = 10 # a value used to provide very large but finite bounds for mvn integration
EPS         = 1e-10 # a very small value (used for numerical stability)
NR_THREADS  = 4    # this is for multithreaded fft
DELTA_T     = 0.025  # size of discrete time increment (sec.)
MAX_T       = 8.0 #ceil(percentile(all_RT,99.5))
NR_TSTEPS   = int(MAX_T/DELTA_T)
NR_SSTEPS   = 892
NR_SAMPLES  = 10000 # number of trials to use for MC likelihood computation

NR_QUANTILES=10

fftw.fftw_setup(pl.zeros(NR_SSTEPS),NR_THREADS)

# 12/24/2016: modified (for flexibility) to use named tuple instead of list
# DPMParams defines the overall parametrs for the set of old and new words
# combined. 
DPMParams = namedtuple('DPMParams',['c_r','c_f','mu_r','mu_f','d_r','d_f','tc_bound_r',
                    'tc_bound_f','z0_r','z0_f','mu_r_new','mu_f_new','deltaT','t_offset'])

params_est = DPMParams(0.9984,0.9984,0.1035,0.1035,0.1736,0.1736,0.0585,0.0586,-0.1306,-0.1306,
                       -0.2438,-0.2438,0.5859,0.5126)
# new params fitted using 10 quantiles: chisq = 794

params_est_spm = [1.0,0.3103,0.4169,-0.269,0.0458,-0.1284,0.5223,0.5656,0.0]
# parameters fit using spm submodel w/ 10 quantiles. chisq = 828

param_bounds = DPMParams((0.0,1.0),(0.0,1.0),(-2.0,2.0),(-2.0,2.0),(EPS,1.0),(EPS,1.0),
    (0.05,1.0),(0.0,1.0),(-1.0,1.0),(-1.0,1.0),(-2.0,2.0),(-2.0,2.0),(EPS,2.0),(0,0.5))


def find_ml_params_all(quantiles=NR_QUANTILES):
    """
    does a global maximum-likelihood parameter search, constrained by the bounds
    listed in param_bounds, and returns the result. Each RT distribution (i.e.,
    for each judgment category and confidence level) is represented using the
    number of quantiles specified by the 'quantiles' parameter.
    """
    obj_func = lambda x:compute_gof_all(x,quantiles)
    return optimize.differential_evolution(obj_func,param_bounds)

def find_ml_params_all_lm(quantiles=NR_QUANTILES,spm=False):
    """
    computes MLE of params using a local (fast) and unconstrained optimization
    algorithm. Each RT distribution (i.e., for each judgment category and
    confidence level) is represented using the number of quantiles specified by
    the 'quantiles' parameter.
    """
    obj_func = lambda x:compute_gof_all(x,quantiles,spm)
    # computes mle of params using a local (fast) optimization algorithm
    if(spm):
        return optimize.fmin(obj_func,params_est_spm)
    else:
        return optimize.fmin(obj_func,params_est)

def compute_gof_all(model_params,quantiles=NR_QUANTILES,spm=False,data=DATA,use_chisq=True):
    """
    computes the overall goodness-of-fit of the model defined by model_params.
    This is the sum of the NLL or chi-square statistics for the distribution
    of responses to both the old and new words.
    """
    # unpack the model parameters
    #c,mu_old,d,mu_new,tc_bound,z0,deltaT,t_offset = model_params
    if(spm):
        c,mu_f,d_f,mu_f0,tc_bound,z0,deltaT,t_offset,r_bound = model_params
        mu_r = mu_r0 = 0
        d_r = EPS
    else:
        c_r,c_f,mu_r,mu_f,d_r,d_f,tc_bound_r,tc_bound_f,z0_r,z0_f,mu_r0,mu_f0,deltaT,t_offset = model_params
    params_est_old = [[c_r,0],[c_f,0],mu_r,mu_f,d_r,d_f,tc_bound_r,tc_bound_f,z0_r,z0_f,deltaT,t_offset]
    params_est_new = [[c_r,0],[c_f,0],mu_r0,mu_f0,d_r,d_f,tc_bound_r,tc_bound_f,z0_r,z0_f,deltaT,t_offset]
    old_data = [data.rem_hit.rt,data.know_hit.rt,data.miss.rt,data.rem_hit.conf,data.know_hit.conf]
    new_data = [data.rem_fa.rt,data.know_fa.rt,data.CR.rt,data.rem_fa.conf,data.know_fa.conf]
    # compute the combined goodness-of-fit
    res = compute_model_gof(params_est_old,*old_data,nr_quantiles=quantiles,use_chisq=use_chisq)\
    + compute_model_gof(params_est_new,*new_data,nr_quantiles=quantiles,use_chisq=use_chisq)
    return res

    
def compute_prob_all(model_params,quantiles=NR_QUANTILES,spm=False,data=DATA):  
    """
    computes the overall goodness-of-fit of the model defined by model_params.
    This is the sum of the NLL or chi-square statistics for the distribution
    of responses to both the old and new words.
    """
    # unpack the model parameters
    #c,mu_old,d,mu_new,tc_bound,z0,deltaT,t_offset = model_params
    if(spm):
        c,mu_f,d_f,mu_f0,tc_bound,z0,deltaT,t_offset,r_bound = model_params
        mu_r = mu_r0 = 0
        d_r = EPS
    else:
        c_r,c_f,mu_r,mu_f,d_r,d_f,tc_bound_r,tc_bound_f,z0_r,z0_f,mu_r0,mu_f0,deltaT,t_offset = model_params
    params_est_old = [[c_r,0],[c_f,0],mu_r,mu_f,d_r,d_f,tc_bound_r,tc_bound_f,z0_r,z0_f,deltaT,t_offset]
    params_est_new = [[c_r,0],[c_f,0],mu_r0,mu_f0,d_r,d_f,tc_bound_r,tc_bound_f,z0_r,z0_f,deltaT,t_offset]
    old_data = [data.rem_hit.rt,data.know_hit.rt,data.miss.rt,data.rem_hit.conf,data.know_hit.conf]
    new_data = [data.rem_fa.rt,data.know_fa.rt,data.CR.rt,data.rem_fa.conf,data.know_fa.conf]
    # compute the combined goodness-of-fit

    p_obs_old, p_pred_old = compute_model_prop(params_est_old,*old_data,nr_quantiles=quantiles)
    p_obs_new, p_pred_new = compute_model_prop(params_est_new,*new_data,nr_quantiles=quantiles)
    p_obs = pl.hstack([p_obs_old,p_obs_new])
    p_pred = pl.hstack([p_pred_old,p_pred_new])
    return p_obs, p_pred

def compute_model_gof(model_params,rem_RTs,know_RTs,new_RTs,rem_conf,know_conf,nr_quantiles=NR_QUANTILES,use_chisq=True):
    # computes the chi square fit of the model to the data
    # compute N, the total number of trials
    N = len(rem_RTs)+len(know_RTs)+len(new_RTs)
    # compute x, the observed frequency for each category
    rem_quantiles,know_quantiles,new_quantiles,p_r,p_k,p_n = compute_model_quantiles(model_params,nr_quantiles)
    # determine the number of confidence levels being used in the model
    nr_conf_levels = len(rem_quantiles)
    # adjust the number of confidence levels in the data to match
    rem_conf = pl.clip(rem_conf,0,nr_conf_levels-1)
    know_conf = pl.clip(know_conf,0,nr_conf_levels-1)
    ## compute the number of RTs falling into each quantile bin
    rem_freqs = pl.array([-pl.diff([pl.sum(rem_RTs[rem_conf==i]>q) for q in rem_quantiles[i]]+[0]) for i in range(nr_conf_levels)])
    know_freqs = pl.array([-pl.diff([pl.sum(know_RTs[know_conf==i]>q) for q in know_quantiles[i]]+[0]) for i in range(nr_conf_levels)])
    ## Added 11/11/2016 by Melchi
    ## flip these frequencies so that they represent the frequencies in order of
    ## descending confidence levels
    rem_freqs = pl.flipud(rem_freqs)
    know_freqs = pl.flipud(know_freqs)
    new_freqs = -pl.diff([pl.sum(new_RTs>q) for q in new_quantiles]+[0])
    x = pl.hstack([rem_freqs.flatten(),know_freqs.flatten(),new_freqs])
    # compute p, the probability associated with each category in the model
    p_rem = p_r[:,None]*pl.ones((nr_conf_levels,nr_quantiles))/float(nr_quantiles)
    p_know = p_k[:,None]*pl.ones((nr_conf_levels,nr_quantiles))/float(nr_quantiles)
    p_new = p_n*pl.ones(nr_quantiles)/float(nr_quantiles)
    p = pl.hstack([p_rem.flatten(),p_know.flatten(),p_new])
    if(use_chisq):
        return chi_square_gof(x,N,p)
    else: # use NLL
        return -multinom_loglike(x,N,p)
    
def compute_model_prop(model_params,rem_RTs,know_RTs,new_RTs,rem_conf,know_conf,nr_quantiles=NR_QUANTILES):
    # computes the chi square fit of the model to the data
    # compute N, the total number of trials
    N = len(rem_RTs)+len(know_RTs)+len(new_RTs)
    # compute x, the observed frequency for each category
    rem_quantiles,know_quantiles,new_quantiles,p_r,p_k,p_n = compute_model_quantiles(model_params,nr_quantiles)
    # determine the number of confidence levels being used in the model
    nr_conf_levels = len(rem_quantiles)
    # adjust the number of confidence levels in the data to match
    rem_conf = pl.clip(rem_conf,0,nr_conf_levels-1)
    know_conf = pl.clip(know_conf,0,nr_conf_levels-1)
    ## compute the number of RTs falling into each quantile bin
    rem_freqs = pl.array([-pl.diff([pl.sum(rem_RTs[rem_conf==i]>q) for q in rem_quantiles[i]]+[0]) for i in range(nr_conf_levels)])
    know_freqs = pl.array([-pl.diff([pl.sum(know_RTs[know_conf==i]>q) for q in know_quantiles[i]]+[0]) for i in range(nr_conf_levels)])
    ## Added 11/11/2016 by Melchi
    ## flip these frequencies so that they represent the frequencies in order of
    ## descending confidence levels
    rem_freqs = pl.flipud(rem_freqs)
    know_freqs = pl.flipud(know_freqs)
    new_freqs = -pl.diff([pl.sum(new_RTs>q) for q in new_quantiles]+[0])
    x = pl.hstack([rem_freqs.flatten(),know_freqs.flatten(),new_freqs])
    # compute p, the probability associated with each category in the model
    p_rem = p_r[:,None]*pl.ones((nr_conf_levels,nr_quantiles))/float(nr_quantiles)
    p_know = p_k[:,None]*pl.ones((nr_conf_levels,nr_quantiles))/float(nr_quantiles)
    p_new = p_n*pl.ones(nr_quantiles)/float(nr_quantiles)
    p = pl.hstack([p_rem.flatten(),p_know.flatten(),p_new])

    p_pred = p
    p_obs = x/N
    return p_obs,p_pred

def compute_model_quantiles(params,nr_quantiles=NR_QUANTILES):
    # This function is set up to deal with multiple confidence levels
    quantile_increment = 1.0/nr_quantiles
    quantiles = pl.arange(0,1,quantile_increment)
    # compute marginal distributions
    p_remember,p_know,p_new,t = predicted_proportions(*params)
    # compute marginal category proportions (per confidence level)
    remember_total = p_remember.sum(-1)+EPS
    know_total = p_know.sum(-1)+EPS
    new_total = p_new.sum()+EPS
    # compute integrals of marginal distributions
    P_r = pl.cumsum(p_remember,-1)/remember_total[:,None]
    P_k = pl.cumsum(p_know,-1)/know_total[:,None]
    P_n = pl.cumsum(p_new)/new_total
    
    # compute RT quantiles (by confidence level for know and rem judgments)
    rem_quantiles = pl.array([t[pl.argmax(P_r>q,-1)] for q in quantiles]).T
    know_quantiles = pl.array([t[pl.argmax(P_k>q,-1)] for q in quantiles]).T
    new_quantiles = pl.array([t[pl.argmax(P_n>q)] for q in quantiles])
    rem_quantiles[:,0] = 0
    know_quantiles[:,0] = 0
    new_quantiles[0] = 0
    # return quantile locations and marginal p(new)
    return rem_quantiles,know_quantiles,new_quantiles,pl.sum(p_remember,1),pl.sum(p_know,1),pl.sum(p_new)

print('now')
def predicted_proportions(c_r,c_f,mu_r,mu_f,d_r,d_f,tc_bound_r,tc_bound_f,z0_r,z0_f,deltaT,
                          t_offset=0,use_fftw=USE_FFTW):
    # make c (the confidence levels) an array in case it is a scalar value
    c_r = pl.array(c_r,ndmin=1)
    c_f = pl.array(c_f,ndmin=1)
    
    n = len(c_r)
    # form an array consisting of the appropriate (upper) integration limits
    clims_r = pl.hstack(([INF_PROXY],c_r,[-INF_PROXY]))
    clims_f = pl.hstack(([INF_PROXY],c_f,[-INF_PROXY]))    
    
    # compute process SD
    sigma_r = pl.sqrt(2*d_r*DELTA_T)
    sigma_f = pl.sqrt(2*d_f*DELTA_T)
#    sigma = pl.sqrt(sigma_r**2+sigma_f**2)

    # compute the correlation for r given r+f
#    rho = sigma_r/sigma

    t = pl.linspace(DELTA_T,MAX_T,NR_TSTEPS) # this is the time axis
    to_idx = pl.argmin((t-t_offset)**2) # compute the index for t_offset
    bound_r = pl.exp(-tc_bound_r*pl.clip(t-t_offset,0,None)) # this is the collapsing bound
    bound_f = pl.exp(-tc_bound_f*pl.clip(t-t_offset,0,None)) # this is the collapsing bound
    
    #mu = (mu_r+mu_f)*DELTA_T # this is the average overall drift rate, with r = 'recall' and f = 'familiar'
    
    # compute the bounding limit of the space domain. This should include at 
    # least 99% of the probability mass when the particle is at the largest 
    # possible bound
    space_lim_r = max(bound_r)+3*sigma_r
    space_lim_f = max(bound_f)+3*sigma_f
    
    delta_s_r = 2*space_lim_r/NR_SSTEPS
    delta_s_f = 2*space_lim_f/NR_SSTEPS
    
    # finally, construct the space axis
    x_r = pl.linspace(-space_lim_r,space_lim_r,NR_SSTEPS)
    x_f = pl.linspace(-space_lim_f,space_lim_f,NR_SSTEPS)

    # compute the diffusion kernel
    kernel_r = stats.norm.pdf(x_r,mu_r,sigma_r) * delta_s_r
    kernel_f = stats.norm.pdf(x_f,mu_f,sigma_f) * delta_s_f
    
    # compute the 2D kernel
    kernel = np.asarray(np.transpose(np.matrix(kernel_r)) * np.matrix(kernel_f))

    # rr,ff = pl.meshgrid(x_r,x_f)
    # kernel2 = stats.norm.pdf(rr,mu_r,sigma_r)*stats.norm.pdf(ff,mu_f,sigma_f)\
    #     * delta_s_r*delta_s_f
    # ... and its Fourier transform. We'll use this to compute FD convolutions
    if(use_fftw):
        ft_kernel = fftw.fft2(kernel)
    else:
        ft_kernel = pl.fft2(kernel)
    
    
    tx = pl.zeros((len(t),len(x_r),len(x_f)))
    
    # Construct arrays to hold RT distributions
    p_old = pl.zeros(pl.shape(t))
    p_new = pl.zeros(pl.shape(t))
    p_rem_conf = pl.zeros((n+1,pl.size(t))) 
    p_know_conf = pl.zeros((n+1,pl.size(t)))
    
    
    ############################################
    ## take care of the first timestep #########
    ############################################
    tx_r_to_idx = stats.norm.pdf(x_r,mu_r+z0_r,sigma_r)*delta_s_r
    tx_f_to_idx = stats.norm.pdf(x_f,mu_f+z0_f,sigma_f)*delta_s_f
    tx[to_idx] = np.transpose(np.matrix(tx_r_to_idx)) * np.matrix(tx_f_to_idx)
    tx[to_idx] = np.asarray(tx[to_idx]) # change matrix back to array
    
    # prob of old / new
    area_fnew_rnew = pl.sum(tx[to_idx][x_r<= -bound_r[to_idx]][:,x_f <= -bound_f[to_idx]])
    area_fnew_rold = pl.sum(tx[to_idx][x_r>=bound_r[to_idx]][:,x_f <= -bound_f[to_idx]])
    area_fold_rnew = pl.sum(tx[to_idx][x_r<= -bound_r[to_idx]][:,x_f >= bound_f[to_idx]])
    area_fold_rold = pl.sum(tx[to_idx][x_r>=bound_r[to_idx]][:,x_f >= bound_f[to_idx]])
    
    area_fold = pl.sum(tx[to_idx][:,x_f >= bound_f[to_idx]])
    area_fnew = pl.sum(tx[to_idx][:,x_f <= -bound_f[to_idx]])
    area_rold = pl.sum(tx[to_idx][x_r>= bound_r[to_idx]])
    area_rnew = pl.sum(tx[to_idx][x_r<= -bound_r[to_idx]])
    
    p_old[to_idx] = area_fold + area_rold - area_fold_rold - 0.5 * area_fnew_rold - 0.5 * area_fold_rnew
    p_new[to_idx] = area_fnew + area_rnew - area_fnew_rnew - 0.5 * area_fnew_rold - 0.5 * area_fold_rnew
    
    # compute STD(r) for the current time
    s_r = sigma_r
    s_f = sigma_f
    # compute STD(r+f) for the current time
    #s_comb = sigma
    # compute E[r|(r+f)]
    #mu_r_cond = mu_r*t[0]+rho*s_r*(bound[0]-t[0]*(mu_r+mu_f))/s_comb
    #mu_r_cond = mu_r*t[to_idx]+(bound[to_idx]-t[to_idx]*(mu_r+mu_f)-z0)*rho**2
    # compute STD[r|(r+f)]
    #s_r_cond = s_r*pl.sqrt(1-rho**2)
    #s_f_cond = s_f*pl.sqrt(1-(sigma_f/sigma)**2)
    
    
    # remove from consideration any particles that already hit the bound    
    b_hit_r = abs(x_r)<bound_r[to_idx]
    b_hit_f = abs(x_f)<bound_f[to_idx]
    b_hit_mat = np.transpose(np.matrix(b_hit_r))*np.matrix(b_hit_f)
    b_hit_mat = np.asarray(b_hit_mat)
    tx[to_idx]*= b_hit_mat

    ############################################################################
    # compute the parameters of the bivariate distribution of particle locations
    # deltaT seconds after old/new decision
    
    mu_r_delta = bound_r[to_idx]+mu_r*deltaT
    mu_f_delta = bound_f[to_idx]+mu_f*deltaT
    
    s2_r_delta = s_r**2+2*d_r*deltaT
    s2_f_delta = s_f**2+2*d_f*deltaT
    
    
    #s2_comb_delta = 2*deltaT*(d_r+d_f)
    #s2_comb_delta = s_r_cond**2+s_f_cond**2+2*deltaT*(d_r+d_f)
    ############################################################################
    for j in range(1,len(clims_r)):
        # Note that the clims appear in descending order, from highest to lowest value
        p_old_rem = area_rold - 0.5*area_fold_rold - 0.5*area_fnew_rold
        p_old_know = area_fold - 0.5*area_fold_rold - 0.5*area_fold_rnew
        
        RLL_cdf = stats.norm.cdf(clims_r[j],loc = mu_r_delta, scale = s2_r_delta)
        RUL_cdf = stats.norm.cdf(clims_r[j-1],loc = mu_r_delta, scale = s2_r_delta)

        KLL_cdf = stats.norm.cdf(clims_f[j],loc = mu_f_delta, scale = s2_f_delta)
        KUL_cdf = stats.norm.cdf(clims_f[j-1],loc = mu_f_delta, scale = s2_f_delta)        
        #print('rem: '+str(p_old_rem))
        #print('low bound: '+str(RLL_cdf))
        #print('high bound: '+ str(RUL_cdf))
        p_rem_conf[j-1,to_idx] = p_old_rem * (RUL_cdf - RLL_cdf)
        p_know_conf[j-1,to_idx] = p_old_know * (KUL_cdf - KLL_cdf)
        
    #######################################
    ## take care of subsequent timesteps ##
    #######################################
    
    for i in range(to_idx+1,len(t)):
        #tx[i] = convolve(tx[i-1],kernel,'same')
        # convolve the particle distribution from the previous timestep
        # with the diffusion kernel (using Fourier domain convolution)
        if(use_fftw):
            tx[i] = abs(pl.ifftshift(fftw.ifft2(fftw.fft2(tx[i-1])*ft_kernel)))
        else:
            tx[i] = abs(pl.ifftshift(pl.ifft2(pl.fft2(tx[i-1])*ft_kernel)))


        # compute the expected value of a particle that just exceeded the bound
        # during the last time interval
        
        print(pl.sum(tx[i]))
        
        # probability of each particle position above the upper bound
        prob_fnew_rnew = tx[i][x_r<= -bound_r[i]][:,x_f <= -bound_f[i]]
        prob_fnew_rold = tx[i][x_r>=bound_r[i]][:,x_f <= -bound_f[i]]
        prob_fold_rnew = tx[i][x_r<= -bound_r[i]][:,x_f >= bound_f[i]]
        prob_fold_rold = tx[i][x_r>=bound_r[i]][:,x_f >= bound_f[i]]
        
        prob_fold = tx[i][:,x_f >= bound_f[i]]
        prob_fnew = tx[i][:,x_f <= -bound_f[i]]
        prob_rold = tx[i][x_r>= bound_r[i]]
        prob_rnew = tx[i][x_r<= -bound_r[i]]
        
        p_old[i] = pl.sum(prob_fold) + pl.sum(prob_rold) - pl.sum(prob_fold_rold) - 0.5 * pl.sum(prob_fnew_rold) - 0.5 * pl.sum(prob_fold_rnew)
        p_new[i] = pl.sum(prob_fnew) + pl.sum(prob_rnew) - pl.sum(prob_fnew_rnew) - 0.5 * pl.sum(prob_fnew_rold) - 0.5 * pl.sum(prob_fold_rnew)
        
        
        # location of each particle position above the upper bound
        x_rold = x_r[x_r >= bound_r[i]]
        x_fold = x_f[x_f >= bound_f[i]]
        
        
        comb_est_r = (pl.dot(pl.sum(prob_rold,axis = 1),x_rold)+EPS)/(pl.sum(pl.sum(prob_rold,axis =1))+EPS)
        comb_est_f = (pl.dot(pl.sum(prob_fold,axis = 0),x_fold)+EPS)/(pl.sum(pl.sum(prob_fold,axis=0))+EPS)
                
        # compute STD(r) for the current time
        s_r = pl.sqrt(2*d_r*t[i])
        s_f = pl.sqrt(2*d_f*t[i])
 
        # remove from consideration any particles that already hit the bound
        b_hit_r = abs(x_r)<bound_r[i]
        b_hit_f = abs(x_f)<bound_f[i]
        b_hit_mat = np.asarray(np.transpose(np.matrix(b_hit_r))*np.matrix(b_hit_f))
        tx[i]*= b_hit_mat
        ############################################################################
        # compute the parameters of the bivariate distribution of particle locations
        # deltaT seconds after old/new decision
        
        mu_r_delta = comb_est_r + mu_r*deltaT
        mu_f_delta = comb_est_f + mu_f*deltaT
        
        s2_r_delta = s_r**2+2*d_r*deltaT
        s2_f_delta = s_f**2+2*d_f*deltaT
        
        
        #s2_comb_delta = 2*deltaT*(d_r+d_f)
        #s2_comb_delta = s_r_cond**2+s_f_cond**2+2*deltaT*(d_r+d_f)
        ############################################################################
        for j in range(1,len(clims_r)):
            # Note that the clims appear in descending order, from highest to lowest value
            p_old_rem = pl.sum(prob_rold) - 0.5*pl.sum(prob_fold_rold) - 0.5*pl.sum(prob_fnew_rold)
            p_old_know = pl.sum(prob_fold) - 0.5*pl.sum(prob_fold_rold) - 0.5*pl.sum(prob_fold_rnew)
            
            RLL_cdf = stats.norm.cdf(clims_r[j],loc = mu_r_delta, scale = s2_r_delta)
            RUL_cdf = stats.norm.cdf(clims_r[j-1],loc = mu_r_delta, scale = s2_r_delta)
    
            KLL_cdf = stats.norm.cdf(clims_f[j],loc = mu_f_delta, scale = s2_f_delta)
            KUL_cdf = stats.norm.cdf(clims_f[j-1],loc = mu_f_delta, scale = s2_f_delta)        
            
            p_rem_conf[j-1,i] = p_old_rem * (RUL_cdf - RLL_cdf)
            p_know_conf[j-1,i] = p_old_know * (KUL_cdf - KLL_cdf)
            
            # print(i)
            # print('p_old_know:')
            # print(p_old_rem)
            # print('KLL:')
            # print(RLL_cdf)
            # print('KUL:')
            # print(RUL_cdf)
        
        
    # compute the marginal distributions for remember and know (i.e., across all confidence levels)
    #p_remember = p_rem_conf.sum(0)
    #p_know = p_know_conf.sum(0)
    return p_rem_conf,p_know_conf,p_new,t
print('exit')

################################################################################
## Plotting functions
################################################################################

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def emp_v_prediction(model_params,data=DATA):
    conf_r,conf_f,mu_r,mu_f,d_r,d_f,tc_bound_r,tc_bound_f,z0_r,z0_f,mu_r0,mu_f0,deltaT,t_offset = model_params
    #mu_f0 = mu_r0 =0
    c_r = [conf_r,0]
    c_f = [conf_f,0]
    params_est_old = [c_r,c_f,mu_r,mu_f,d_r,d_f,tc_bound_r,tc_bound_f,z0_r,z0_f,deltaT,t_offset]
    params_est_new = [c_r,c_f,mu_r0,mu_f0,d_r,d_f,tc_bound_r,tc_bound_f,z0_r,z0_f,deltaT,t_offset]
    
    nr_conf = len(c_r)
    # adjust the number of confidence levels in the data to match number in model
    hr_conf = pl.clip(data.rem_hit.conf,0,nr_conf)
    hk_conf = pl.clip(data.know_hit.conf,0,nr_conf)
    fr_conf = pl.clip(data.rem_fa.conf,0,nr_conf)
    fk_conf = pl.clip(data.know_fa.conf,0,nr_conf)
    
    # flip the arrays below so that the confidence levels appear in descending order
    hr_rts = [data.rem_hit.rt[hr_conf==i] for i in reversed(pl.unique(hr_conf))]
    hk_rts = [data.know_hit.rt[hk_conf==i] for i in reversed(pl.unique(hk_conf))]
    fr_rts = [data.rem_fa.rt[fr_conf==i] for i in reversed(pl.unique(fr_conf))]
    fk_rts = [data.know_fa.rt[fk_conf==i] for i in reversed(pl.unique(fk_conf))]
    
    n_old = len(hr_conf)+len(hk_conf)+len(data.miss.rt)
    n_new = len(fr_conf)+len(fk_conf)+len(data.CR.rt)
    
    # compute predicted proportions
    pp_old = predicted_proportions(*params_est_old)[:-1]
    pp_new = predicted_proportions(*params_est_new)[:-1]
    
    old_data = [hr_rts,hk_rts,data.miss.rt]
    new_data = [fr_rts,fk_rts,data.CR.rt]
    
    return old_data,new_data,pp_old,pp_new,n_old,n_new
    
# plotting the data
def plot_evp_pair(p_dist,e_dist,e_total,col='g'):
    """
    makes a plot comparing the model-generated RT distribution p_dist to an
    the distribution of a sample of empirical (observed) reaction times e_dist.
    e_total represents the size of the sample.
    """
    # plot the histogram for observed data
    t = pl.linspace(DELTA_T,MAX_T,NR_TSTEPS) # this is the time axis
    # compute the prior distribution for the response category
    p_cat = len(e_dist)*1.0/e_total
    ## compute density histogram
    hd,edges = pl.histogram(e_dist,bins=40,range=[0,10],density=True)
    hist_density = pl.hstack([[0],hd])
    pl.plot(edges,hist_density*p_cat,color=col,lw=2,ls='--',drawstyle='steps')
    # note: the division by DELTA_T below is to make sure that you are plotting
    # probability densities (rather than probability masses)
    curve = pl.plot(t,p_dist/DELTA_T,col,lw=2)
    pl.axis([0,t.max(),None,None])
    pl.show()
    return curve
    
def plot_comparison(model_params,nr_conf_bounds=2):
    """
    makes a set of two plots comparing the predictions of a model parameterized
    by model_params to the observed reaction times and confidence judgments.
    The left panel represents the distribution of judgments and RTs for old
    (target) words, while the right panel represents the distribution for new
    (lure) words.
    """
    nr_conf = nr_conf_bounds+1
    old_data,new_data,pp_old,pp_new,n_old,n_new = emp_v_prediction(model_params)
    #nr_conf=len(pp_old[0])
    colors = ['k','r','b','g','c']
    kcolors = ['#ff8080','#8080ff','#80c080']

    # plot comparison for 'old' words
    pl.figure(figsize=(12,5))
    pl.subplot(1,2,1)
    curves = []
    c_idx = 0
    kc_idx = 0
    # 1. plot misses
    axes = plt.gca()
    #axes.set_xlim([0,8])
    #axes.set_ylim([0,0.7])
    pl.xlabel('Reaction time (sec.)')
    pl.ylabel('p(RT,judgment)')
    curve, = plot_evp_pair(pp_old[-1],old_data[-1],n_old,colors[c_idx])
    curves.append(curve)
    c_idx+=1
    # 2. plot hits
    for conf in range(nr_conf):
        axes = plt.gca()
        #axes.set_xlim([0,8])
        #axes.set_ylim([0,0.7])
        pl.xlabel('Reaction time (sec.)')
        pl.ylabel('p(RT,judgment)')
        curve, = plot_evp_pair(pp_old[0][conf],old_data[0][conf],n_old,colors[c_idx])
        curves.append(curve)

        c_idx+=1
        # plot "know" hits
        axes = plt.gca()
        #axes.set_xlim([0,8])
        #axes.set_ylim([0,0.7])
        pl.xlabel('Reaction time (sec.)')
        pl.ylabel('p(RT,judgment)')
        curve, = plot_evp_pair(pp_old[1][conf],old_data[1][conf],n_old,kcolors[kc_idx])
        curves.append(curve)
        kc_idx+=1

    #pl.axis([0,8,0,0.7])
    simpleaxis(pl.gca())
    pl.title('RT Distributions for Target Words')
    pl.xlabel('Reaction time (sec.)')
    pl.ylabel('p(RT,judgment)')

    pl.legend(curves,['new','rem, conf=2','know, conf=2','rem, conf=1',\
              'know, conf=1','rem, conf=0','know, conf=0'],loc='best',\
              prop={'size':'small'})

    
    # plot comparison for 'new' words
    pl.subplot(1,2,2)
    curves = []
    c_idx = 0
    kc_idx = 0
    # 1. plot misses
    axes = plt.gca()
    #axes.set_xlim([0,8])
    #axes.set_ylim([0,0.7])
    pl.xlabel('Reaction time (sec.)')
    pl.ylabel('p(RT,judgment)')
    curve, = plot_evp_pair(pp_new[-1],new_data[-1],n_new,colors[c_idx])
    curves.append(curve)
    c_idx+=1
    # 2. plot hits
    for conf in range(nr_conf):
        axes = plt.gca()
        #axes.set_xlim([0,8])
        #axes.set_ylim([0,0.7])
        pl.xlabel('Reaction time (sec.)')
        pl.ylabel('p(RT,judgment)')
        curve, = plot_evp_pair(pp_new[0][conf],new_data[0][conf],n_new,colors[c_idx])
        curves.append(curve)
        c_idx+=1
        # plot "know" hits
        axes = plt.gca()
        #axes.set_xlim([0,8])
        #axes.set_ylim([0,0.7])
        pl.xlabel('Reaction time (sec.)')
        pl.ylabel('p(RT,judgment)')
        curve, = plot_evp_pair(pp_new[1][conf],new_data[1][conf],n_new,kcolors[kc_idx])
        curves.append(curve)
        kc_idx+=1
    #pl.axis([0,6,0,0.7])
    #simpleaxis(pl.gca())
    pl.title('RT Distributions for Lure Words')
    pl.xlabel('Reaction time (sec.)')


#a = plot_comparison(params_est,nr_conf_bounds=2)
