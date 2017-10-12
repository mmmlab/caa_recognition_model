# standard library imports
import shelve
from collections import namedtuple
# scientific library imports
import pylab as pl
import numpy
from scipy import stats
from scipy import optimize
# local imports
import fftw_test as fftw
from multinomial_funcs import multinom_loglike,chi_square_gof
from simpleaxis import simpleaxis

data_path = 'neha/data/'; # this is the base path for the data files

## Start by reading in the data.
# the reason to do this first is that, in order to be efficient,
# we don't want to represent any more of the time axis than we have to.

# Read in new Vincentized RT data
db = shelve.open(data_path+'neha_data.dat','r');
DATA = db['empirical_results']; 
db.close();


INF_PROXY   = 10; # a value used to provide very large but finite bounds for mvn integration
EPS         = 1e-10 # a very small value (used for numerical stability)
NR_THREADS  = 1;    # this is for multithreaded fft
DELTA_T     = 0.025;  # size of discrete time increment (sec.)
MAX_T       = 8.0; #ceil(percentile(all_RT,99.5))
NR_TSTEPS   = MAX_T/DELTA_T;
NR_SSTEPS   = 8192;
NR_SAMPLES  = 10000; # number of trials to use for MC likelihood computation

NR_QUANTILES=10;

fftw.fftw_setup(pl.zeros(NR_SSTEPS),NR_THREADS);

# 12/24/2016: modified (for flexibility) to use named tuple instead of list
# DPMParams defines the overall parametrs for the set of old and new words
# combined. 
DPMParams = namedtuple('DPMParams',['c','mu_r','mu_f','d_r','d_f','tc_bound',
                    'r_bound','z0','mu_r_new','mu_f_new','deltaT','t_offset'])
#c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0, mu_r0,mu_f0,deltaT,t_offset = model_params;
params_est = DPMParams(0.9984,0.002,0.3035,0.0037,0.3736,0.0585,0.001,-0.1306,
                       -0.0142,-0.2438,0.5859,0.5126);
# new params fitted using 10 quantiles: chisq = 794
# parameters taken from single-process model
#params_est_spm = [0.9452,0.0,0.3236,EPS,0.4126,0.0486,0,-0.124,0.0,-0.2745,0.5001,0.5527];

params_est_spm = [1.0,0.3103,0.4169,-0.269,0.0458,-0.1284,0.5223,0.5656,0.0];
# parameters fit using spm submodel w/ 10 quantiles. chisq = 828

param_bounds = DPMParams((0.0,1.0),(-2.0,2.0),(-2.0,2.0),(EPS,1.0),(EPS,1.0),
    (0.05,1.0),(0.0,1.0),(-1.0,1.0),(-2.0,2.0),(-2.0,2.0),(EPS,2.0),(0,0.5));


def find_ml_params_all(quantiles=NR_QUANTILES):
    """
    does a global maximum-likelihood parameter search, constrained by the bounds
    listed in param_bounds, and returns the result. Each RT distribution (i.e.,
    for each judgment category and confidence level) is represented using the
    number of quantiles specified by the 'quantiles' parameter.
    """
    obj_func = lambda x:compute_gof_all(x,quantiles);
    return optimize.differential_evolution(obj_func,param_bounds);

def find_ml_params_all_lm(quantiles=NR_QUANTILES,spm=False):
    """
    computes MLE of params using a local (fast) and unconstrained optimization
    algorithm. Each RT distribution (i.e., for each judgment category and
    confidence level) is represented using the number of quantiles specified by
    the 'quantiles' parameter.
    """
    obj_func = lambda x:compute_gof_all(x,quantiles,spm);
    # computes mle of params using a local (fast) optimization algorithm
    if(spm):
        return optimize.fmin(obj_func,params_est_spm);
    else:
        return optimize.fmin(obj_func,params_est);

def compute_gof_all(model_params,quantiles=NR_QUANTILES,spm=False,data=DATA,use_chisq=True):
    """
    computes the overall goodness-of-fit of the model defined by model_params.
    This is the sum of the NLL or chi-square statistics for the distribution
    of responses to both the old and new words.
    """
    # unpack the model parameters
    #c,mu_old,d,mu_new,tc_bound,z0,deltaT,t_offset = model_params;
    if(spm):
        c,mu_f,d_f,mu_f0,tc_bound,z0,deltaT,t_offset,r_bound = model_params;
        mu_r = mu_r0 = 0;
        d_r = EPS;
    else:
        c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,mu_r0,mu_f0,deltaT,t_offset = model_params;
    params_est_old = [[c,0],mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT,t_offset];
    params_est_new = [[c,0],mu_r0,mu_f0,d_r,d_f,tc_bound,r_bound,z0,deltaT,t_offset];
    old_data = [data.rem_hit.rt,data.know_hit.rt,data.miss.rt,data.rem_hit.conf,data.know_hit.conf];
    new_data = [data.rem_fa.rt,data.know_fa.rt,data.CR.rt,data.rem_fa.conf,data.know_fa.conf];
    # compute the combined goodness-of-fit
    res = compute_model_gof(params_est_old,*old_data,nr_quantiles=quantiles,use_chisq=use_chisq)\
        + compute_model_gof(params_est_new,*new_data,nr_quantiles=quantiles,use_chisq=use_chisq);
    return res;

def compute_model_gof(model_params,rem_RTs,know_RTs,new_RTs,rem_conf,know_conf,nr_quantiles=NR_QUANTILES,use_chisq=True):
    # computes the chi square fit of the model to the data
    # compute N, the total number of trials
    N = len(rem_RTs)+len(know_RTs)+len(new_RTs);
    # compute x, the observed frequency for each category
    rem_quantiles,know_quantiles,new_quantiles,p_r,p_k,p_n = compute_model_quantiles(model_params,nr_quantiles);
    # determine the number of confidence levels being used in the model
    nr_conf_levels = len(rem_quantiles);
    # adjust the number of confidence levels in the data to match
    rem_conf = pl.clip(rem_conf,0,nr_conf_levels-1);
    know_conf = pl.clip(know_conf,0,nr_conf_levels-1);
    ## compute the number of RTs falling into each quantile bin
    rem_freqs = pl.array([-pl.diff([pl.sum(rem_RTs[rem_conf==i]>q) for q in rem_quantiles[i]]+[0]) for i in range(nr_conf_levels)]);
    know_freqs = pl.array([-pl.diff([pl.sum(know_RTs[know_conf==i]>q) for q in know_quantiles[i]]+[0]) for i in range(nr_conf_levels)]);
    ## Added 11/11/2016 by Melchi
    ## flip these frequencies so that they represent the frequencies in order of
    ## descending confidence levels
    rem_freqs = pl.flipud(rem_freqs);
    know_freqs = pl.flipud(know_freqs);
    new_freqs = -pl.diff([pl.sum(new_RTs>q) for q in new_quantiles]+[0]);
    x = pl.hstack([rem_freqs.flatten(),know_freqs.flatten(),new_freqs]);
    # compute p, the probability associated with each category in the model
    p_rem = p_r[:,None]*pl.ones((nr_conf_levels,nr_quantiles))/float(nr_quantiles);
    p_know = p_k[:,None]*pl.ones((nr_conf_levels,nr_quantiles))/float(nr_quantiles);
    p_new = p_n*pl.ones(nr_quantiles)/float(nr_quantiles);
    p = pl.hstack([p_rem.flatten(),p_know.flatten(),p_new]);
    if(use_chisq):
        return chi_square_gof(x,N,p);
    else: # use NLL
        return -multinom_loglike(x,N,p);

def compute_model_quantiles(params,nr_quantiles=NR_QUANTILES):
    # This function is set up to deal with multiple confidence levels
    quantile_increment = 1.0/nr_quantiles;
    quantiles = pl.arange(0,1,quantile_increment);
    # compute marginal distributions
    p_remember,p_know,p_new,t = predicted_proportions(*params);
    # compute marginal category proportions (per confidence level)
    remember_total = p_remember.sum(-1)+EPS;
    know_total = p_know.sum(-1)+EPS;
    new_total = p_new.sum()+EPS;
    # compute integrals of marginal distributions
    P_r = pl.cumsum(p_remember,-1)/remember_total[:,None];
    P_k = pl.cumsum(p_know,-1)/know_total[:,None];
    P_n = pl.cumsum(p_new)/new_total;
    
    # compute RT quantiles (by confidence level for know and rem judgments)
    rem_quantiles = pl.array([t[pl.argmax(P_r>q,-1)] for q in quantiles]).T;
    know_quantiles = pl.array([t[pl.argmax(P_k>q,-1)] for q in quantiles]).T;
    new_quantiles = pl.array([t[pl.argmax(P_n>q)] for q in quantiles]);
    rem_quantiles[:,0] = 0;
    know_quantiles[:,0] = 0;
    new_quantiles[0] = 0;
    # return quantile locations and marginal p(new)
    return rem_quantiles,know_quantiles,new_quantiles,pl.sum(p_remember,1),pl.sum(p_know,1),pl.sum(p_new);


# (10/06/2016) This is the new version of the function, which implements the
# collapsing bound model, but as (effectively) a single-process model
# One major advantage is that we can eliminate at least 3 parameters:
# mu_r, d_r, and r_bound

def predicted_proportions_spm(c,mu_f,d_f,tc_bound,z0,deltaT,use_fftw=True):
    # make c (the confidence levels) an array in case it is a scalar value
    c = pl.array(c,ndmin=1);
    n = len(c);
    # form an array consisting of the appropriate (upper) integration limits
    clims = pl.hstack(([INF_PROXY],c,[-INF_PROXY]));
    # compute process SD
    sigma_f = pl.sqrt(2*d_f*DELTA_T);
    sigma = sigma_f;

    t = pl.linspace(DELTA_T,MAX_T,NR_TSTEPS); # this is the time axis
    bound = pl.exp(-tc_bound*t); # this is the collapsing bound
     
    mu = mu_f*DELTA_T; # this is the expected drift over time interval DELTA_T
    # compute the bounding limit of the space domain. This should include at
    # least 99% of the probability mass when the particle is at the largest possible bound
    space_lim = max(bound)+3*sigma;
    delta_s = 2*space_lim/NR_SSTEPS;
    # finally, construct the space axis
    x = pl.linspace(-space_lim,space_lim,NR_SSTEPS);
    # compute the diffusion kernel
    kernel = stats.norm.pdf(x,mu,sigma)*delta_s;
    # ... and its Fourier transform. We'll use this to compute FD convolutions
    if(use_fftw):
        ft_kernel = fftw.fft(kernel);
    else:
        ft_kernel = pl.fft(kernel);
    tx = pl.zeros((len(t),len(x)));
    
    # Construct arrays to hold RT distributions
    p_old = pl.zeros(pl.shape(t));
    p_new = pl.zeros(pl.shape(t));
    p_old_conf = pl.zeros((n+1,size(t)));  
    
    ############################################
    ## take care of the first timestep #########
    ############################################
    tx[0] = stats.norm.pdf(x,mu+z0,sigma)*delta_s;
    p_old[0] = pl.sum(tx[0][x>=bound[0]]);
    p_new[0] = pl.sum(tx[0][x<=-bound[0]]);
    
    # remove from consideration any particles that already hit the bound
    tx[0]*=(abs(x)<bound[0]);
    
    ############################################################################
    # compute the parameters for the distribution of particle locations
    # deltaT seconds after old/new decision
    mu_delta = mu_f*deltaT+bound[0];
    s_delta = pl.sqrt(2*d_f*deltaT);
    
    # compute the probability that the particle falls within the region for
    # each category. Note that now the particle's trajectory is 1D, so that
    # the only thing that determines the region is the bounding interval,
    # specified in terms of f
    for j in range(1,len(clims)):
        p_old_conf[j-1,0] = p_old[0]*pl.diff(stats.norm.cdf([clims[j],clims[j-1]],mu_delta,s_delta));
        
    #######################################
    ## take care of subsequent timesteps ##
    #######################################
    
    for i in range(1,len(t)):
        # convolve the particle distribution from the previous timestep
        # with the diffusion kernel (using Fourier domain convolution)
        if(use_fftw):
            tx[i] = abs(pl.ifftshift(fftw.ifft(fftw.fft(tx[i-1])*ft_kernel)));
        else:
            tx[i] = abs(pl.ifftshift(pl.ifft(pl.fft(tx[i-1])*ft_kernel)));

        p_pos = tx[i][x>=bound[i]]; # probability of each particle position above the upper bound
        x_pos = x[x>=bound[i]];     # location of each particle position above the upper bound
        
        # compute the expected value of a particle that just exceeded the bound
        # during the last time interval
        comb_est = (pl.dot(p_pos,x_pos)+EPS)/(pl.sum(p_pos)+EPS); 

        p_old[i] = pl.sum(p_pos); # total probability that particle crosses upper bound
        p_new[i] = pl.sum(tx[i][x<=-bound[i]]); # probability that particle crosses lower bound
        
               # remove from consideration any particles that already hit the bound
        tx[i]*=(abs(x)<bound[i]);
        
        # compute the parameters for the distribution of particle locations
        # deltaT seconds after old/new decision
        mu_delta = mu_f*deltaT+bound[i];
        s_delta = pl.sqrt(2*d_f*deltaT);
        
        for j in range(1,len(clims)):
            p_old_conf[j-1,i] = p_old[i]*pl.diff(stats.norm.cdf([clims[j],clims[j-1]],mu_delta,s_delta));
    return p_old_conf,p_new,t;


def predicted_proportions(c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT,
                          t_offset=0,use_fftw=True):
    # make c (the confidence levels) an array in case it is a scalar value
    c = pl.array(c,ndmin=1);
    n = len(c);
    # form an array consisting of the appropriate (upper) integration limits
    clims = pl.hstack(([INF_PROXY],c,[-INF_PROXY]));
    # compute process SD
    sigma_r = pl.sqrt(2*d_r*DELTA_T);
    sigma_f = pl.sqrt(2*d_f*DELTA_T);
    sigma = pl.sqrt(sigma_r**2+sigma_f**2);

    # compute the correlation for r given r+f
    rho = sigma_r/sigma;

    t = pl.linspace(DELTA_T,MAX_T,NR_TSTEPS); # this is the time axis
    to_idx = pl.argmin((t-t_offset)**2); # compute the index for t_offset
    bound = pl.exp(-tc_bound*pl.clip(t-t_offset,0,None)); # this is the collapsing bound
     
    mu = (mu_r+mu_f)*DELTA_T; # this is the average overall drift rate, with r = 'recall' and f = 'familiar'
    # compute the bounding limit of the space domain. This should include at least 99% of the probability mass when the particle is at the largest possible bound
    space_lim = max(bound)+3*sigma;
    delta_s = 2*space_lim/NR_SSTEPS;
    # finally, construct the space axis
    x = pl.linspace(-space_lim,space_lim,NR_SSTEPS);
    # compute the diffusion kernel
    kernel = stats.norm.pdf(x,mu,sigma)*delta_s;
    # ... and its Fourier transform. We'll use this to compute FD convolutions
    if(use_fftw):
        ft_kernel = fftw.fft(kernel);
    else:
        ft_kernel = pl.fft(kernel);
    tx = pl.zeros((len(t),len(x)));
    
    # Construct arrays to hold RT distributions
    p_old = pl.zeros(pl.shape(t));
    p_new = pl.zeros(pl.shape(t));
    p_rem_conf = pl.zeros((n+1,pl.size(t))); 
    p_know_conf = pl.zeros((n+1,pl.size(t)));
    
    
    ############################################
    ## take care of the first timestep #########
    ############################################
    tx[to_idx] = stats.norm.pdf(x,mu+z0,sigma)*delta_s;
    p_old[to_idx] = pl.sum(tx[to_idx][x>=bound[to_idx]]);
    p_new[to_idx] = pl.sum(tx[to_idx][x<=-bound[to_idx]]);
    # compute STD(r) for the current time
    s_r = sigma_r;
    s_f = sigma_f;
    # compute STD(r+f) for the current time
    s_comb = sigma;
    # compute E[r|(r+f)]
    #mu_r_cond = mu_r*t[0]+rho*s_r*(bound[0]-t[0]*(mu_r+mu_f))/s_comb;
    mu_r_cond = mu_r*t[to_idx]+(bound[to_idx]-t[to_idx]*(mu_r+mu_f)-z0)*rho**2;
    # compute STD[r|(r+f)]
    s_r_cond = s_r*pl.sqrt(1-rho**2);
    s_f_cond = s_f*pl.sqrt(1-(sigma_f/sigma)**2);
    
    # remove from consideration any particles that already hit the bound
    tx[to_idx]*=(abs(x)<bound[to_idx]);
    
    ############################################################################
    # compute the parameters of the bivariate distribution of particle locations
    # deltaT seconds after old/new decision
    
    mu_r_delta = mu_r_cond+mu_r*deltaT;
    mu_comb_delta = (mu_r+mu_f)*deltaT+bound[to_idx];
    s2_r_delta = s_r_cond**2+2*d_r*deltaT;
    s2_f_delta = s_f_cond**2+2*d_f*deltaT;
    s2_comb_delta = s2_r_delta+s2_f_delta;
    #s2_comb_delta = 2*deltaT*(d_r+d_f);
    #s2_comb_delta = s_r_cond**2+s_f_cond**2+2*deltaT*(d_r+d_f);
    cov_delta = s2_r_delta;
    mu_mvn = pl.array([mu_r_delta,mu_comb_delta]);
    sigma_mvn = pl.array([[s2_r_delta,cov_delta],[cov_delta,s2_comb_delta]]);
    ############################################################################
    for j in range(1,len(clims)):
        # Note that the clims appear in descending order, from highest to lowest value
        KLL = pl.array([-INF_PROXY,clims[j]]);     # lower limit for 'know' class
        KUL = pl.array([r_bound,clims[j-1]]);      # upper limit for 'know' class
        RLL = pl.array([r_bound,clims[j]]);        # lower limit for 'remember' class
        RUL = pl.array([INF_PROXY,clims[j-1]]);    # upper limit for 'remember' class
        p_know_conf[j-1,to_idx] = p_old[to_idx]*stats.mvn.mvnun(KLL,KUL,mu_mvn,sigma_mvn)[0];
        p_rem_conf[j-1,to_idx] = p_old[to_idx]*stats.mvn.mvnun(RLL,RUL,mu_mvn,sigma_mvn)[0];
        
    #######################################
    ## take care of subsequent timesteps ##
    #######################################
    
    for i in range(to_idx+1,len(t)):
        #tx[i] = convolve(tx[i-1],kernel,'same');
        # convolve the particle distribution from the previous timestep
        # with the diffusion kernel (using Fourier domain convolution)
        if(use_fftw):
            tx[i] = abs(pl.ifftshift(fftw.ifft(fftw.fft(tx[i-1])*ft_kernel)));
        else:
            tx[i] = abs(pl.ifftshift(pl.ifft(pl.fft(tx[i-1])*ft_kernel)));

        p_pos = tx[i][x>=bound[i]]; # probability of each particle position above the upper bound
        x_pos = x[x>=bound[i]];     # location of each particle position above the upper bound
        
        # compute the expected value of a particle that just exceeded the bound
        # during the last time interval
        comb_est = (pl.dot(p_pos,x_pos)+EPS)/(pl.sum(p_pos)+EPS); 

        p_old[i] = pl.sum(p_pos); # total probability that particle crosses upper bound
        p_new[i] = pl.sum(tx[i][x<=-bound[i]]); # probability that particle crosses lower bound
        
        # compute STD(r) for the current time
        s_r = pl.sqrt(2*d_r*t[i]);
        s_f = pl.sqrt(2*d_f*t[i]);
        # compute STD[r|(r+f)]
        s_r_cond = s_r*pl.sqrt(1-rho**2);
        s_f_cond = s_f*pl.sqrt(1-(sigma_f/sigma)**2);
        # compute E[r|(r+f)]
        mu_r_cond = mu_r*t[i]+(comb_est-t[i]*(mu_r+mu_f)-z0)*rho**2;
        # remove from consideration any particles that already hit the bound
        tx[i]*=(abs(x)<bound[i]);
        
        # (6/23/2015) New method for computing p_know, p_remember, and confidences
        # simultaneously using cumulative bivariate normal integral.
        # The idea is to:
        #   1. model the bivariate distribution of (r,r+f) particle locations 'deltaT'
        #   seconds after the old/new decision
        #   2. use the multivariate normal integral function (stats.mvn.mvnun) to compute
        #   the probability that the particle falls into any of the relevant regions
        #   defined by the constant "r" and "conf" bounds
        
        ########################################################################
        # compute the parameters of the bivariate distribution of particle
        # locations deltaT seconds after old/new decision
        mu_r_delta = mu_r_cond+mu_r*deltaT;
        mu_comb_delta = (mu_r+mu_f)*deltaT+comb_est;
        s2_r_delta = s_r_cond**2+2*d_r*deltaT;
        s2_f_delta = s_f_cond**2+2*d_f*deltaT;
        s2_comb_delta = s2_r_delta+s2_f_delta;
        #s2_comb_delta = 2*deltaT*(d_r+d_f);
        #s2_comb_delta = s_r_cond**2+s_f_cond**2+2*deltaT*(d_r+d_f);
        cov_delta = s2_r_delta;
        mu_mvn = pl.array([mu_r_delta,mu_comb_delta]);
        sigma_mvn = pl.array([[s2_r_delta,cov_delta],[cov_delta,s2_comb_delta]]);
        ########################################################################
        # Test Code:
        #if(t[i]>0.5):
            #slim = 3.0;
            #xaxis = linspace(-slim,slim,200);
            #xsup,ysup = meshgrid(xaxis,pl.flipud(xaxis));
            #supp = pl.vstack([xsup.flatten(),ysup.flatten()]).T;
            #z = stats.multivariate_normal.pdf(supp,mu_mvn,sigma_mvn);
            #figure(); imshow(reshape(z,shape(xsup)),cmap=cm.gray,extent=[-slim,slim,-slim,slim]);
            #vlines(r_bound,-slim,slim,colors='g');
            #hlines(c,-3,3,colors='r');
            #1/0
        ########################################################################
        for j in range(1,len(clims)):
            # remember that clims contains the bin edges in descending order
            KLL = pl.array([-INF_PROXY,clims[j]]);
            KUL = pl.array([r_bound,clims[j-1]]);
            RLL = pl.array([r_bound,clims[j]]);
            RUL = pl.array([INF_PROXY,clims[j-1]]);
            p_know_conf[j-1,i] = p_old[i]*stats.mvn.mvnun(KLL,KUL,mu_mvn,sigma_mvn)[0];
            p_rem_conf[j-1,i] = p_old[i]*stats.mvn.mvnun(RLL,RUL,mu_mvn,sigma_mvn)[0];
        
    # compute the marginal distributions for remember and know (i.e., across all confidence levels)
    p_remember = p_rem_conf.sum(0);
    p_know = p_know_conf.sum(0);
    return p_rem_conf,p_know_conf,p_new,t;

#def predicted_proportions_sim(mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0):
def predicted_proportions_sim(c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT,t_offset=0):
    # make c (the confidence levels) an array in case it is a scalar value
    c = pl.array(c,ndmin=1);
    n = len(c);
    # form an array consisting of the appropriate (upper) integration limits
    clims = pl.hstack(([INF_PROXY],c,[-INF_PROXY]));
    # compute process SD
    sigma_r = pl.sqrt(2*d_r*DELTA_T);
    sigma_f = pl.sqrt(2*d_f*DELTA_T);
    sigma = pl.sqrt(sigma_r**2+sigma_f**2);
    
    t = pl.linspace(DELTA_T,MAX_T,NR_TSTEPS);
    to_idx = pl.argmin((t-t_offset)**2); # compute the index for t_offset
    bound = pl.exp(-tc_bound*pl.clip(t-t_offset,0,None)); # this is the collapsing bound

    # Now simulate NR_SAMPLES trials
    # 1. Generate a random position change for each time interval
    #   these position changes should be drawn from a normal distribution with mean
    #   mu and standard deviation sigma

    delta_r = stats.norm.rvs(mu_r*DELTA_T,sigma_r,size=(NR_SAMPLES,NR_TSTEPS));
    delta_f = stats.norm.rvs(mu_f*DELTA_T,sigma_f,size=(NR_SAMPLES,NR_TSTEPS));
    delta_r[:to_idx] = 0;
    delta_f[:to_idx] = 0;
    delta_pos = delta_r+delta_f;

    # 2. Use cumsum to compute absolute positions from delta_pos
    positions = pl.cumsum(delta_pos,1)+z0;
    r_positions = pl.cumsum(delta_r,1);
    # 3. Now loop through each sample trial to compute decisions and resp times
    decisions = pl.zeros(NR_SAMPLES);
    resp_times = pl.zeros(NR_SAMPLES);
    final_pos = pl.zeros((NR_SAMPLES,2))
    
    for i, pos in enumerate(positions):
        # Find the index where the position first crosses a boundary (i.e., 1 or -1)
        cross_indices = pl.find(abs(pos)>=bound);
        if len(cross_indices):
            cross_idx = cross_indices[0]; # take the first index
        else: # i.e., if no crossing was found
            cross_idx = NR_TSTEPS-1; # set it to the final index
        # 4. Now we can use this index to determine both the decision and the response time
        decisions[i] = pos[cross_idx]>0;
        resp_times[i] = t[cross_idx]-0.5*DELTA_T; #i.e., the midpoint of the crossing interval
        final_pos[i] = [pos[cross_idx],r_positions[i,cross_idx]];
        
    # compute the distribution of particle positions deltaT seconds after the
    # old/new decision and choose a random position from the resulting
    # distribution to add to the position at decision
    # the parameters below are for the post old/new decision interval
    
    sigma_r_deltaT = pl.sqrt(2*d_r*deltaT);
    sigma_f_deltaT = pl.sqrt(2*d_f*deltaT);
    r_deltaTs = stats.norm.rvs(mu_r*deltaT,sigma_r_deltaT,size=NR_SAMPLES);
    pos_deltaTs = stats.norm.rvs(mu_f*deltaT,sigma_f_deltaT,size=NR_SAMPLES)+r_deltaTs;
    final_pos[:,0]+=pos_deltaTs; final_pos[:,1]+=r_deltaTs;
    
    # now make the remember decision on the basis of the final position in the
    # r dimension
    remembers = pl.logical_and(decisions,final_pos[:,1]>r_bound);
    
    p_remember  = [];
    p_know = [];
    for j in range(1,len(clims)):
        # remember that the confidence levels are arranged in decreasing order
        valid_confs = pl.logical_and(final_pos[:,0]>clims[j],final_pos[:,0]<=clims[j-1]);
        conf_rems = pl.logical_and(valid_confs,remembers);
        conf_knows = pl.logical_and(valid_confs,pl.logical_and(pl.logical_not(remembers),decisions));
        rem_RTs = resp_times[conf_rems];
        know_RTs = resp_times[conf_knows];
        params_rem = stats.gamma.fit(rem_RTs,floc=0);
        params_know = stats.gamma.fit(know_RTs,floc=0);
        p_remember_conf = stats.gamma.pdf(t,*params_rem)*DELTA_T*len(rem_RTs)/float(NR_SAMPLES);
        p_know_conf = stats.gamma.pdf(t,*params_know)*DELTA_T*len(know_RTs)/float(NR_SAMPLES);
        
        p_remember.append(p_remember_conf);
        p_know.append(p_know_conf);
        
    p_remember = pl.array(p_remember);
    p_know = pl.array(p_know);
        
    new_RTs = resp_times[pl.logical_not(decisions)];
    params_new = stats.gamma.fit(new_RTs,floc=0);
    p_new = stats.gamma.pdf(t,*params_new)*DELTA_T*len(new_RTs)/float(NR_SAMPLES);
    return p_remember,p_know,p_new,t;

def predicted_proportions_NC(mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT,use_fftw=True):
    # compute process SD
    sigma_r = pl.sqrt(2*d_r*DELTA_T);
    sigma_f = pl.sqrt(2*d_f*DELTA_T);
    sigma = pl.sqrt(sigma_r**2+sigma_f**2);

    # compute the correlation for r given r+f
    rho = sigma_r/sigma;
    rhoF = sigma_f/sigma;

    t = pl.linspace(DELTA_T,MAX_T,NR_TSTEPS); # this is the time axis
    bound = pl.exp(-tc_bound*t); # this is the collapsing bound
     
    mu = (mu_r+mu_f)*DELTA_T; # this is the average overall drift rate, with r = 'recall' and f = 'familiar'
    # compute the bounding limit of the space domain. This should include at least 99% of the probability mass when the particle is at the largest possible bound
    space_lim = max(bound)+3*sigma;
    delta_s = 2*space_lim/NR_SSTEPS;
    # finally, construct the space axis
    x = pl.linspace(-space_lim,space_lim,NR_SSTEPS);
    # compute the diffusion kernel
    kernel = stats.norm.pdf(x,mu,sigma)*delta_s;
    # ... and its Fourier transform. We'll use this to compute FD convolutions
    if(use_fftw):
        ft_kernel = fftw.fft(kernel);
    else:
        ft_kernel = pl.fft(kernel);
    tx = pl.zeros((len(t),len(x)));
    #p_know = pl.zeros(shape(t));
    p_remember = pl.zeros(pl.shape(t));
    p_old = pl.zeros(pl.shape(t));
    p_new = pl.zeros(pl.shape(t));

    # take care of the first timestep
    tx[0] = stats.norm.pdf(x,mu+z0,sigma)*delta_s;
    p_old[0] = pl.sum(tx[0][x>=bound[0]]);
    p_new[0] = pl.sum(tx[0][x<=-bound[0]]);
    # compute STD(r) for the current time
    s_r = sigma_r;
    s_f = sigma_f;
    # compute STD(r+f) for the current time
    s_comb = sigma;
    # compute E[r|(r+f)]
    mu_r_cond = mu_r*t[0]+rho*s_r*(bound[0]-t[0]*(mu_r+mu_f))/s_comb;
    mu_f_cond = mu_f*t[0]+rhoF*s_f*(bound[0]-t[0]*(mu_r+mu_f))/s_comb;
    # compute STD[r|(r+f)]
    s_r_cond = s_r*pl.sqrt(1-rho**2);
    s_f_cond = s_f*pl.sqrt(1-rhoF**2);
    
    mu_r_delta = mu_r_cond+mu_r*deltaT;
    s_r_delta = pl.sqrt(s_r_cond**2+2*d_r*deltaT);
    p_remember[0] = p_old[0]*stats.norm.sf(r_bound,mu_r_delta,s_r_delta);
    # remove from consideration any particles that already hit the bound
    tx[0]*=(abs(x)<bound[0]);
    for i in range(1,len(t)):
        # convolve the particle distribution from the previous timestep
        # with the diffusion kernel (using Fourier domain convolution)
        if(use_fftw):
            tx[i] = abs(pl.ifftshift(fftw.ifft(fftw.fft(tx[i-1])*ft_kernel)));
        else:
            tx[i] = abs(pl.ifftshift(pl.ifft(pl.fft(tx[i-1])*ft_kernel)));

        p_pos = tx[i][x>=bound[i]]; # probability of each particle position above the upper bound
        x_pos = x[x>=bound[i]];     # location of each particle position above the upper bound

        p_old[i] = pl.sum(p_pos); # total probability that particle crosses upper bound
        p_new[i] = pl.sum(tx[i][x<=-bound[i]]); # probability that particle crosses lower bound
        
        # compute STD(r) for the current time
        s_r = pl.sqrt(2*d_r*t[i]);
        s_f = pl.sqrt(2*d_f*t[i]);
        # compute STD[r|(r+f)]
        s_r_cond = s_r*pl.sqrt(1-rho**2);
        s_f_cond = s_f*pl.sqrt(1-rhoF**2);
        # compute E[r|(r+f)]
        mu_r_cond = mu_r*t[i]+(x_pos-t[i]*(mu_r+mu_f)-z0)*rho**2;
        mu_f_cond = mu_f*t[i]+(x_pos-t[i]*(mu_r+mu_f)-z0)*rhoF**2;
        # remove from consideration any particles that already hit the bound
        tx[i]*=(abs(x)<bound[i]);
        
        # old method for computing p_remember
        # p_remember[i] = sum(p_pos*stats.norm.sf(r_bound,mu_r_cond,s_r_cond));
        # new method for computin p_remember
        mu_r_delta = mu_r_cond+mu_r*deltaT;
        s_r_delta = pl.sqrt(s_r_cond**2+2*d_r*deltaT);
        p_remember[i] = pl.sum(p_pos*stats.norm.sf(r_bound,mu_r_delta,s_r_delta));

    #p_remember = p_old-p_know;
    p_know = p_old-p_remember;
    
    # TO DO:
    ## Added by Melchi on 6/15/2015 ##
    # The conditional distribution of position for the 'recall' component of the
    # particle (conditioned on time of new/old decision and/or value of r+f) has
    # already been computed. It's a normal distribution with mean mu_r_cond and
    # standard deviation s_r_cond. What remains to be done is to compute the
    # distribtuion of positions for the recall component after deltaT additional
    # seconds have elapsed
    # 1. The resulting distribution should have mean = mu_r_cond + (mu_r*deltaT)
    # 2. The resulting distribution should have SD = sqrt(s_r_cond^2+2*d_r*deltaT)
    ######################################################################################
    # determine the proportion of new, remember and know responses by confidence
    # determine the time points corresponding to quartiles within the overall distribution of remember and know responses

    
    return p_remember,p_know,p_new,t;

################################################################################
## Plotting functions
################################################################################

def emp_v_prediction(model_params,data=DATA):
    conf,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,mu_f0,mu_r0,deltaT,t_offset = model_params;
    #mu_f0 = mu_r0 =0;
    c = [conf,0];
    params_est_old = [c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT,t_offset];
    params_est_new = [c,mu_r0,mu_f0,d_r,d_f,tc_bound,r_bound,z0,deltaT,t_offset];
    
    nr_conf = len(c);
    # adjust the number of confidence levels in the data to match number in model
    hr_conf = pl.clip(data.rem_hit.conf,0,nr_conf);
    hk_conf = pl.clip(data.know_hit.conf,0,nr_conf);
    fr_conf = pl.clip(data.rem_fa.conf,0,nr_conf);
    fk_conf = pl.clip(data.know_fa.conf,0,nr_conf);
    
    # flip the arrays below so that the confidence levels appear in descending order
    hr_rts = [data.rem_hit.rt[hr_conf==i] for i in reversed(pl.unique(hr_conf))];
    hk_rts = [data.know_hit.rt[hk_conf==i] for i in reversed(pl.unique(hk_conf))];
    fr_rts = [data.rem_fa.rt[fr_conf==i] for i in reversed(pl.unique(fr_conf))];
    fk_rts = [data.know_fa.rt[fk_conf==i] for i in reversed(pl.unique(fk_conf))];
    
    n_old = len(hr_conf)+len(hk_conf)+len(data.miss.rt);
    n_new = len(fr_conf)+len(fk_conf)+len(data.CR.rt);
    
    # compute predicted proportions
    pp_old = predicted_proportions(*params_est_old)[:-1];
    pp_new = predicted_proportions(*params_est_new)[:-1];
    
    old_data = [hr_rts,hk_rts,data.miss.rt];
    new_data = [fr_rts,fk_rts,data.CR.rt];
    
    return old_data,new_data,pp_old,pp_new,n_old,n_new
    
# plotting the data
def plot_evp_pair(p_dist,e_dist,e_total,col='g'):
    """
    makes a plot comparing the model-generated RT distribution p_dist to an
    the distribution of a sample of empirical (observed) reaction times e_dist.
    e_total represents the size of the sample.
    """
    # plot the histogram for observed data
    t = pl.linspace(DELTA_T,MAX_T,NR_TSTEPS); # this is the time axis
    # compute the prior distribution for the response category
    p_cat = len(e_dist)*1.0/e_total;
    ## compute density histogram
    hd,edges = pl.histogram(e_dist,bins=40,range=[0,10],density=True);
    hist_density = pl.hstack([[0],hd]);
    pl.plot(edges,hist_density*p_cat,color=col,lw=2,ls='--',drawstyle='steps');
    # note: the division by DELTA_T below is to make sure that you are plotting
    # probability densities (rather than probability masses)
    curve = pl.plot(t,p_dist/DELTA_T,col,lw=2)
    pl.axis([0,t.max(),None,None])
    pl.show();
    return curve;
    
def plot_comparison(model_params,nr_conf_bounds=2):
    """
    makes a set of two plots comparing the predictions of a model parameterized
    by model_params to the observed reaction times and confidence judgments.
    The left panel represents the distribution of judgments and RTs for old
    (target) words, while the right panel represents the distribution for new
    (lure) words.
    """
    nr_conf = nr_conf_bounds+1;
    old_data,new_data,pp_old,pp_new,n_old,n_new = emp_v_prediction(model_params);
    #nr_conf=len(pp_old[0]);
    colors = ['k','r','b','g','c']
    kcolors = ['#ff8080','#8080ff','#80c080'];
    # plot comparison for 'old' words
    pl.figure(figsize=(12,5));
    pl.subplot(1,2,1);
    curves = [];
    c_idx = 0;
    kc_idx = 0;
    # 1. plot misses
    curve, = plot_evp_pair(pp_old[-1],old_data[-1],n_old,colors[c_idx]);
    curves.append(curve);
    c_idx+=1;
    # 2. plot hits
    for conf in range(nr_conf):
        curve, = plot_evp_pair(pp_old[0][conf],old_data[0][conf],n_old,colors[c_idx]);
        curves.append(curve);
        c_idx+=1;
        # plot "know" hits
        curve, = plot_evp_pair(pp_old[1][conf],old_data[1][conf],n_old,kcolors[kc_idx]);
        curves.append(curve);
        kc_idx+=1;
    pl.axis([0,6,0,0.7]);
    simpleaxis(pl.gca());
    pl.title('RT Distributions for Target Words');
    pl.xlabel('Reaction time (sec.)');
    pl.ylabel('p(RT,judgment)');

    pl.legend(curves,['new','rem, conf=2','know, conf=2','rem, conf=1',\
              'know, conf=1','rem, conf=0','know, conf=0'],loc='best',\
              prop={'size':'small'})

    
    # plot comparison for 'new' words
    pl.subplot(1,2,2);
    curves = [];
    c_idx = 0;
    kc_idx = 0;
    # 1. plot misses
    curve, = plot_evp_pair(pp_new[-1],new_data[-1],n_new,colors[c_idx]);
    curves.append(curve);
    c_idx+=1;
    # 2. plot hits
    for conf in range(nr_conf):
        curve, = plot_evp_pair(pp_new[0][conf],new_data[0][conf],n_new,colors[c_idx]);
        curves.append(curve);
        c_idx+=1;
        # plot "know" hits
        curve, = plot_evp_pair(pp_new[1][conf],new_data[1][conf],n_new,kcolors[kc_idx]);
        curves.append(curve);
        kc_idx+=1;
    pl.axis([0,6,0,0.7]);
    simpleaxis(pl.gca());
    pl.title('RT Distributions for Lure Words');
    pl.xlabel('Reaction time (sec.)');
