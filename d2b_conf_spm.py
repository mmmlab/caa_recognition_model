import shelve
from pylab import *
import pylab as pl
import numpy
from scipy import stats
from scipy import optimize
import pyfftw
import fftw_test as fftw
from multinomial_funcs import multinom_loglike,chi_square_gof
from scipy.stats import gaussian_kde

data_path = 'neha/data/'; # this is the base path for the data files

## Start by reading in the data.
# the reason to do this first is that, in order to be efficient,
# we don't want to represent any more of the time axis than we have to.

# Read in new Vincentized RT data
db = shelve.open(data_path+'neha_data.dat','r');
rem_hit = db['rem_hit'];
know_hit = db['know_hit'];
rem_fa = db['rem_fa'];
know_fa = db['know_fa'];
CR = db['CR'];
miss = db['miss'];
db.close();

 
INF_PROXY   = 10; # a value used to provide very large but finite bounds for mvn integration
EPS         = 1e-10 # a very small value (used for numerical stability)
NR_THREADS  = 1;    # this is for multithreaded fft
DELTA_T     = 0.02;# 0.05;  # size of discrete time increment (sec.)
MAX_T       = 8.0;#24; #ceil(percentile(all_RT,99.5))
NR_TSTEPS   = MAX_T/DELTA_T;
NR_SSTEPS   = 8192; #8192#4096#2048;
NR_SAMPLES  = 10000; # number of trials to use for MC likelihood computation
n = 2; # number of confidence critetion
QUANT = array([0,0.25,0.50,0.75]);
QUANT_DIFF = 0.25;
NR_QUANTILES = 4;
R = 0.1; D = 0.05; L = 0.1; Z = 0.0;


fftw.fftw_setup(zeros(NR_SSTEPS),NR_THREADS);

params_est = [0.99,0.11,0.06,0.66,-0.11,1.27]; # new values from global search
param_bounds = [(0.0,1.0),(-2.0,2.0),(EPS,2.0),(0.05,1.0),(-1.0,1.0),(EPS,2.0)];
# version with free parameters for "new" words
params_est_all = [0.91,0.11,0.09,-0.09,0.09,0.52,-0.07,0.52]; # new values from global search
param_bounds_all = [(0.0,1.0),(-2.0,2.0),(EPS,2.0),(-2.0,2.0),(EPS,2.0),(0.05,1.0),(-1.0,1.0),(EPS,2.0)];

# version with three confidence levels
params_est_all_c = [0.589,0.0026,0.119,0.156,-0.197,0.128,0.370,0.0351,0.701,0]; # values without temporal offset
# values including temporal offset below
#params_est_all_c = [0.9367,0.0,0.3047,0.4217,-0.2659,0.3585,0.0533,-0.1185,0.499 ,0.4965];
params_est_all_c = [ 0.924,0.0,0.312,0.411,-0.255,0.346,0.053,-0.123,0.496,0.53]; # fitted w/ 10 quantiles, chisq = 655 
param_bounds_all_c = [(0.0,1.0),(0.0,1.0),(-2.0,2.0),(EPS,2.0),(-2.0,2.0),(EPS,2.0),(0.05,1.0),(-1.0,1.0),(EPS,2.0),(0,0.3)];

# version with single diffusion parameter and lowest confidence bound fixed at zero
params_est_all_rp = [0.9169,0.319,0.3888,-0.265,0.0505,-0.1198,0.4968,0.5799]; # fitted w/ 10 quantiles, chisq = 606


def find_ml_params_all(quantiles=4,nr_conf_bounds=2):
    return optimize.differential_evolution(compute_gof_all,param_bounds_all_c)

def find_ml_params_all_lm(quantiles=4,nr_conf_bounds=2):
    # computes mle of params using a local (fast) optimization algorithm
    return optimize.fmin(compute_gof_all,params_est_all_c)

#def find_ml_params_all_lm_rp(quantiles=4):
#    # computes mle of params using a local (fast) optimization algorithm
#    def obj_func(model_params):
#        c,mu_old,d,mu_new,tc_bound,z0,deltaT,t_offset = model_params;
#        params_est_old = [[c,0],mu_old,d,tc_bound,z0,deltaT,t_offset];
#        params_est_new = [[c,0],mu_new,d,tc_bound,z0,deltaT,t_offset];
#        old_data = [hstack([rem_hit[:,0],know_hit[:,0]]),miss[:,0],hstack([rem_hit[:,1],know_hit[:,1]])];
#        new_data = [hstack([rem_fa[:,0],know_fa[:,0]]),CR[:,0],hstack([rem_fa[:,1],know_fa[:,1]])];
#        res = compute_model_gof(params_est_old,*old_data,nr_quantiles=quantiles)+ \
#        compute_model_gof(params_est_new,*new_data,nr_quantiles=quantiles);
#        return res;
#    return optimize.fmin(obj_func,params_est_all_rp)

def compute_gof_all(model_params,quantiles=4):
        c,mu_old,d,mu_new,tc_bound,z0,deltaT,t_offset = model_params;
        params_est_old = [[c,0],mu_old,d,tc_bound,z0,deltaT,t_offset];
        params_est_new = [[c,0],mu_new,d,tc_bound,z0,deltaT,t_offset];
        old_data = [hstack([rem_hit[:,0],know_hit[:,0]]),miss[:,0],hstack([rem_hit[:,1],know_hit[:,1]])];
        new_data = [hstack([rem_fa[:,0],know_fa[:,0]]),CR[:,0],hstack([rem_fa[:,1],know_fa[:,1]])];
        res = compute_model_gof(params_est_old,*old_data,nr_quantiles=quantiles)+ \
        compute_model_gof(params_est_new,*new_data,nr_quantiles=quantiles);
        return res;

def compute_model_gof(model_params,old_RTs,new_RTs,old_conf,nr_quantiles=4):
    # computes the chi square fit of the model to the data
    # compute N, the total number of trials
    N = len(old_RTs)+len(new_RTs);
    # compute x, the observed frequency for each category
    old_quantiles,new_quantiles,p_o,p_n = compute_model_quantiles(model_params,nr_quantiles);
    # determine the number of confidence levels being used in the model
    nr_conf_levels = len(old_quantiles);
    # adjust the number of confidence levels in the data to match
    old_conf = clip(old_conf,0,nr_conf_levels-1);
    ## compute the number of RTs falling into each quantile bin
    old_freqs = array([-diff([sum(old_RTs[old_conf==i]>q) for q in old_quantiles[i]]+[0]) for i in range(nr_conf_levels)]);
    ## I think this is where the problem was. The confidence levels in the
    ## model are in descending order, while these (for the empircal data) are
    ## in ascending order. I'll flip them here
    old_freqs = flipud(old_freqs);
    new_freqs = -diff([sum(new_RTs>q) for q in new_quantiles]+[0]);
    x = hstack([old_freqs.flatten(),new_freqs]);
    
    # compute p, the probability associated with each category in the model
    p_o = p_o[:,newaxis]*ones((nr_conf_levels,nr_quantiles))/float(nr_quantiles);
    p_new = p_n*ones(nr_quantiles)/float(nr_quantiles);
    p = hstack([p_o.flatten(),p_new]);
    return chi_square_gof(x,N,p)

def compute_model_quantiles(params,nr_quantiles=4):
    # This function is set up to deal with multiple confidence levels
    quantile_increment = 1.0/nr_quantiles;
    quantiles = arange(0,1,quantile_increment);
    # compute marginal distributions
    p_old,p_new,t = predicted_proportions(*params);
    # compute marginal category proportions (per confidence level)
    old_total = p_old.sum(-1)+EPS;
    new_total = p_new.sum()+EPS;
    # compute integrals of marginal distributions
    P_o = cumsum(p_old,-1)/old_total[:,newaxis];
    P_n = cumsum(p_new)/new_total;
    
    # compute RT quantiles (by confidence level)
    old_quantiles = array([t[argmax(P_o>q,-1)] for q in quantiles]).T;
    new_quantiles = array([t[argmax(P_n>q)] for q in quantiles]);
    old_quantiles[:,0] = 0;
    new_quantiles[0] = 0;
    # return quantile locations and marginal p(new)
    return old_quantiles,new_quantiles,sum(p_old,1),sum(p_new);


# (10/06/2016) This is the new version of the function, which implements the
# collapsing bound model, but as (effectively) a single-process model
# One major advantage is that we can eliminate at least 3 parameters:
# mu_r, d_r, and r_bound

def predicted_proportions(c,mu_f,d_f,tc_bound,z0,deltaT,t_offset=0,use_fftw=True):
    # make c (the confidence levels) an array in case it is a scalar value
    c = array(c,ndmin=1);
    n = len(c);
    # form an array consisting of the appropriate (upper) integration limits
    # note that the limits appear in descending order, with high confidence appearing
    # first and low confidence appearing last within the array
    clims = hstack(([INF_PROXY],c,[-INF_PROXY]));
    # compute process SD
    sigma_f = sqrt(2*d_f*DELTA_T);
    sigma = sigma_f;

    t = linspace(DELTA_T,MAX_T,NR_TSTEPS); # this is the time axis
    to_idx = argmin((t-t_offset)**2); # compute the index for t_offset
    to_idx = 0;
    bound = exp(-tc_bound*clip(t-t_offset,0,None)); # this is the collapsing bound
    bound = exp(-tc_bound*t); # this is the collapsing bound
     
    mu = mu_f*DELTA_T; # this is the expected drift over time interval DELTA_T
    # compute the bounding limit of the space domain. This should include at
    # least 99% of the probability mass when the particle is at the largest possible bound
    space_lim = max(bound)+3*sigma;
    delta_s = 2*space_lim/NR_SSTEPS;
    # finally, construct the space axis
    x = linspace(-space_lim,space_lim,NR_SSTEPS);
    # compute the diffusion kernel
    kernel = stats.norm.pdf(x,mu,sigma)*delta_s;
    # ... and its Fourier transform. We'll use this to compute FD convolutions
    if(use_fftw):
        ft_kernel = fftw.fft(kernel);
    else:
        ft_kernel = fft(kernel);
    tx = zeros((len(t),len(x)));
    
    
    
    # Construct arrays to hold RT distributions
    p_old = zeros(shape(t));
    p_new = zeros(shape(t));
    p_old_conf = zeros((n+1,size(t)));  
    
    ############################################
    ## take care of the first timestep #########
    ############################################
    tx[to_idx] = stats.norm.pdf(x,mu+z0,sigma)*delta_s;
    p_old[to_idx] = sum(tx[0][x>=bound[to_idx]]);
    p_new[to_idx] = sum(tx[to_idx][x<=-bound[to_idx]]);
    
    # remove from consideration any particles that already hit the bound
    tx[to_idx]*=(abs(x)<bound[to_idx]);
    
    ############################################################################
    # compute the parameters for the distribution of particle locations
    # deltaT seconds after old/new decision
    mu_delta = mu_f*deltaT+bound[to_idx];
    s_delta = sqrt(2*d_f*deltaT);
    
    # compute the probability that the particle falls within the region for
    # each category. Note that now the particle's trajectory is 1D, so that
    # the only thing that determines the region is the bounding interval,
    # specified in terms of f
    for j in range(1,len(clims)):
        # based on this code, the confidence probabilities should also be in
        # reverse (decreasing order)
        p_old_conf[j-1,to_idx] = p_old[to_idx]*diff(stats.norm.cdf([clims[j],clims[j-1]],\
                                                    mu_delta,s_delta));
        
    #######################################
    ## take care of subsequent timesteps ##
    #######################################
    
    for i in range(to_idx+1,len(t)):
        # convolve the particle distribution from the previous timestep
        # with the diffusion kernel (using Fourier domain convolution)
        if(use_fftw):
            tx[i] = abs(ifftshift(fftw.ifft(fftw.fft(tx[i-1])*ft_kernel)));
        else:
            tx[i] = abs(ifftshift(ifft(fft(tx[i-1])*ft_kernel)));

        p_pos = tx[i][x>=bound[i]]; # probability of each particle position above the upper bound
        x_pos = x[x>=bound[i]];     # location of each particle position above the upper bound
        
        # compute the expected value of a particle that just exceeded the bound
        # during the last time interval
        comb_est = (dot(p_pos,x_pos)+EPS)/(sum(p_pos)+EPS); 

        p_old[i] = sum(p_pos); # total probability that particle crosses upper bound
        p_new[i] = sum(tx[i][x<=-bound[i]]); # probability that particle crosses lower bound
        
        # remove from consideration any particles that already hit the bound
        tx[i]*=(abs(x)<bound[i]);
        
        # compute the parameters for the distribution of particle locations
        # deltaT seconds after old/new decision
        mu_delta = mu_f*deltaT+bound[i];
        s_delta = sqrt(2*d_f*deltaT);
        
        for j in range(1,len(clims)):
            p_old_conf[j-1,i] = p_old[i]*diff(stats.norm.cdf([clims[j],clims[j-1]],mu_delta,s_delta));
    return p_old_conf,p_new,t;

def predicted_proportions_sim(c,mu_f,d_f,tc_bound,z0,deltaT,t_offset=0):
    # make c (the confidence levels) an array in case it is a scalar value
    c = array(c,ndmin=1);
    n = len(c);
    
    # form an array consisting of the appropriate (upper) integration limits
    clims = hstack(([INF_PROXY],c,[-INF_PROXY]));
    # compute process SD
    sigma_f = sqrt(2*d_f*DELTA_T);
    sigma = sigma_f;

    t = linspace(DELTA_T,MAX_T,NR_TSTEPS); # this is the time axis
    bound = exp(-tc_bound*t); # this is the collapsing bound

    # Now simulate NR_SAMPLES trials
    # 1. Generate a random position change for each time interval
    #   these position changes should be drawn from a normal distribution with mean
    #   mu and standard deviation sigma
    delta_pos = stats.norm.rvs(mu_f*DELTA_T,sigma,size=(NR_SAMPLES,NR_TSTEPS));

    # 2. Use cumsum to compute absolute positions from delta_pos
    positions = pl.cumsum(delta_pos,1)+z0;
    # 3. Now loop through each sample trial to compute decisions and resp times
    decisions = zeros(NR_SAMPLES);
    resp_times = zeros(NR_SAMPLES);
    final_pos = zeros(NR_SAMPLES)
    
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
        final_pos[i] = pos[cross_idx];
        
    # compute the distribution of particle positions deltaT seconds after the
    # old/new decision and choose a random position from the resulting
    # distribution to add to the position at decision
    # the parameters below are for the post old/new decision interval
    
    sigma_deltaT = sqrt(2*d_f*deltaT);
    pos_deltaTs = stats.norm.rvs(mu_f*deltaT,sigma_deltaT,size=NR_SAMPLES);
    final_pos+=pos_deltaTs;
    
    p_old = [];
    for j in range(1,len(clims)):
        # remember that the confidence levels are arranged in decreasing order
        valid_confs = logical_and(final_pos>clims[j],final_pos<=clims[j-1]);
        conf_old = logical_and(valid_confs,decisions);
        old_RTs = resp_times[conf_old];
        params_old = stats.gamma.fit(old_RTs,floc=0);
        p_old_conf = stats.gamma.pdf(t,*params_old)*DELTA_T*len(old_RTs)/float(NR_SAMPLES);
        
        p_old.append(p_old_conf);
        
    p_old = array(p_old);
        
    new_RTs = resp_times[logical_not(decisions)];
    params_new = stats.gamma.fit(new_RTs,floc=0);
    p_new = stats.gamma.pdf(t,*params_new)*DELTA_T*len(new_RTs)/float(NR_SAMPLES);
    return p_old,p_new,t;


##########################################################################################
## Plotting functions
##########################################################################################

def emp_v_prediction(model_params,nr_conf=2):
    conf,mu_old,d_old,mu_new,tc_bound,z0,deltaT,t_offset = model_params;
    c = [conf,0]
    params_est_old = [c,mu_old,d_old,tc_bound,z0,deltaT,t_offset];
    params_est_new = [c,mu_new,d_old,tc_bound,z0,deltaT,t_offset];
    
    hits = vstack([rem_hit,know_hit]);
    FAs = vstack([rem_fa,know_fa]);

    nr_conf = len(c);
    # adjust the number of confidence levels in the data to match number in model
    hconf = clip(hits[:,1],0,nr_conf);
    fconf = clip(FAs[:,1],0,nr_conf);
    
    # flip the arrays below so that the confidence levels appear in descending order
    h_rts = [hits[hconf==i,0] for i in reversed(unique(hconf))];
    f_rts = [FAs[fconf==i,0] for i in reversed(unique(fconf))];

    old_data = [h_rts,miss[:,0]];
    new_data = [f_rts,CR[:,0]];
    
    n_old = len(hits)+len(miss);
    n_new = len(FAs)+len(CR);
    
    # compute predicted proportions
    pp_old = predicted_proportions(*params_est_old);
    pp_new = predicted_proportions(*params_est_new);
    
    return old_data,new_data,pp_old,pp_new,n_old,n_new
    
# plotting the data
def plot_evp_pair(p_dist,e_dist,e_total,col='g'):
    
    # plot the histogram for observed data
    t = linspace(DELTA_T,MAX_T,NR_TSTEPS); # this is the time axis
    
    # compute the prior distribution for the response category
    p_cat = len(e_dist)*1.0/e_total;
    
    ## compute density histogram
    hd,edges = histogram(e_dist,bins=40,range=[0,10],density=True);
    hist_density = hstack([[0],hd]);
    step(edges,hist_density*p_cat,color=col,lw=2);
    # plot the KDE for observed data
    #kde = stats.kde.gaussian_kde(e_dist);
    #plot(t,kde(t)*p_cat,col+'--',lw=2)
    ## generate the raster plot of raw RTs
    #rmin = 0;
    #rmax = max(kde(t))*0.1;
    #vlines(e_dist,rmin,rmax,color='k',alpha=0.15);
    # plot the PDF for predicted data
    # note: the division by DELTA_T below is to make sure that you are plotting
    # probability densities (rather than probability masses)
    plot(t,p_dist/DELTA_T,col,lw=2)
    axis([0,t.max(),None,None])
    show();
    
def plot_comparison(model_params,nr_conf_bounds=2):
    nr_conf = nr_conf_bounds+1;
    old_data,new_data,pp_old,pp_new,n_old,n_new = emp_v_prediction(model_params,nr_conf_bounds);
    #nr_conf=len(pp_old[0]);
    colors = ['k','r','b','g','c']
    # plot comparison for 'old' words
    figure(); title('RT Distributions for Old Words');
    c_idx = 0;
    # 1. plot misses
    plot_evp_pair(pp_old[1],old_data[1],n_old,colors[c_idx]);
    c_idx+=1;
    # 2. plot hits
    for conf in range(nr_conf):
        plot_evp_pair(pp_old[0][conf],old_data[0][conf],n_old,colors[c_idx]);
        c_idx+=1;
    axis([0,6,0,0.7]);
    # plot comparison for 'new' words
    figure(); title('RT Distributions for New Words');
    c_idx = 0;
    # 1. plot misses
    plot_evp_pair(pp_new[1],new_data[1],n_new,colors[c_idx]);
    c_idx+=1;
    # 2. plot hits
    for conf in range(nr_conf):
        plot_evp_pair(pp_new[0][conf],new_data[0][conf],n_new,colors[c_idx]);
        c_idx+=1;
    axis([0,6,0,0.7]);