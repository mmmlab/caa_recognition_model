from pylab import *
import numpy as np
import pylab as pl
from scipy import stats
import scipy.optimize as opt
import fftw_test as fftw
from multinomial_funcs import multinom_loglike,chi_square_gof

################################################################################
observed = np.loadtxt('neha/myData.txt'); # load the observed data
remember_hit = np.loadtxt('neha/remRT_hit.txt'); # load remember RTs for hits
know_hit = np.loadtxt('neha/knowRT_hit.txt'); # load know RTs for hits
miss = np.loadtxt('neha/miss.txt'); # load miss RTs
remember_fa = np.loadtxt('neha/remRT_fa.txt'); # load remember RTs for false alarms
know_fa = np.loadtxt('neha/knowRT_fa.txt');  # load know RTs for false alarms
CR = np.loadtxt('neha/CR.txt'); # load CR RTs

remH_RT,remH_conf = np.split(remember_hit,2,axis=1);
knowH_RT,knowH_conf = np.split(know_hit,2,axis=1);
remFA_RT,remFA_conf = np.split(remember_fa,2,axis=1);
knowFA_RT,knowFA_conf = np.split(know_fa,2,axis=1);
miss_RT,junk = np.split(miss,2,axis=1);
CR_RT,junk = np.split(CR,2,axis=1);

all_RT = vstack([remH_RT,remFA_RT,knowH_RT,knowFA_RT,miss_RT,CR_RT]);
################################################################################

# Out[34]:

#     array([[  918.,   526.,     0.],
#            [  321.,   353.,     0.],
#            [   64.,   125.,     0.],
#            [    0.,     0.,  1661.],
#            [  196.,   164.,     0.],
#            [  184.,   283.,     0.],
#            [   72.,   132.,     0.],
#            [    0.,     0.,  2937.]])

# In[35]:

# Constants
NR_THREADS  = 1;    # this is for multithreaded fft
DELTA_T     = 0.05;  # size of discrete time increment (sec.)
MAX_T       = 12.0; #(i.e., 20 sec)
NR_TSTEPS   = MAX_T/DELTA_T;
NR_SSTEPS   = 8192#4096#2048;
NR_SAMPLES  = 10000; # number of trials to use for MC likelihood computation
n = 3; # number of confidence criteria

R = 0.1; D = 0.1; L = 0.2; Z = -0.2;

params_init = (R/2,R/2,D/2,D/2,L,0.4,Z);

param_bounds = [(0.0,1.0),(-1.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(-1.0,1.0)];

# [Melchi 2/23/2015]: added provisions for using fftw for convolutions 
fftw.fftw_setup(zeros(NR_SSTEPS),NR_THREADS);

def find_ml_params(params_init,rem_RTs,know_RTs,new_RTs,nr_quantiles=4):
    objective_function = lambda x:compute_model_gof(x,rem_RTs,know_RTs,new_RTs,nr_quantiles);
    return opt.fmin(objective_function,params_init);
    #return opt.basinhopping(objective_function,params_init,stepsize=0.1);

def find_ml_params_fixed_means(params_init,rem_RTs,know_RTs,new_RTs,nr_quantiles=4):
    objective_function = lambda x:compute_model_gof(hstack([params_init[:2],x[2:]]),rem_RTs,know_RTs,new_RTs,nr_quantiles);
    return opt.fmin(objective_function,params_init);

def compute_model_nllr(model_params,rem_RTs,know_RTs,new_RTs,nr_quantiles=4):
    # computes the negative log of the ratio of the current model fit
    # to the best achievable fit
    
    # compute N, the total number of trials
    N = len(rem_RTs)+len(know_RTs)+len(new_RTs);
    # compute x, the observed frequency for each category
    rem_quantiles,know_quantiles,new_quantiles = compute_model_quantiles(model_params,nr_quantiles);
    ## compute the number of RTs falling into each quantile bin
    rem_freqs = -diff([sum(rem_RTs>q) for q in rem_quantiles]+[0]);
    know_freqs = -diff([sum(know_RTs>q) for q in know_quantiles]+[0]);
    new_freqs = -diff([sum(new_RTs>q) for q in new_quantiles]+[0]);
    x = hstack([rem_freqs,know_freqs,new_freqs]);
    
    # compute p the probability associated with each category
    p_rem = ones(nr_quantiles)*len(rem_RTs)/(nr_quantiles*float(N));
    p_know = ones(nr_quantiles)*len(know_RTs)/(nr_quantiles*float(N));
    p_new = ones(nr_quantiles)*len(new_RTs)/(nr_quantiles*float(N));
    p = hstack([p_rem,p_know,p_new]);
    return -multinom_loglike(x,N,p);#+multinom_loglike(N*p,N,p);

def compute_model_gof(model_params,rem_RTs,know_RTs,new_RTs,nr_quantiles=4):
    # computes the negative log of the ratio of the current model fit
    # to the best achievable fit
    
    # compute N, the total number of trials
    N = len(rem_RTs)+len(know_RTs)+len(new_RTs);
    # compute x, the observed frequency for each category
    rem_quantiles,know_quantiles,new_quantiles = compute_model_quantiles(model_params,nr_quantiles);
    ## compute the number of RTs falling into each quantile bin
    rem_freqs = -diff([sum(rem_RTs>q) for q in rem_quantiles]+[0]);
    know_freqs = -diff([sum(know_RTs>q) for q in know_quantiles]+[0]);
    new_freqs = -diff([sum(new_RTs>q) for q in new_quantiles]+[0]);
    x = hstack([rem_freqs,know_freqs,new_freqs]);
    
    # compute p the probability associated with each category
    p_rem = ones(nr_quantiles)*len(rem_RTs)/(nr_quantiles*float(N));
    p_know = ones(nr_quantiles)*len(know_RTs)/(nr_quantiles*float(N));
    p_new = ones(nr_quantiles)*len(new_RTs)/(nr_quantiles*float(N));
    p = hstack([p_rem,p_know,p_new]);
    return chi_square_gof(x,N,p)

def compute_model_quantiles(params,nr_quantiles=4):
    quantile_increment = 1.0/nr_quantiles;
    quantiles = arange(0,1,quantile_increment);
    # unpack model parameters
    mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0 = params;
    # compute marginal distributions
    p_remember,p_know,p_new,t = predicted_proportions(mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0);
    # compute marginal category proportions
    remember_total = p_remember.sum();
    know_total = p_know.sum();
    new_total = p_new.sum();
    # compute integrals of marginal distributions
    P_r = cumsum(p_remember)/remember_total;
    P_k = cumsum(p_know)/know_total;
    P_n = cumsum(p_new)/new_total;
    
    # compute RT quantiles
    rem_quantiles = array([t[argmax(P_r>q)] for q in quantiles]);
    know_quantiles = array([t[argmax(P_k>q)] for q in quantiles]);
    new_quantiles = array([t[argmax(P_n>q)] for q in quantiles]);
    rem_quantiles[0] = 0;
    know_quantiles[0] = 0;
    new_quantiles[0] = [0];
    # return quantile locations and marginal p(new)
    return rem_quantiles,know_quantiles,new_quantiles;

def predicted_proportions(mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,use_fftw=True):
    # Hack to enforce reasonable parameter bounds in fmin
    #mu_r = clip(mu_r,-1.0,1.0);
    #mu_f = clip(mu_f,-1.0,1.0);
    d_r = clip(d_r,1e-10,1.0);
    d_f = clip(d_f,1e-10,1.0);
    tc_bound = clip(tc_bound,0.0,1.0);
    #r_bound = clip(r_bound,0.0,1.0);
    # compute process SD
    sigma_r = sqrt(2*d_r*DELTA_T);
    sigma_f = sqrt(2*d_f*DELTA_T);
    sigma = sqrt(sigma_r**2+sigma_f**2);

    # compute the correlation for r given r+f
    rho = sigma_r/sigma;

    t = linspace(DELTA_T,MAX_T,NR_TSTEPS); # this is the time axis
    bound = exp(-tc_bound*t); # this is the collapsing bound
    
    ## Note that this means the lower temporal boundaries for arbitrary confidence
    ## criteria 'C' can be computed as t_C = -ln(C)/tc_bound
    
    mu = (mu_r+mu_f)*DELTA_T; # this is the average overall drift rate, with r = 'recall' and f = 'familiar'
    # compute the bounding limit of the space domain. This should include at least 99% of the probability mass when the particle is at the largest possible bound
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
    p_remember = zeros(shape(t));
    p_old = zeros(shape(t));
    p_new = zeros(shape(t));

    # take care of the first timestep
    tx[0] = stats.norm.pdf(x,mu+z0,sigma)*delta_s;
    p_old[0] = sum(tx[0][x>=bound[0]]);
    p_new[0] = sum(tx[0][x<=-bound[0]]);
    # compute STD(r) for the current time
    s_r = sigma_r;
    # compute STD(r+f) for the current time
    s_comb = sigma;
    # compute E[r|(r+f)]
    mu_r_cond = mu_r*t[0]+rho*s_r*(bound[0]-t[0]*(mu_r+mu_f))/s_comb;
    # compute STD[r|(r+f)]
    s_r_cond = s_r*sqrt(1-rho**2);
    p_remember[0] = p_old[0]*stats.norm.sf(r_bound,mu_r_cond,s_r_cond);
    # remove from consideration any particles that already hit the bound
    tx[0]*=(abs(x)<bound[0]);
    for i in range(1,len(t)):
        #tx[i] = convolve(tx[i-1],kernel,'same');
        # convolve the particle distribution from the previous timestep
        # with the diffusion kernel (using Fourier domain convolution)
        if(use_fftw):
            tx[i] = abs(ifftshift(fftw.ifft(fftw.fft(tx[i-1])*ft_kernel)));
        else:
            tx[i] = abs(ifftshift(ifft(fft(tx[i-1])*ft_kernel)));
        #tx[i] = abs(ifftshift(IFFT(FFT(tx[i-1])*ft_kernel)));
        p_pos = tx[i][x>=bound[i]]; # probability of each particle position above the upper bound
        x_pos = x[x>=bound[i]];     # location of each particle position above the upper bound

        p_old[i] = sum(p_pos); # total probability that particle crosses upper bound
        p_new[i] = sum(tx[i][x<=-bound[i]]); # probability that particle crosses lower bound
        
        # compute STD(r) for the current time
        s_r = sqrt(2*d_r*t[i]);
        # compute STD(r+f) for the current time
        #s_comb = sqrt(2*t[i]*(d_r+d_f));
        # compute E[r|(r+f)]
        #mu_r_cond = mu_r*t[i]+(bound[i]-t[i]*(mu_r+mu_f)-z0)*rho**2;
        mu_r_cond = mu_r*t[i]+(x_pos-t[i]*(mu_r+mu_f)-z0)*rho**2;
        # compute STD[r|(r+f)]
        s_r_cond = s_r*sqrt(1-rho**2);
        p_remember[i] = sum(p_pos*stats.norm.sf(r_bound,mu_r_cond,s_r_cond));
        # remove from consideration any particles that already hit the bound
        tx[i]*=(abs(x)<bound[i]);

    p_know = p_old-p_remember;
    return p_remember,p_know,p_new,t;


def predicted_proportions_sim(mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0):
    # compute process SD
    sigma_r = sqrt(2*d_r*DELTA_T);
    sigma_f = sqrt(2*d_f*DELTA_T);
    sigma = sqrt(sigma_r**2+sigma_f**2);
    
    t = linspace(DELTA_T,MAX_T,NR_TSTEPS);
    bound = exp(-tc_bound*t); # this is the collapsing bound

    # Now simulate NR_SAMPLES trials
    # 1. Generate a random position change for each time interval
    #   these position changes should be drawn from a normal distribution with mean
    #   mu and standard deviation sigma

    delta_r = stats.norm.rvs(mu_r*DELTA_T,sigma_r,size=(NR_SAMPLES,NR_TSTEPS));
    delta_f = stats.norm.rvs(mu_f*DELTA_T,sigma_f,size=(NR_SAMPLES,NR_TSTEPS));
    delta_pos = delta_r+delta_f;

    # 2. Use cumsum to compute absolute positions from delta_pos
    positions = pl.cumsum(delta_pos,1)+z0;
    r_positions = pl.cumsum(delta_r,1);
    # 3. Now loop through each sample trial to compute decisions and resp times
    decisions = [];
    resp_times = [];
    remembers = [];
    for i, pos in enumerate(positions):
        # Find the index where the position first crosses a boundary (i.e., 1 or -1)
        cross_indices = pl.find(abs(pos)>=bound);
        if len(cross_indices):
            cross_idx = cross_indices[0]; # take the first index
        else: # i.e., if no crossing was found
            cross_idx = NR_TSTEPS-1; # set it to the final index
        # 4. Now we can use this index to determine both the decision and the response time
        decision = pos[cross_idx]>0;
        resp_time = t[cross_idx]-0.5*DELTA_T; #i.e., the midpoint of the crossing interval
        remember = r_positions[i][cross_idx]>=r_bound;

        decisions.append(decision);
        resp_times.append(resp_time);
        remembers.append(remember);

    # Now estimate the joint distribution
    decisions = array(decisions);
    resp_times = array(resp_times);
    remembers = array(remembers);

    remember_RTs = resp_times[logical_and(remembers,decisions)];
    know_RTs = resp_times[logical_and(logical_not(remembers),decisions)];
    new_RTs = resp_times[logical_not(decisions)];
    
    params_rem = stats.gamma.fit(remember_RTs,floc=0);
    params_know = stats.gamma.fit(know_RTs,floc=0);
    params_new = stats.gamma.fit(new_RTs,floc=0);
    
    p_remember = stats.gamma.pdf(t,*params_rem)*DELTA_T*len(remember_RTs)/float(NR_SAMPLES);
    p_know = stats.gamma.pdf(t,*params_know)*DELTA_T*len(know_RTs)/float(NR_SAMPLES);
    p_new = stats.gamma.pdf(t,*params_new)*DELTA_T*len(new_RTs)/float(NR_SAMPLES);
    return p_remember,p_know,p_new,t;

def generate_samples(params,nr_samples=NR_SAMPLES,delta_t=DELTA_T):
    # unpack parameters
    mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0 = params;
    # compute process SD
    sigma_r = sqrt(2*d_r*delta_t);
    sigma_f = sqrt(2*d_f*delta_t);
    sigma = sqrt(sigma_r**2+sigma_f**2);
    
    nr_tsteps = int(ceil(MAX_T/delta_t));
    t = linspace(delta_t,MAX_T,nr_tsteps);
    bound = exp(-tc_bound*t); # this is the collapsing bound

    # Now simulate NR_SAMPLES trials
    # 1. Generate a random position change for each time interval
    #   these position changes should be drawn from a normal distribution with mean
    #   mu and standard deviation sigma

    delta_r = stats.norm.rvs(mu_r*delta_t,sigma_r,size=(nr_samples,nr_tsteps));
    delta_f = stats.norm.rvs(mu_f*delta_t,sigma_f,size=(nr_samples,nr_tsteps));
    delta_pos = delta_r+delta_f;

    # 2. Use cumsum to compute absolute positions from delta_pos
    positions = pl.cumsum(delta_pos,1)+z0;
    r_positions = pl.cumsum(delta_r,1);
    # 3. Now loop through each sample trial to compute decisions and resp times
    decisions = [];
    resp_times = [];
    remembers = [];
    for i, pos in enumerate(positions):
        # Find the index where the position first crosses a boundary (i.e., 1 or -1)
        cross_indices = pl.find(abs(pos)>=bound);
        if len(cross_indices):
            cross_idx = cross_indices[0]; # take the first index
        else: # i.e., if no crossing was found
            cross_idx = nr_tsteps-1; # set it to the final index
        # 4. Now we can use this index to determine both the decision and the response time
        decision = pos[cross_idx]>0;
        resp_time = t[cross_idx]-0.5*delta_t; #i.e., the midpoint of the crossing interval
        remember = r_positions[i][cross_idx]>=r_bound;

        decisions.append(decision);
        resp_times.append(resp_time);
        remembers.append(remember);

    # Now estimate the joint distribution
    decisions = array(decisions);
    resp_times = array(resp_times);
    remembers = array(remembers);

    remember_RTs = resp_times[logical_and(remembers,decisions)];
    know_RTs = resp_times[logical_and(logical_not(remembers),decisions)];
    new_RTs = resp_times[logical_not(decisions)];
    
    return remember_RTs,know_RTs,new_RTs;
