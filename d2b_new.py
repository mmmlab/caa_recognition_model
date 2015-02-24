from pylab import *
import numpy as np
import pylab as pl
from scipy import stats
from scipy import optimize
import fftw_test as fftw
#from reikna.fft import FFT,IFFT

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

R = 0.1; D = 0.05; L = 0.1; Z = 0.0;

# [Melchi 2/23/2015]: added provisions for using fftw for convolutions 
fftw.fftw_setup(zeros(NR_SSTEPS),NR_THREADS);

def predicted_proportions(mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,use_fftw=True):
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
