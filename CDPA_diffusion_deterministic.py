from pylab import *
import numpy as np
import pylab as pl
from scipy import stats
from scipy import optimize
from scipy.integrate import fixed_quad 


# Constants
DELTA_T     = 0.025;  # size of discrete time increment (sec.)
MAX_T       = 20; #(i.e., 1 minute)
NR_TSTEPS   = MAX_T/DELTA_T;
NR_SSTEPS   = 8192#4096#2048;
NR_SAMPLES  = 10000; # number of trials to use for MC likelihood computation
n = 2; # number of confidence critetion

R = 0.1; D = 0.05; L = 0.1; Z = 0.0;
observed = numpy.loadtxt('genData.txt'); # load the observed data
observed


# Out[2]:

#     array([[  150.136507  ,   964.30955485,     0.        ],
#            [  455.06551089,  1714.08984407,     0.        ],
#            [   73.77821667,   147.59024649,     0.        ],
#            [    0.        ,     0.        ,   494.99060954],
#            [  147.23156078,   933.87395198,     0.        ],
#            [  446.60996393,  1760.7390149 ,     0.        ],
#            [   63.14882566,   157.37592496,     0.        ],
#            [    0.        ,     0.        ,   490.99655556]])

# In[3]:

def predicted_proportions(c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,f_bound,z0):
    # compute process SD
    sigma_r = sqrt(2*d_r*DELTA_T);
    sigma_f = sqrt(2*d_f*DELTA_T);
    sigma = sqrt(sigma_r**2+sigma_f**2);

    # compute the correlation for r given r+f
    rho = sigma_r/sigma;
    rhoF = sigma_f/sigma;

    t = linspace(DELTA_T,MAX_T,NR_TSTEPS); # this is the time axis
    bound = exp(-tc_bound*t); # this is the collapsing bound
    
    ## Note that this means the lower temporal boundaries for arbitrary confidence
    ## criteria 'C' can be computed as t_C = -ln(C)/tc_bound
    t_C = zeros((n));
    for i in range(n):
        temp_time = -log(c[i])/tc_bound;
        temp_index = pl.find(temp_time<=t);
        t_C[i]= temp_index[0];
    
    mu = (mu_r+mu_f)*DELTA_T; # this is the average overall drift rate, with r = 'recall' and f = 'familiar'
    # compute the bounding limit of the space domain. This should include at least 99% of the probability mass when the particle is at the largest possible bound
    space_lim = max(bound)+3*sigma;
    delta_s = 2*space_lim/NR_SSTEPS;
    # finally, construct the space axis
    x = linspace(-space_lim,space_lim,NR_SSTEPS);
    # compute the diffusion kernel
    kernel = stats.norm.pdf(x,mu,sigma)*delta_s;
    # ... and its Fourier transform. We'll use this to compute FD convolutions
    ft_kernel = fft(kernel);
    tx = zeros((len(t),len(x)));
    p_know = zeros(shape(t));
    p_old = zeros(shape(t));
    p_new = zeros(shape(t));

    # take care of the first timestep
    tx[0] = stats.norm.pdf(x,mu+z0,sigma)*delta_s;
    p_old[0] = sum(tx[0][x>=bound[0]]);
    p_new[0] = sum(tx[0][x<=-bound[0]]);
    # compute STD(r) for the current time
    s_r = sigma_r;
    s_f = sigma_f;
    # compute STD(r+f) for the current time
    s_comb = sigma;
    # compute E[r|(r+f)]
    mu_r_cond = mu_r*t[0]+rho*s_r*(bound[0]-t[0]*(mu_r+mu_f))/s_comb;
    mu_f_cond = mu_f*t[0]+rhoF*s_f*(bound[0]-t[0]*(mu_r+mu_f))/s_comb;
    # compute STD[r|(r+f)]
    s_r_cond = s_r*sqrt(1-rho**2);
    s_f_cond = s_f*sqrt(1-rhoF**2);
    
    p_know[0] = p_old[0]*stats.norm.sf(f_bound,mu_f_cond,s_f_cond)+p_old[0]*stats.norm.cdf(r_bound,mu_r_cond,s_r_cond);
    
    # remove from consideration any particles that already hit the bound
    tx[0]*=(abs(x)<bound[0]);
    for i in range(1,len(t)):
        #tx[i] = convolve(tx[i-1],kernel,'same');
        # convolve the particle distribution from the previous timestep
        # with the diffusion kernel (using Fourier domain convolution)
        tx[i] = abs(ifftshift(ifft(fft(tx[i-1])*ft_kernel)));
        p_pos = tx[i][x>=bound[i]]; # probability of each particle position above the upper bound
        x_pos = x[x>=bound[i]];     # location of each particle position above the upper bound

        #print "x_pos", size(x_pos);
        p_old[i] = sum(p_pos); # total probability that particle crosses upper bound
        p_new[i] = sum(tx[i][x<=-bound[i]]); # probability that particle crosses lower bound
        
        # compute STD(r) for the current time
        s_r = sqrt(2*d_r*t[i]);
        s_f = sqrt(2*d_f*t[i]);
        # compute STD[r|(r+f)]
        s_r_cond = s_r*sqrt(1-rho**2);
        s_f_cond = s_f*sqrt(1-rhoF**2);
        # compute E[r|(r+f)]
        mu_r_cond = mu_r*t[i]+(x_pos-t[i]*(mu_r+mu_f)-z0)*rho**2;
        mu_f_cond = mu_f*t[i]+(x_pos-t[i]*(mu_r+mu_f)-z0)*rhoF**2;
        
        #print "mu_f", size(mu_f_cond);

        p_know[i] = sum(p_pos*stats.norm.sf(f_bound,mu_f_cond,s_f_cond))+sum(p_pos*stats.norm.cdf(r_bound,mu_r_cond,s_r_cond));
        # remove from consideration any particles that already hit the bound
        tx[i]*=(abs(x)<bound[i]);

    p_remember = p_old-p_know;
    
    # determine the proportion of remember and know responses by confidence
    rem = zeros((n+1,1));
    know = zeros((n+1,1));
    
    know[0]= sum(p_know[0:t_C[0]]);
    rem[0] = sum(p_remember[0:t_C[0]]);
    
    for i in range(1,n):
        know[i]= sum(p_know[0:t_C[i]])-sum(know[0:i]);
        rem[i] = sum(p_remember[0:t_C[i]])-sum(rem[0:i]);
    know[n] = sum(p_know)-sum(know[0:n]);
    rem[n] = sum(p_remember)-sum(rem[0:n]);
    
    temp = zeros((n+1,1));
    data = vstack((hstack((rem,know,temp)),array([0,0,sum(p_new)])));
    
    return data;
    #return p_remember,p_know,p_new,t;


def compute_proportion(parameters): 
    c = parameters[0:n];
    mu_r = parameters[n+0]; 
    mu_f = parameters[n+1]; 
    d_r = parameters[n+2]; 
    d_f = parameters[n+3]; 
    tc_bound = parameters[n+4]; 
    r_bound = parameters[n+5];
    f_bound = parameters[n+6];
    z0 = parameters[n+7]; 
    mu_r0 = parameters[n+8]; 
    mu_f0 = parameters[n+9]; 
    d_r0 = parameters[n+10]; 
    d_f0 = parameters[n+11]; 
    
    data_old = predicted_proportions(c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,f_bound,z0);
    data_new = predicted_proportions(c,mu_r0,mu_f0,d_r0,d_f0,tc_bound,r_bound,f_bound,z0);
    predicted_data = vstack((data_old,data_new));  
    return predicted_data;


def chi_square(parm,observed=observed):
    #total = sum(observed);
    old = sum(observed[0:n+2,0:3]);
    new = sum(observed[n+2:2*(n+2),0:3]);
    predicted = compute_proportion(parm);
    chi = 0;
    for i in range(size(observed,0)): 
        for j in range(size(observed,1)):
            if predicted[i,j]>0:
                if i <= n+1:
                    #print "old",i,j;
                    predicted[i,j]=predicted[i,j]*old;
                else:
                    #print "new",i,j;
                    predicted[i,j]=predicted[i,j]*new;
                chi = chi + ((predicted[i,j]-observed[i,j])**2)/predicted[i,j];
    return chi;

param = array([0.8,0.5,3*R/4,R/4,D/2,D/2,L,0.6,0.9,0.2,3*R/4,R/4,D/2,D/2]);
chi = chi_square(param);
chi
