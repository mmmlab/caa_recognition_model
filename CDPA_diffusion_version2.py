
# In[1]:

from pylab import *
import numpy as np
import pylab as pl
from scipy import stats
from scipy import optimize
from scipy.integrate import fixed_quad 


# In[2]:

# Constants
DELTA_T     = 0.025;  # size of discrete time increment (sec.)
MAX_T       = 20.0; #(i.e., 1 minute)
NR_TSTEPS   = MAX_T/DELTA_T;
NR_SSTEPS   = 8192#4096#2048;
NR_SAMPLES  = 10000; # number of trials to use for MC likelihood computation
n = 3; # number of confidence critetion

R = 0.1; D = 0.05; L = 0.1; Z = 0.0;


# In[10]:

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
    
    # integrating individual distributions 
    integrand1 = lambda T: (stats.gamma.pdf(T,*params_rem)*DELTA_T*len(remember_RTs)/float(NR_SAMPLES));
    integrand2 = lambda T: (stats.gamma.pdf(T,*params_know)*DELTA_T*len(know_RTs)/float(NR_SAMPLES));
    integrand3 = lambda T: (stats.gamma.pdf(T,*params_new)*DELTA_T*len(new_RTs)/float(NR_SAMPLES));
    rem = fixed_quad(integrand1,0,MAX_T,n=20);
    know = fixed_quad(integrand2,0,MAX_T,n=20);
    new = fixed_quad(integrand3,0,MAX_T,n=20);
    
    return rem[0],know[0],new[0];
    #return p_remember,p_know,p_new,t;


# In[11]:

pr,pk,pn,tot= predicted_proportions_sim(3*R/4,R/4,D/2,D/2,L,0.6,0.2);
print pr+pk+pn;


# Out[11]:


    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)

    <ipython-input-11-5e59e3a83305> in <module>()
    ----> 1 pr,pk,pn,tot= predicted_proportions_sim(3*R/4,R/4,D/2,D/2,L,0.6,0.2);
          2 print pr+pk+pn;
    

    ValueError: need more than 3 values to unpack


# In[8]:

1-tot


# Out[8]:

#     0.12509999999999999

# In[ ]:



