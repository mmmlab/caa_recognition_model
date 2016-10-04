
# In[1]:

from pylab import *
#import numpy as np
import pylab as pl
import numpy
from scipy import stats
from scipy import optimize
from scipy.integrate import fixed_quad 

## Start by reading in the data.
# the reason to do this first is that, in order to be efficient,
# we don't want to represent any more of the time axis than we have to.

################################################################################
observed = numpy.loadtxt('neha/myData.txt'); # load the observed data
remember_hit = numpy.loadtxt('neha/remRT_hit.txt'); # load remember RTs for hits
know_hit = numpy.loadtxt('neha/knowRT_hit.txt'); # load know RTs for hits
remember_fa = numpy.loadtxt('neha/remRT_fa.txt'); # load remember RTs for false alarms
know_fa = numpy.loadtxt('neha/knowRT_fa.txt');  # load know RTs for false alarms

remH_RT,remH_conf = numpy.split(remember_hit,2,axis=1);
knowH_RT,knowH_conf = numpy.split(know_hit,2,axis=1);
remFA_RT,remFA_conf = numpy.split(remember_fa,2,axis=1);
knowFA_RT,knowFA_conf = numpy.split(know_fa,2,axis=1);

all_RT = vstack([remH_RT,remFA_RT,knowH_RT,knowFA_RT]);
################################################################################


DELTA_T     = 0.025;  # size of discrete time increment (sec.)
MAX_T       = ceil(percentile(all_RT,99.5))
NR_TSTEPS   = MAX_T/DELTA_T;
NR_SSTEPS   = 8192#4096#2048;
NR_SAMPLES  = 10000; # number of trials to use for MC likelihood computation
n = 2; # number of confidence critetion
QUANT = array([0.25,0.50,0.75]);
QUANT_DIFF = 0.25;


# In[3]:

R = 0.1; D = 0.05; L = 0.1; Z = 0.0;

print observed;


# Out[3]:

#     [[  918.   526.     0.]
#      [  362.   431.     0.]
#      [   23.    47.     0.]
#      [    0.     0.  1661.]
#      [  196.   164.     0.]
#      [  228.   356.     0.]
#      [   28.    59.     0.]
#      [    0.     0.  2937.]]
#     

# In[18]:


print "rem hit high", remH_RT[remH_conf==2].mean();
print "rem hit med", remH_RT[remH_conf==1].mean();
print "rem hit low", remH_RT[remH_conf==0].mean();
print "know hit high", knowH_RT[knowH_conf==2].mean();
print "know hit med", knowH_RT[knowH_conf==1].mean();
print "know hit low", knowH_RT[knowH_conf==0].mean();


# Out[18]:

#     rem hit high 1.82368573676
#     rem hit med 2.1346722535
#     rem hit low 2.20608522991
#     know hit high 1.60773045629
#     know hit med 2.17318544077
#     know hit low 2.57370984398
#     

# In[4]:

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
    
    ## Note that this means the higher temporal boundaries for arbitrary confidence
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
    # [Melchi] I'm not sure I understand what is being computed below.
    # Why was this changed from the original code that directly computed
    # p_remember, and what does 'f_bound' represent? [Melchi]
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

        p_know[i] = sum(p_pos*stats.norm.sf(f_bound,mu_f_cond,s_f_cond))+sum(p_pos*stats.norm.cdf(r_bound,mu_r_cond,s_r_cond));
        # remove from consideration any particles that already hit the bound
        tx[i]*=(abs(x)<bound[i]);

    p_remember = p_old-p_know;
    
    ######################################################################################
    # determine the proportion of remember and know responses by confidence
    rem = zeros((n+1,1));
    know = zeros((n+1,1));
    quant_rem = zeros((n+1,size(QUANT)));
    quant_know = zeros((n+1,size(QUANT)));
     
    for i,hi_edge in enumerate(t_C):
        # find the distribution bounds for the current confidence level
        if(i==0):
            lo_edge = 0;
        else:
            lo_edge = t_C[i-1];
            
        
        # compute the CDF for this quantile
        #prob_rem = p_remember[logical_and(t>=lo_edge,t<hi_edge)];
        prob_rem = p_remember[lo_edge:hi_edge];
        rem[i] = prob_rem.sum()
        prob_rem = cumsum(prob_rem/rem[i]);
        
        
        #prob_know = p_know[logical_and(t>=lo_edge,t<hi_edge)];
        prob_know = p_know[lo_edge:hi_edge];
        know[i] = prob_know.sum();
        prob_know = cumsum(prob_know/know[i]);
        
        
        # find the index of the CDF value that most closely matches the desire quantile rank.
        # the time associated with that index is the quantile value
        t_temp = t[lo_edge:hi_edge];
        #quant_rem[i,:] = array([t_temp[argmin(abs(prob_rem-q))] for q in QUANT]);
        #quant_know[i,:] = array([t_temp[argmin(abs(prob_know-q))] for q in QUANT]);
        
        # The following version is more efficent
        quant_rem[i,:] = array([t_temp[argmax(prob_rem>q)] for q in QUANT]);
        quant_know[i,:] = array([t_temp[argmax(prob_know>q)] for q in QUANT]);
        
    
    prob_rem = p_remember[t_C[n-1]:];
    rem[n] = prob_rem.sum();
    prob_rem = cumsum(prob_rem/rem[n]);
    
    prob_know = p_know[t_C[n-1]:];
    know[n] = prob_know.sum();
    prob_know = cumsum(prob_know/know[n]);
    
    t_temp = t[t_C[n-1]:];
    quant_rem[n,:] = array([t_temp[argmin(abs(prob_rem-q))] for q in QUANT]);
    quant_know[n,:] = array([t_temp[argmin(abs(prob_know-q))] for q in QUANT]); 
    
    temp = zeros((n+1,1));
    data = vstack((hstack((rem,know,temp)),array([0,0,sum(p_new)])));
    data_quant = vstack((quant_rem,quant_know));
    
    #return data;
    return data,data_quant;
    #return sum(p_remember),sum(p_know),sum(p_new),t;


#get_ipython().magic(u'timeit data,dataquant= predicted_proportions(array([0.8,0.5]),3*R/4,R/4,D/2,D/2,L,0.6,0.9,0.2);')




def compute_proportion(parameters):
    # unpack parameters
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
    
    # compute predicted proportion matrix for 'old' words
    data_old,quant_old = predicted_proportions(c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,f_bound,z0);
    # compute predicted proportion matrix for 'new' words
    data_new,quant_new = predicted_proportions(c,mu_r0,mu_f0,d_r0,d_f0,tc_bound,r_bound,f_bound,z0);
    predicted_data = vstack((data_old,data_new));
    return predicted_data,quant_old,quant_new;


# In[8]:

param = array([0.8,0.5,3*R/4,R/4,D/2,D/2,L,0.6,0.9,0.2,3*R/4,R/4,D/2,D/2]);
prop,dataold,datanew = compute_proportion(param);
print prop;
print dataold;
print datanew;


# Out[8]:

#     [[ 0.03597732  0.24335071  0.        ]
#      [ 0.10122512  0.39883889  0.        ]
#      [ 0.01872873  0.05082803  0.        ]
#      [ 0.          0.          0.13279225]
#      [ 0.03597732  0.24335071  0.        ]
#      [ 0.10122512  0.39883889  0.        ]
#      [ 0.01872873  0.05082803  0.        ]
#      [ 0.          0.          0.13279225]]
#     [[ 1.325  1.675  1.95 ]
#      [ 3.     3.9    5.075]
#      [ 7.45   8.125  9.125]
#      [ 1.2    1.55   1.875]
#      [ 2.85   3.65   4.825]
#      [ 7.4    8.025  8.975]]
#     [[ 1.325  1.675  1.95 ]
#      [ 3.     3.9    5.075]
#      [ 7.45   8.125  9.125]
#      [ 1.2    1.55   1.875]
#      [ 2.85   3.65   4.825]
#      [ 7.4    8.025  8.975]]
#     

# In[9]:

def chi_square_quantile(quantile_edge,remember,know,rem_prop,know_prop,total):
    rem_edge = quantile_edge[0:n+1,0:size(QUANT)];
    know_edge = quantile_edge[n+1:2*(n+1),0:size(QUANT)];
    rem_RT,rem_conf = numpy.split(remember,2,axis=1);
    know_RT,know_conf = numpy.split(know,2,axis=1);
    chi = zeros((n+1));
    for i in range(n+1):
        #print i;
        # rem_data = observed reaction times for 'remember' judgments
        rem_data = rem_RT[rem_conf==n-i];
        # know_data = observed reaction times for 'know' judgments
        know_data = know_RT[know_conf==n-i];
        chi[i]= (((total*rem_prop[i]*QUANT_DIFF-sum(rem_data<rem_edge[i,0]))**2)/total*rem_prop[i]*QUANT_DIFF)+(((total*know_prop[i]*QUANT_DIFF-sum(know_data<know_edge[i,0]))**2)/total*know_prop[i]*QUANT_DIFF);
        total_predicted_rem = total*rem_prop[i]*QUANT_DIFF;
        total_predicted_know = total*know_prop[i]*QUANT_DIFF;
        total_observed_rem = sum(rem_data<rem_edge[i,0]);
        total_observed_know = sum(rem_data<know_edge[i,0]);
        
        for j in range(1,size(QUANT)):
            observed_rem = sum(rem_data<rem_edge[i,j])-sum(rem_data<rem_edge[i,j-1]);
            predicted_rem = total*rem_prop[i]*QUANT_DIFF
            observed_know = sum(know_data<know_edge[i,j])-sum(know_data<know_edge[i,j-1]);
            predicted_know = total*know_prop[i]*QUANT_DIFF
            total_predicted_rem += predicted_rem;
            total_predicted_know += predicted_know;
            total_observed_rem += observed_rem;
            total_observed_know += observed_know
            chi[i] += ((predicted_rem-observed_rem)**2)/predicted_rem + ((predicted_know-observed_know)**2)/predicted_know;
        chi[i] += (((total*rem_prop[i]-total_predicted_rem)-(size(rem_data)-total_observed_rem))**2)/(total*rem_prop[i]-total_predicted_rem)+(((total*know_prop[i]-total_predicted_know)-(size(know_data)-total_observed_know))**2)/(total*know_prop[i]-total_predicted_know);
    return sum(chi);     


# In[10]:

def chi_square(parm,observed=observed,remember_hit=remember_hit,know_hit=know_hit,remember_fa=remember_fa,know_fa=know_fa):
    old = sum(observed[0:n+2,0:3]);
    new = sum(observed[n+2:2*(n+2),0:3]);
    predicted,quantile_edge_old,quantile_edge_new = compute_proportion(parm);
    chi_hit = chi_square_quantile(quantile_edge_old,remember_hit,know_hit,predicted[0:n+1,0],predicted[0:n+1,1],old);
    chi_fa = chi_square_quantile(quantile_edge_new,remember_fa,know_fa,predicted[n+2:2*(n+2)-1,0],predicted[n+2:2*(n+2)-1,1],new);
    chimiss = ((predicted[3,2]*old-observed[3,2])**2)/(predicted[3,2]*old);
    chicr = ((predicted[7,2]*new-observed[7,2])**2)/(predicted[7,2]*new);
    chi = chi_hit+chi_fa+chimiss+chicr;
    return chi;


## In[21]:
#
##param = array([0.8,0.5,3*R/4,R/4,D/2,D/2,L,0.6,0.9,0.2,3*R/4,R/4,D/2,D/2]);
#param = array([0.7831307,0.60253825,0.02923674,0.07357183,0.16362669,0.06709437,0.10826691,0.49595259,1.34030801,-0.1552733,-0.14614531,-0.0123076,0.07349555,0.06135621]);
#chi = chi_square(param);
#chi
#
#
## Out[21]:
#
##     587.44359692654484
#
## In[22]:
#
##param_initial = array([0.8,0.5,3*R/4,R/4,D/2,D/2,L,0.6,0.9,0.2,3*R/4,R/4,D/2,D/2]);
#param_initial = array([0.7831307,0.60253825,0.02923674,0.07357183,0.16362669,0.06709437,0.10826691,0.49595259,1.34030801,-0.1552733,-0.14614531,-0.0123076,0.07349555,0.06135621]);
#for i in range(10):
#    output = optimize.minimize(chi_square,param_initial,method='Nelder-Mead'); # estimate parameters that minimize the negative log likelihood
#    print output.x;
#    print output.message;
#    param_initial=output.x;
#    chi = chi_square(output.x); # check the value of chi square with estimated parameters (sanity check?)
#    print chi;
#    print "end"



