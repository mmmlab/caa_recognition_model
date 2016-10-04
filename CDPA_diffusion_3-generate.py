
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
MAX_T       = 20; #(i.e., 1 minute)
NR_TSTEPS   = MAX_T/DELTA_T;
NR_SSTEPS   = 8192#4096#2048;
NR_SAMPLES  = 10000; # number of trials to use for MC likelihood computation
n = 2; # number of confidence critetion

R = 0.1; D = 0.05; L = 0.1; Z = 0.0;
#observed = numpy.loadtxt('myData.txt'); # load the observed data
#observed


# In[83]:

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
    
    # generate the RT distribution with confidence (assuming a total of 4000 trials)
    rem_RTs=[];
    know_RTs=[];
    confidence_rem=[];
    confidence_know=[];
    
    for j in range(size(t)):
        rem_RT = t[j];
        know_RT = t[j];
        if j <= t_C[0]: # t_C[0] corresponds to the temporal index for highest confidence
            conf_rem = 2
            conf_know = 2
        elif (j > t_C[0]) and (j<= t_C[1]): #t_C[1] corresponds to the temporal index for medium confidence
            conf_rem = 1;
            conf_know = 1;
        else:
            conf_rem = 0;
            conf_know = 0;
        if (rem_RT):
            for k in range(int(p_remember[j]*4000.0)):
                rem_RTs.append(rem_RT);
                confidence_rem.append(conf_rem);
        if(know_RT):
            for m in range(int(p_know[j]*4000.0)):
                know_RTs.append(know_RT);
                confidence_know.append(conf_know);
            
      
    rem_RTs = array(rem_RTs);
    know_RTs = array(know_RTs);
    confidence_rem=array(confidence_rem);
    confidence_know=array(confidence_know);
    
    Remember = vstack((rem_RTs,confidence_rem));
    Know = vstack((know_RTs,confidence_know));
    
    return data*4000,Remember.T,Know.T;
    #return p_remember,p_know,p_new,t;


# In[97]:

data,remember_hit,know_hit= predicted_proportions(array([0.78459678,0.63952194]),0.05422463,0.07452259,0.10671192,0.06402805,0.11025454,0.5152371,2.7871542,-0.12625914);
remH_RT,remH_conf = numpy.split(remember_hit,2,axis=1);
knowH_RT,knowH_conf = numpy.split(know_hit,2,axis=1);
print data;
print "rem hit high", size(remH_RT[remH_conf==2]);
print "rem hit med", size(remH_RT[remH_conf==1]);
print "rem hit low", size(remH_RT[remH_conf==0]);
print "know hit high", size(knowH_RT[knowH_conf==2]);
print "know hit med", size(knowH_RT[knowH_conf==1]);
print "know hit low", size(knowH_RT[knowH_conf==0]);


# Out[97]:

#     [[  848.31626927   490.80124766     0.        ]
#      [  364.99685766   396.24921546     0.        ]
#      [   84.2443294    128.4317921      0.        ]
#      [    0.             0.          1645.62808806]]
#     rem hit high 815
#     rem hit med 322
#     rem hit low 35
#     know hit high 460
#     know hit med 356
#     know hit low 69
#     

# In[94]:

def generate(parameters): 
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
    
    data_old,remember_hit,know_hit = predicted_proportions(c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,f_bound,z0);
    data_new,remember_fa,know_fa= predicted_proportions(c,mu_r0,mu_f0,d_r0,d_f0,tc_bound,r_bound,f_bound,z0);
    predicted_data = vstack((data_old,data_new));
    numpy.savetxt("genData.txt", predicted_data);
    numpy.savetxt("gen_rem_hit.txt", remember_hit);
    numpy.savetxt("gen_know_hit.txt", know_hit);
    numpy.savetxt("gen_rem_fa.txt", remember_fa);
    numpy.savetxt("gen_know_fa.txt", know_fa);


# In[98]:

#param = array([0.78459678,0.63952194,0.05422463,0.07452259,0.10671192,0.06402805,0.11025454,0.5152371,2.7871542,-0.12625914,-0.09405059,-0.01071414,0.06782042,0.05254881]);
param = array([0.77370433,0.64656344,0.04382392,0.06021717,0.13038513,0.06539262,0.11125704,0.46793749,2.38316611,-0.14153438,-0.12914206,-0.01057224,0.06711001,0.05758781]);
generate(param);


# In[ ]:



