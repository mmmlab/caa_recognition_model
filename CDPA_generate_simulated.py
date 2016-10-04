
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
NR_SAMPLES  = 20000; # number of trials to use for MC likelihood computation
n = 2; # number of confidence critetion

R = 0.1; D = 0.05; L = 0.1; Z = 0.0;
#observed = numpy.loadtxt('myData.txt'); # load the observed data
#observed


# In[11]:

def predicted_proportions_sim(c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,f_bound,z0):
    # compute process SD
    sigma_r = sqrt(2*d_r*DELTA_T);
    sigma_f = sqrt(2*d_f*DELTA_T);
    sigma = sqrt(sigma_r**2+sigma_f**2);
    
    t = linspace(DELTA_T,MAX_T,NR_TSTEPS);
    #t = linspace(0.001,3.0,NR_TSTEPS);
    bound = exp(-tc_bound*t); # this is the collapsing bound

    t_C = zeros((n));
    Time = zeros((n));
    for i in range(n):
        Time [i] = -log(c[i])/tc_bound;
        temp_index = pl.find(Time[i]<=t);
        t_C[i]= temp_index[0];   
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
    f_positions = pl.cumsum(delta_f,1);
    # 3. Now loop through each sample trial to compute decisions and resp times
    decisions = [];
    resp_times = [];
    remembers = [];
    confidences = [];
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
        #remember = r_positions[i][cross_idx]>=r_bound;
        
        if((f_positions[i][cross_idx]>=f_bound)or(r_positions[i][cross_idx]<r_bound)):
            remember=0;
        elif (r_positions[i][cross_idx]>=r_bound):
            remember = 1;
        else:
            remember = 0;
            
        if resp_time <= Time[0]:
            confidence = 2;
        elif (resp_time > Time[0]) and (resp_time <= Time[1]):
            confidence = 1;
        else:
            confidence = 0;

        decisions.append(decision);
        resp_times.append(resp_time);
        remembers.append(remember);
        confidences.append(confidence);

    # Now estimate the joint distribution
    decisions = array(decisions);
    resp_times = array(resp_times);
    remembers = array(remembers);
    confidences = array(confidences);

    rem_conf = confidences[logical_and(remembers,decisions)];
    know_conf = confidences[logical_and(logical_not(remembers),decisions)];
    remember_RTs = resp_times[logical_and(remembers,decisions)];
    know_RTs = resp_times[logical_and(logical_not(remembers),decisions)];
    new_RTs = resp_times[logical_not(decisions)];
    
    params_rem = stats.gamma.fit(remember_RTs,floc=0);
    params_know = stats.gamma.fit(know_RTs,floc=0);
    params_new = stats.gamma.fit(new_RTs,floc=0);
     
    p_remember = stats.gamma.pdf(t,*params_rem)*DELTA_T*len(remember_RTs)/float(NR_SAMPLES);
    p_know = stats.gamma.pdf(t,*params_know)*DELTA_T*len(know_RTs)/float(NR_SAMPLES);
    p_new = stats.gamma.pdf(t,*params_new)*DELTA_T*len(new_RTs)/float(NR_SAMPLES);
    
    
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
    
    Remember = vstack((remember_RTs,rem_conf));
    Know = vstack((know_RTs,know_conf));
    
    return data,Remember.T,Know.T;  
    #return rem[0],know[0],new[0],t;
    #return sum(p_remember),sum(p_know),sum(p_new),t;
    #return p_remember,p_know,p_new,t;


# In[12]:

#pr,pk,pn,t = predicted_proportions_sim(array([0.8,0.5]),3*R/4,R/4,D/2,D/2,L,0.6,0.9,0.2);
#plot(t,pr,'b',label = 'remember');
#plot(t,pk,'g',label='know');
#plot(t,pn,'r',label='new');
#legend(loc = 'upper right');
data,remember_hit,know_hit= predicted_proportions_sim(array([0.8,0.5]),3*R/4,R/4,D/2,D/2,L,0.6,0.9,0.2);


# In[15]:

data1,remember_hit1,know_hit1= predicted_proportions_sim(array([0.8,0.5]),3*R/4,R/4,D/2,D/2,L,0.6,0.9,0.2);
remH_RT,remH_conf = numpy.split(remember_hit1,2,axis=1);
knowH_RT,knowH_conf = numpy.split(know_hit1,2,axis=1);
print data1*NR_SAMPLES;
print "rem hit high", size(remH_RT[remH_conf==2]);
print "rem hit med", size(remH_RT[remH_conf==1]);
print "rem hit low", size(remH_RT[remH_conf==0]);
print "know hit high", size(knowH_RT[knowH_conf==2]);
print "know hit med", size(knowH_RT[knowH_conf==1]);
print "know hit low", size(knowH_RT[knowH_conf==0]);


# Out[15]:

#     [[  708.74506219  4582.8648632      0.        ]
#      [ 2288.58969839  8708.421876       0.        ]
#      [  355.5957949    791.65447742     0.        ]
#      [    0.             0.          2563.93568115]]
#     rem hit high 767
#     rem hit med 2170
#     rem hit low 416
#     know hit high 5048
#     know hit med 8010
#     know hit low 1025
#     

# In[16]:

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
    
    data_old,remember_hit,know_hit = predicted_proportions_sim(c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,f_bound,z0);
    data_new,remember_fa,know_fa= predicted_proportions_sim(c,mu_r0,mu_f0,d_r0,d_f0,tc_bound,r_bound,f_bound,z0);
    predicted_data = vstack((data_old,data_new));
    numpy.savetxt("genData.txt", predicted_data*NR_SAMPLES);
    numpy.savetxt("gen_rem_hit.txt", remember_hit);
    numpy.savetxt("gen_know_hit.txt", know_hit);
    numpy.savetxt("gen_rem_fa.txt", remember_fa);
    numpy.savetxt("gen_know_fa.txt", know_fa);


param = array([0.8,0.5,3*R/4,R/4,D/2,D/2,L,0.6,0.9,0.2,3*R/4,R/4,D/2,D/2]);
#param = array([0.7831307,0.60253825,0.02923674,0.07357185,0.1636267,0.06709437,0.10826691,0.49595259,1.34030798,-0.15527331,-0.14614542,-0.0123076,0.07349555,0.06135621]);
generate(param);




