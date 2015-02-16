from pylab import *
import numpy as np
import pylab as pl
from scipy import stats
from pylab import array, linspace, arange, sign, sqrt, log, sum
from scipy import optimize
from operator import itemgetter

# Constants
DELTA_T     = 0.1;  # size of discrete time increment (sec.)
MAX_T       = 60.0; #(i.e., 1 minute)
NR_TSTEPS  = MAX_T/DELTA_T;
NR_SAMPLES  = 1000; # number of trials to use for MC likelihood computation

# D is diffusion constant
# r is drift rate
# z is starting point

def generate_sample(r,D,L,a2,a1):     #z=0):
    # compute process SD
    sigma = sqrt(2*D*DELTA_T);
    
    t = linspace(DELTA_T,MAX_T,NR_TSTEPS);
    # compute exponentially decreasing bounds
    bound = 2.0*exp(-L*t);
    # compute mean drift over a single time interval
    mu = r*DELTA_T;
    # Now simulate an individual trial
    # 1. Generate a random position change for each time interval
    #   these position changes should be drawn from a normal distribution with mean
    #   mu and standard deviation sigma
    delta_pos = stats.norm.rvs(mu,sigma,size=(NR_TSTEPS,));
    # 2. Use cumsum to compute absolute position from delta_pos
    pos = pl.cumsum(delta_pos);   #+z;
    # 3. Find the index where the position first crosses a boundary (i.e., 1 or -1)
    cross_indices = pl.find(abs(pos)>=bound);
    if len(cross_indices):
        cross_idx = cross_indices[0]; # take the first index
    else: # i.e., if no crossing was found
        cross_idx = NR_TSTEPS-1; # set it to the final index (i.e., maximum time)
    # 4. Now we can use this index to determine both the decision and the response time
    decision = pos[cross_idx]>0;
    resp_time = t[cross_idx];
    if bound[cross_idx]>a2:
        confidence =2;
    elif bound[cross_idx]>a1:
        confidence=1;
    else:
        confidence=0;
    return array([confidence,decision, resp_time]);

observed = pl.array([generate_sample(0.5,0.3,0.3,1.4,0.9) for i in range(40)]);
observed_data = array(sorted(observed, key = itemgetter(0))).T;


def compute_likelihood(parameters,value=observed_data):
    r = parameters[0];
    D = parameters[1];
    L = parameters[2];
    a2 = parameters[3];
    a1 = parameters[4];
    # I'm assuming that data is a 3xN array with the first row indicating the confidence,
    # second row indicating decisions and the third row indicating the RT's for each trial
    confidence,decisions,rts = value;
    input_confidence = array(confidence);
    input_decisions = array(decisions);
    input_rts = array(rts);
    
    # convert input_decisions to proper boolean types
    input_decisions = input_decisions.astype(bool);
    
    input_decisions_low = input_decisions[input_confidence==0];
    input_decisions_medium = input_decisions[input_confidence==1];
    input_decisions_high = input_decisions[input_confidence==2];
    input_rts_low = input_rts[input_confidence==0];
    input_rts_medium = input_rts[input_confidence==1];
    input_rts_high = input_rts[input_confidence==2];
    
    #return input_decisions_low
    
    # compute process SD
    sigma = sqrt(2*D*DELTA_T);
    
    t = linspace(DELTA_T,MAX_T,NR_TSTEPS);
    bound = 2.0*exp(-L*t);
    # compute mean drift over a single time interval
    mu = r*DELTA_T;
    # Now simulate NR_SAMPLES trials
    # 1. Generate a random position change for each time interval
    #   these position changes should be drawn from a normal distribution with mean
    #   mu and standard deviation sigma
    delta_pos = stats.norm.rvs(mu,sigma,size=(NR_SAMPLES,NR_TSTEPS));
    # 2. Use cumsum to compute absolute positions from delta_pos
    positions = pl.cumsum(delta_pos,1);   #+z;
    # 3. Now loop through each sample trial to compute decisions and resp times
    decisions = [];
    resp_times = [];
    confidences = [];
    for pos in positions:
        # Find the index where the position first crosses a boundary (i.e., 1 or -1)
        cross_indices = pl.find(abs(pos)>=bound);
        if len(cross_indices):
            cross_idx = cross_indices[0]; # take the first index
        else: # i.e., if no crossing was found
            cross_idx = NR_TSTEPS-1; # set it to the final index
        # 4. Now we can use this index to determine both the decision and the response time
        decision = pos[cross_idx]>0;
        resp_time = t[cross_idx]-0.5*DELTA_T; #i.e., the midpoint of the crossing interval
        if bound[cross_idx]>a2:
            confidence=2;
        elif bound[cross_idx]>a1:
            confidence=1;
        else:
            confidence=0;
        decisions.append(decision);
        resp_times.append(resp_time);
        confidences.append(confidence);
            
    # Now estimate the joint distribution    
    decisions = array(decisions);
    resp_times = array(resp_times);
    confidences = array(confidences);
    

    # 1. fit gamma functions to the response
    resp_times_no_high = resp_times[logical_and(logical_not(decisions),confidences==2)];
    resp_times_yes_high = resp_times[logical_and(decisions,confidences==2)];
    
    resp_times_no_medium = resp_times[logical_and(logical_not(decisions),confidences==1)];
    resp_times_yes_medium = resp_times[logical_and(decisions,confidences==1)];
   
    resp_times_no_low = resp_times[logical_and(logical_not(decisions),confidences==0)];
    resp_times_yes_low = resp_times[logical_and(decisions,confidences==0)];
    
    params_no_high = stats.gamma.fit(resp_times_no_high,floc=0);
    params_yes_high = stats.gamma.fit(resp_times_yes_high,floc=0);
    
    params_no_medium = stats.gamma.fit(resp_times_no_medium,floc=0);
    params_yes_medium = stats.gamma.fit(resp_times_yes_medium,floc=0);
    
    params_no_low = stats.gamma.fit(resp_times_no_low,floc=0);
    params_yes_low = stats.gamma.fit(resp_times_yes_low,floc=0);
    
    # compute the probability of making an "old" decision
    p_yes_high = logical_and(decisions,confidences==2).mean();
    p_yes_medium = logical_and(decisions,confidences==1).mean();
    p_yes_low = logical_and(decisions,confidences==0).mean();
    
    log_likelihood_high = sum(log(p_yes_high)+stats.gamma.logpdf(input_rts_high[input_decisions_high],*params_yes_high)) + sum(log(1-p_yes_high)+stats.gamma.logpdf(input_rts_high[pl.logical_not(input_decisions_high)],*params_no_high));

    log_likelihood_medium = sum(log(p_yes_medium)+stats.gamma.logpdf(input_rts_medium[input_decisions_medium],*params_yes_medium)) + sum(log(1-p_yes_medium)+stats.gamma.logpdf(input_rts_medium[pl.logical_not(input_decisions_medium)],*params_no_medium));
    
    log_likelihood_low = sum(log(p_yes_low)+stats.gamma.logpdf(input_rts_low[input_decisions_low],*params_yes_low)) + sum(log(1-p_yes_low)+stats.gamma.logpdf(input_rts_low[pl.logical_not(input_decisions_low)],*params_no_low));
    
    log_likelihood = log_likelihood_high+log_likelihood_low+log_likelihood_medium

    if pl.isnan(log_likelihood):
        log_likelihood = -pl.inf;
    return log_likelihood;
