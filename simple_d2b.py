import pylab as pl
from scipy import stats
from pylab import array, linspace, arange, sign, sqrt, log, sum

# Constants
DELTA_T     = 0.1;  # size of discrete time increment (sec.)
MAX_T       = 60.0; #(i.e., 1 minute)
NR_TSTEPS  = MAX_T/DELTA_T;
NR_SAMPLES  = 1000; # number of trials to use for MC likelihood computation

# D is diffusion constant
# r is drift rate
# z is starting point

def generate_sample(r,D,z=0):
    # compute process SD
    sigma = sqrt(2*D*DELTA_T);
    
    t = linspace(DELTA_T,MAX_T,NR_TSTEPS);
    # compute mean drift over a single time interval
    mu = r*DELTA_T;
    # Now simulate an individual trial
    # 1. Generate a random position change for each time interval
    #   these position changes should be drawn from a normal distribution with mean
    #   mu and standard deviation sigma
    delta_pos = stats.norm.rvs(mu,sigma,size=(NR_TSTEPS,));
    # 2. Use cumsum to compute absolute position from delta_pos
    pos = pl.cumsum(delta_pos)+z;
    # 3. Find the index where the position first crosses a boundary (i.e., 1 or -1)
    cross_indices = pl.find(abs(pos)>=1);
    if len(cross_indices):
        cross_idx = cross_indices[0]; # take the first index
    else: # i.e., if no crossing was found
        cross_idx = NR_TSTEPS-1; # set it to the final index (i.e., maximum time)
    # 4. Now we can use this index to determine both the decision and the response time
    decision = pos[cross_idx]>0;
    resp_time = t[cross_idx];
    return array([decision, resp_time])#,pos,t;

def compute_likelihood(value,r,D,z):
    # I'm assuming that data is a 2xN array with the first row indicating the
    # decisions and the second row indicating the RT's for each trial
    input_decisions, input_rts = value;
    # convert input_decisions to proper boolean types
    input_decisions = array(input_decisions.astype(bool),ndmin=1);
    input_rts = array(input_rts,ndmin=1);
    # compute process SD
    sigma = sqrt(2*D*DELTA_T);
    
    t = linspace(DELTA_T,MAX_T,NR_TSTEPS);
    # compute mean drift over a single time interval
    mu = r*DELTA_T;
    # Now simulate NR_SAMPLES trials
    # 1. Generate a random position change for each time interval
    #   these position changes should be drawn from a normal distribution with mean
    #   mu and standard deviation sigma
    delta_pos = stats.norm.rvs(mu,sigma,size=(NR_SAMPLES,NR_TSTEPS));
    # 2. Use cumsum to compute absolute positions from delta_pos
    positions = pl.cumsum(delta_pos,1)+z;
    # 3. Now loop through each sample trial to compute decisions and resp times
    decisions = [];
    resp_times = [];
    for pos in positions:
        # Find the index where the position first crosses a boundary (i.e., 1 or -1)
        cross_indices = pl.find(abs(pos)>=1);
        if len(cross_indices):
            cross_idx = cross_indices[0]; # take the first index
        else: # i.e., if no crossing was found
            cross_idx = NR_TSTEPS-1; # set it to the final index
        # 4. Now we can use this index to determine both the decision and the response time
        decision = pos[cross_idx]>0;
        resp_time = t[cross_idx]-0.5*DELTA_T; #i.e., the midpoint of the crossing interval
        decisions.append(decision);
        resp_times.append(resp_time);
    # Now estimate the joint distribution
    decisions = array(decisions);
    resp_times = array(resp_times);
    # 1. fit gamma functions to the response
    resp_times_no = resp_times[pl.logical_not(decisions)];
    resp_times_yes = resp_times[decisions];
    
    params_no = stats.lognorm.fit(resp_times_no);
    params_yes = stats.lognorm.fit(resp_times_yes);
    
    # compute the probability of making an "old" decision
    p_yes = decisions.mean();
    
    log_likelihood = sum(log(p_yes)+stats.lognorm.logpdf(input_rts[input_decisions],*params_yes)) + sum(log(1-p_yes)+stats.lognorm.logpdf(input_rts[pl.logical_not(input_decisions)],*params_no));
    
    if pl.isnan(log_likelihood):
        log_likelihood = -pl.inf;
    return log_likelihood;