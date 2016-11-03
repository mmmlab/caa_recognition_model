import shelve
import pylab as pl
from pylab import fft,ifft,fftshift,ifftshift
from scipy import stats
from scipy import optimize
import fftw_test as fftw
from multinomial_funcs import multinom_loglike,chi_square_gof

# set a few global matplotlib plotting parameters
pl.rcParams['legend.frameon'] = 'False'
pl.rcParams['font.family'] = 'Arial'
pl.rcParams['font.size'] = 16.0

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
DELTA_T     = 0.01;# 0.05;  # size of discrete time increment (sec.)
MAX_T       = 8.0; #ceil(percentile(all_RT,99.5))
NR_TSTEPS   = MAX_T/DELTA_T; # number of steps along the temporal axis
NR_SSTEPS   = 8192; # number of steps along the spatial axis
NR_SAMPLES  = 10000; # number of trials to use for MC likelihood computation

# previously fitted parameters and bounds
# version with single diffusion parameter and lowest confidence bound fixed at zero
params_est_old = [0.9169,0.319,0.3888,-0.265,0.0505,-0.1198,0.4968,0.5799]; # fitted w/ 10 quantiles, chisq = 606
params_est = [0.9452,0.3236,0.4126,-0.2745,0.0486,-0.124,0.5001,0.5527]; # fitted w/ 10 quantiles, chisq = 440

params_est = [0.8,0.3236,0.1,-0.2745,0.1,-0.124,0.5001,0.5527]; # demo values

# This is a Monte-Carlo simulation approach to approximating the same RT distributions
# that are computed in predicted_proportions (above). I include it here as a sanity check
# and as a more accessible description of what the code above is doing.
def plot_schematic(model_params):
    # unpack the model parameters
    conf,mu_old,d,mu_new,tc_bound,z0,deltaT,t_offset = model_params;
    # make c (the confidence levels) an array in case it is a scalar value
    c = pl.array([conf,0]);
    n = len(c);
    # form an array consisting of the appropriate (upper) integration limits
    clims = pl.hstack(([INF_PROXY],c,[-INF_PROXY]));
    # compute process SD
    sigma_f = pl.sqrt(2*d*DELTA_T);
    sigma = sigma_f;
    
    t = pl.linspace(DELTA_T,MAX_T,NR_TSTEPS); # this is the time axis
    to_idx = pl.argmin((t-t_offset)**2); # compute the index for t_offset
    bound = pl.exp(-tc_bound*pl.clip(t-t_offset,0,None)); # this is the collapsing bound
    # Now simulate NR_SAMPLES trials
    # 1. Generate a random position change for each time interval
    #   these position changes should be drawn from a normal distribution with mean
    #   mu and standard deviation sigma
    delta_pos = stats.norm.rvs(mu_old*DELTA_T,sigma,size=NR_TSTEPS);
    # for timesteps before t_offset, we're not accumulating any information
    # therefore set the delta_pos for these timesteps to 0
    delta_pos[:to_idx] = 0;
    # 2. Use cumsum to compute absolute positions from delta_pos
    positions = pl.cumsum(delta_pos)+z0;
    # Find the index where the position first crosses a boundary (i.e., 1 or -1)
    cross_indices = pl.find(abs(positions)>=bound);
    if len(cross_indices):
        cross_idx = cross_indices[0]; # take the first index
    else: # i.e., if no crossing was found
        cross_idx = NR_TSTEPS-1; # set it to the final index
    # 4. Now we can use this index to determine both the decision and the response time
    decisions = positions[cross_idx]>0;
    resp_time = t[cross_idx]-0.5*DELTA_T; #i.e., the midpoint of the crossing interval
    cross_pos = positions[cross_idx];
    
    final_idx = min(int(deltaT/DELTA_T)+cross_idx,NR_TSTEPS-1);
    final_pos = positions[final_idx];
    final_time = t[final_idx];
    
    # now do the same thing for a new word trial
    delta_pos_n = stats.norm.rvs(mu_new*DELTA_T,sigma,size=NR_TSTEPS);
    # for timesteps before t_offset, we're not accumulating any information
    # therefore set the delta_pos for these timesteps to 0
    delta_pos_n[:to_idx] = 0;
    # 2. Use cumsum to compute absolute positions from delta_pos
    positions_n = pl.cumsum(delta_pos_n)+z0;
    # Find the index where the position first crosses a boundary (i.e., 1 or -1)
    cross_indices = pl.find(abs(positions_n)>=bound);
    if len(cross_indices):
        cross_idx_n = cross_indices[0]; # take the first index
    else: # i.e., if no crossing was found
        cross_idx_n = NR_TSTEPS-1; # set it to the final index
    # 4. Now we can use this index to determine both the decision and the response time
    decisions_n = positions_n[cross_idx_n]>0;
    resp_time_n = t[cross_idx_n]-0.5*DELTA_T; #i.e., the midpoint of the crossing interval
    cross_pos_n = positions[cross_idx_n];
    
    final_idx_n = min(int(deltaT/DELTA_T)+cross_idx_n,NR_TSTEPS-1);
    final_pos_n = positions_n[final_idx_n];
    final_time_n= t[final_idx_n];
        
    # compute the distribution of particle positions deltaT seconds after the
    # old/new decision and choose a random position from the resulting
    # distribution to add to the position at decision
    # the parameters below are for the post old/new decision interval
    #sigma_deltaT = pl.sqrt(2*d*deltaT);
    #pos_deltaT = stats.norm.rvs(mu_old*deltaT,sigma_deltaT);
    #final_pos = cross_pos+pos_deltaT;
    # compute the rate trendline for old words
    pos_avg = t*mu_old+z0;
    pos_avg_n = t*mu_new+z0;
    # compute SD as a function of time
    pos_sd = pl.sqrt(2*d*t);
    
    # now do the plotting
    pl.figure();
    # 1. plot the collapsing bounds
    pl.plot(t,bound,'k-',t,-bound,'k-',lw=2);
    # 2. plot the confidence bounds
    pl.hlines(c,t.min(),t.max(),linestyles='dashed',color='0.5',lw=2);
    # 3. plot the rate trendline(s)
    pl.plot(t,pos_avg,'g-',lw=2,alpha=0.3);
    pl.plot(t,pos_avg_n,'r-',lw=2,alpha=0.3);
    # 4. plot the sampled positions
    pl.plot(t[:final_idx+1],positions[:final_idx+1],'g-',lw=2);
    pl.plot([resp_time,final_time],[cross_pos,final_pos],'go');
    # 5. plot the diffusion SD
    pl.fill_between(t,pos_avg-pos_sd,pos_avg+pos_sd,color='g',alpha=0.15);
    pl.fill_between(t,pos_avg_n-pos_sd,pos_avg_n+pos_sd,color='r',alpha=0.15);
    # 6. plot the region before evidence starts accumulating
    pl.vlines([t_offset],-1.5,1.5,linestyles='dotted',color='k');
    #pl.fill_between([0,t_offset],[-1.5,-1.5],[1.5,1.5],color='k',alpha=0.2);
    pl.axis([-0.1,8,-1.5,1.5]);
    pl.xlabel('Time (sec.)');
    pl.ylabel('Evidence');
    pl.show();
    
    filename = 'neha/d2b_schematic.png';
    pl.savefig(filename,dpi=100,bbox_inches='tight',pad_inches=0,transparent=True);
    pl.savefig(filename.replace('png','svg'),dpi=100,bbox_inches='tight',\
                            pad_inches=0,transparent=True);
    


