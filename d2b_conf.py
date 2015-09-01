from pylab import *
import pylab as pl
import numpy
from scipy import stats
from scipy import optimize
import pyfftw
import fftw_test as fftw
from multinomial_funcs import multinom_loglike,chi_square_gof

data_path = 'neha/data/'; # this is the base path for the data files

## Start by reading in the data.
# the reason to do this first is that, in order to be efficient,
# we don't want to represent any more of the time axis than we have to.

rem_hit = numpy.loadtxt(data_path+'remRT_hit.txt'); # load remember RTs for hits
know_hit = numpy.loadtxt(data_path+'knowRT_hit.txt'); # load know RTs for hits
rem_fa = numpy.loadtxt(data_path+'remRT_fa.txt'); # load remember RTs for false alarms
know_fa = numpy.loadtxt(data_path+'knowRT_fa.txt');  # load know RTs for false alarms
CR = numpy.loadtxt(data_path+'CR.txt');  # load CR RTs 
miss = numpy.loadtxt(data_path+'miss.txt');  # load miss RTs



## read in collapsed data 
#remember_hit = numpy.loadtxt('data_collapsed/remRT_hit.txt'); # load remember RTs for hits
#know_hit = numpy.loadtxt('data_collapsed/knowRT_hit.txt'); # load know RTs for hits
#remember_fa = numpy.loadtxt('data_collapsed/remRT_fa.txt'); # load remember RTs for false alarms#
#know_fa = numpy.loadtxt('data_collapsed/knowRT_fa.txt');  # load know RTs for false alarms
#CR = numpy.loadtxt('data_collapsed/CR.txt');  # load CR RTs 
#miss = numpy.loadtxt('data_collapsed/miss.txt');  # load miss RTs 
INF_PROXY   = 10; # a value used to provide very large but finite bounds for mvn integration
EPS         = 1e-10 # a very small value (used for numerical stability)
NR_THREADS  = 1;    # this is for multithreaded fft
DELTA_T     = 0.05;  # size of discrete time increment (sec.)
MAX_T       = 24; #ceil(percentile(all_RT,99.5))
NR_TSTEPS   = MAX_T/DELTA_T;
NR_SSTEPS   = 4096; #8192#4096#2048;
NR_SAMPLES  = 10000; # number of trials to use for MC likelihood computation
n = 2; # number of confidence critetion
QUANT = array([0,0.25,0.50,0.75]);
QUANT_DIFF = 0.25;
NR_QUANTILES = 4;
R = 0.1; D = 0.05; L = 0.1; Z = 0.0;

#param_bounds = [(0.0,1.0),(0.0,1.0),(EPS,1.0),(EPS,1.0),(0.05,1.0),(0.0,1.0),(-1.0,1.0),(-1.0,1.0),(-1.0,1.0)];
#param_bounds = [(0.51,0.95),(0.1,0.5),(0.0,1.0),(0.0,1.0),(0.05,1.0),(0.05,1,0),(0.1,1.0),(0.0,1.0),(-1.0,1.0),(-1.0,1.0),(-1.0,1.0)];
#param_bounds = [(0.2,1.0),(0.0,0.19),(0.0,1.0),(0.0,1.0),(EPS,1.0),(EPS,1.0),(0.0,1.0),(0.0,1.0),(-1.0,1.0),(-1.0,1.0),(-1.0,1.0),(0.0,10.0)];
#param_bounds = [(0.2,1.0),(0.01,0.19),(0.0,1.0),(0.0,1.0),(EPS,1.0),(EPS,1.0),(0.0,10.0)];


fftw.fftw_setup(zeros(NR_SSTEPS),NR_THREADS);

params_est = [(0.7233685,0.08959858),0.50357936,0.96055488,0.29500501,0.00464397,1.0,0.67084697,-0.74701904,0.22969753];

params_est2 = [  9.96964984e-01,   7.38002355e-04,   1.83856733e-01, 3.52048480e-06,   1.54057350e-01, 4.04694786e-01,5.56870264e-04,  -1.19917001e-01,   9.55494426e-02]

#params_all_est = [1.0,.1,.1,.15,.15,.3,.5,-0.1,0.,.1,.1]
params_all_est = [0.7233685,0.50357936,0.96055488,0.29500501,0.00464397,1.0,0.67084697,-0.74701904,0.50357936,0.50357936,0.22969753];

params_all_est2 = loadtxt('neha/ml_params_all.txt');
#[1.,0.001,  0.137, EPS, 0.153, 0.353, 0.001, -0.07 ,EPS , -0.162, 0.131];

param_bounds = [(0.0,1.0),(-1.0,1.0),(-1.0,1.0),(EPS,1.0),(EPS,1.0),(0.05,1.0),(0.0,1.0),(-1.0,1.0),(EPS,2.0)];
# c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,delta_t = params;

def find_ml_params_all(quantiles=2):
    # computes mle of params using a global (slow) and bounded optimization algorithm
    def obj_func(model_params):
        c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,mu_r0,mu_f0,deltaT = model_params;
        params_est_old = [c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT];
        params_est_new = [c,mu_r0,mu_f0,d_r,d_f,tc_bound,r_bound,z0,deltaT];
        old_data = [rem_hit[:,0],know_hit[:,0],miss[:,0],rem_hit[:,1],know_hit[:,1]];
        new_data = [rem_fa[:,0],know_fa[:,0],CR[:,0],rem_fa[:,1],know_fa[:,1]];
        res = compute_model_gof(params_est_old,*old_data,nr_quantiles=quantiles)+ \
        compute_model_gof(params_est_new,*new_data,nr_quantiles=quantiles);
        return res;
    param_bounds = [(0.0,1.0),(-1.0,1.0),(-1.0,1.0),(EPS,1.0),(EPS,1.0),(0.05,1.0),(0.0,1.0),(-1.0,1.0),(-1.0,1.0),(-1.0,1.0),(EPS,2.0)];
    return optimize.differential_evolution(obj_func,param_bounds)

def find_ml_params_all_mdf(quantiles=2):
    # computes mle of params using as few parameters as possible
    # at the moment, that means freezing mu_f0 and mu_r0 at 0
    def obj_func(model_params):
        c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT = model_params;
        mu_f0 = mu_r0 =0;
        params_est_old = [c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT];
        params_est_new = [c,mu_r0,mu_f0,d_r,d_f,tc_bound,r_bound,z0,deltaT];
        old_data = [rem_hit[:,0],know_hit[:,0],miss[:,0],rem_hit[:,1],know_hit[:,1]];
        new_data = [rem_fa[:,0],know_fa[:,0],CR[:,0],rem_fa[:,1],know_fa[:,1]];
        res = compute_model_gof(params_est_old,*old_data,nr_quantiles=quantiles)+ \
        compute_model_gof(params_est_new,*new_data,nr_quantiles=quantiles);
        return res;
    param_bounds = [(0.0,1.0),(-1.0,1.0),(-1.0,1.0),(EPS,1.0),(EPS,1.0),(0.05,1.0),(0.0,1.0),(-1.0,1.0),(EPS,2.0)];
    return optimize.differential_evolution(obj_func,param_bounds)

def find_ml_params_all_lm(quantiles=2):
    # computes mle of params using a local (fast) optimization algorithm
    def obj_func(model_params):
        c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,mu_r0,mu_f0,deltaT = model_params;
        params_est_old = [c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT];
        params_est_new = [c,mu_r0,mu_f0,d_r,d_f,tc_bound,r_bound,z0,deltaT];
        old_data = [rem_hit[:,0],know_hit[:,0],miss[:,0],rem_hit[:,1],know_hit[:,1]];
        new_data = [rem_fa[:,0],know_fa[:,0],CR[:,0],rem_fa[:,1],know_fa[:,1]];
        res = compute_model_gof(params_est_old,*old_data,nr_quantiles=quantiles)+ \
        compute_model_gof(params_est_new,*new_data,nr_quantiles=quantiles);
        return res;
    return optimize.fmin(obj_func,params_all_est2)

def find_ml_params():
    obj_func = lambda model_params:compute_model_gof(model_params,rem_hit[:,0],know_hit[:,0],miss[:,0],rem_hit[:,1],know_hit[:,1],nr_quantiles=3);
    return optimize.differential_evolution(obj_func,param_bounds);

def find_ml_params2():
    obj_func = lambda model_params:compute_model_gof(model_params,rem_hit[:,0],know_hit[:,0],miss[:,0],rem_hit[:,1],know_hit[:,1],nr_quantiles=3);
    return optimize.fmin(obj_func,params_est2);

def find_ml_params_new():
    obj_func = lambda model_params:compute_model_gof(model_params,rem_fa[:,0],know_fa[:,0],CR[:,0],rem_fa[:,1],know_fa[:,1],nr_quantiles=3);
    return optimize.fmin(obj_func,params_est2);

def compute_nll_conf(remH,knowH,missH,remFA,knowFA,crFA,rem_quantiles_old,know_quantiles_old,rem_quantiles_new,know_quantiles_new,data_old,data_new):
    
    old = len(remH)+len(knowH)+len(missH);
    new = len(remFA)+len(knowFA)+len(crFA);
    # print old;

    rem_freqs_old = -diff([sum(remH>q) for q in rem_quantiles_old]+[0]);
    know_freqs_old = -diff([sum(knowH>q) for q in know_quantiles_old]+[0]);
    x_old = hstack([rem_freqs_old,know_freqs_old]);
    
    p_rem = data_old[0]*ones(NR_QUANTILES)/float(NR_QUANTILES);
    p_know = data_old[1]*ones(NR_QUANTILES)/float(NR_QUANTILES);
    p_old = hstack([p_rem,p_know]);
    nll_old = -multinom_loglike(x_old,old,p_old)
    
    
    rem_freqs_new = -diff([sum(remFA>q) for q in rem_quantiles_new]+[0]);
    know_freqs_new = -diff([sum(knowFA>q) for q in know_quantiles_new]+[0]);
    x_new = hstack([rem_freqs_new,know_freqs_new]);
    
    p_rem_n = data_new[0]*ones(NR_QUANTILES)/float(NR_QUANTILES);
    p_know_n = data_new[1]*ones(NR_QUANTILES)/float(NR_QUANTILES);
    p_new = hstack([p_rem_n,p_know_n]);
    nll_new = -multinom_loglike(x_new,new,p_new)
    nll_conf = nll_old+nll_new;
    return nll_conf;

def compute_chi_conf(remH,knowH,missH,remFA,knowFA,crFA,rem_quantiles_old,know_quantiles_old,rem_quantiles_new,know_quantiles_new,data_old,data_new):
    
    old = len(remH)+len(knowH)+len(missH);
    new = len(remFA)+len(knowFA)+len(crFA);
    #print old;

    rem_freqs_old = -diff([sum(remH>q) for q in rem_quantiles_old]+[0]);
    know_freqs_old = -diff([sum(knowH>q) for q in know_quantiles_old]+[0]);
    x_old = hstack([rem_freqs_old,know_freqs_old]); 
    
    p_rem = data_old[0]*ones(NR_QUANTILES)/float(NR_QUANTILES);
    p_know = data_old[1]*ones(NR_QUANTILES)/float(NR_QUANTILES);
    p_old = hstack([p_rem,p_know]); 
    chi_old = chi_square_gof(x_old,old,p_old)
    
    
    rem_freqs_new = -diff([sum(remFA>q) for q in rem_quantiles_new]+[0]);
    know_freqs_new = -diff([sum(knowFA>q) for q in know_quantiles_new]+[0]);
    x_new = hstack([rem_freqs_new,know_freqs_new]); 
    
    p_rem_n = data_new[0]*ones(NR_QUANTILES)/float(NR_QUANTILES);
    p_know_n = data_new[1]*ones(NR_QUANTILES)/float(NR_QUANTILES);
    p_new = hstack([p_rem_n,p_know_n]);
    chi_new = chi_square_gof(x_new,new,p_new)
    chi_conf = chi_old+chi_new;
    return chi_conf;

def compute_full_model_gof(full_model_params,old_data,new_data):
    c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT = full_model_params;
    mu_f0 = mu_r0 =0;
    params_est_old = [c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT];
    params_est_new = [c,mu_r0,mu_f0,d_r,d_f,tc_bound,r_bound,z0,deltaT];
    old_data = [rem_hit[:,0],know_hit[:,0],miss[:,0],rem_hit[:,1],know_hit[:,1]];
    new_data = [rem_fa[:,0],know_fa[:,0],CR[:,0],rem_fa[:,1],know_fa[:,1]];
    res = compute_model_gof(params_est_old,*old_data,nr_quantiles=quantiles)+ \
    compute_model_gof(params_est_new,*new_data,nr_quantiles=quantiles);
    return res;


def compute_model_gof(model_params,rem_RTs,know_RTs,new_RTs,rem_conf,know_conf,nr_quantiles=4):
    # computes the chi square fit of the model to the data
    # compute N, the total number of trials
    N = len(rem_RTs)+len(know_RTs)+len(new_RTs);
    # compute x, the observed frequency for each category
    rem_quantiles,know_quantiles,new_quantiles,p_r,p_k,p_n = compute_model_quantiles(model_params,nr_quantiles);
    # determine the number of confidence levels being used in the model
    nr_conf_levels = len(rem_quantiles);
    # adjust the number of confidence levels in the data to match
    rem_conf = clip(rem_conf,0,nr_conf_levels-1);
    know_conf = clip(know_conf,0,nr_conf_levels-1);
    ## compute the number of RTs falling into each quantile bin
    rem_freqs = array([-diff([sum(rem_RTs[rem_conf==i]>q) for q in rem_quantiles[i]]+[0]) for i in range(nr_conf_levels)]);
    know_freqs = array([-diff([sum(know_RTs[know_conf==i]>q) for q in know_quantiles[i]]+[0]) for i in range(nr_conf_levels)]);
    new_freqs = -diff([sum(new_RTs>q) for q in new_quantiles]+[0]);
    x = hstack([rem_freqs.flatten(),know_freqs.flatten(),new_freqs]);
    
    # compute p, the probability associated with each category in the model
    p_rem = p_r[:,newaxis]*ones((nr_conf_levels,nr_quantiles))/float(nr_quantiles);
    p_know = p_k[:,newaxis]*ones((nr_conf_levels,nr_quantiles))/float(nr_quantiles);
    p_new = p_n*ones(nr_quantiles)/float(nr_quantiles);
    p = hstack([p_rem.flatten(),p_know.flatten(),p_new]);
    return chi_square_gof(x,N,p)

def compute_model_quantiles(params,nr_quantiles=4):
    # This function is set up to deal with multiple confidence levels
    quantile_increment = 1.0/nr_quantiles;
    quantiles = arange(0,1,quantile_increment);
    # unpack model parameters
    # c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,delta_t = params;
    # compute marginal distributions
    p_remember,p_know,p_new,t = predicted_proportions(*params);
    # compute marginal category proportions (per confidence level)
    remember_total = p_remember.sum(-1)+EPS;
    know_total = p_know.sum(-1)+EPS;
    new_total = p_new.sum()+EPS;
    # compute integrals of marginal distributions
    P_r = cumsum(p_remember,-1)/remember_total[:,newaxis];
    P_k = cumsum(p_know,-1)/know_total[:,newaxis];
    P_n = cumsum(p_new)/new_total;
    
    # compute RT quantiles (by confidence level for know and rem judgments)
    rem_quantiles = array([t[argmax(P_r>q,-1)] for q in quantiles]).T;
    know_quantiles = array([t[argmax(P_k>q,-1)] for q in quantiles]).T;
    new_quantiles = array([t[argmax(P_n>q)] for q in quantiles]);
    rem_quantiles[:,0] = 0;
    know_quantiles[:,0] = 0;
    new_quantiles[0] = 0;
    # return quantile locations and marginal p(new)
    return rem_quantiles,know_quantiles,new_quantiles,sum(p_remember,1),sum(p_know,1),sum(p_new);

# def compute_model_quantiles(params,nr_quantiles=4):
#     quantile_increment = 1.0/nr_quantiles;
#     quantiles = arange(0,1,quantile_increment);
#     # unpack model parameters
#     # c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,delta_t = params;
#     # compute marginal distributions
#     p_remember,p_know,p_new,t = predicted_proportions(*params);
#     # compute marginal category proportions
#     remember_total = p_remember.sum()+EPS;
#     know_total = p_know.sum()+EPS;
#     new_total = p_new.sum()+EPS;
#     # compute integrals of marginal distributions
#     P_r = cumsum(p_remember)/remember_total;
#     P_k = cumsum(p_know)/know_total;
#     P_n = cumsum(p_new)/new_total;
#     
#     # compute RT quantiles
#     rem_quantiles = array([t[argmax(P_r>q)] for q in quantiles]);
#     know_quantiles = array([t[argmax(P_k>q)] for q in quantiles]);
#     new_quantiles = array([t[argmax(P_n>q)] for q in quantiles]);
#     rem_quantiles[0] = 0;
#     know_quantiles[0] = 0;
#     new_quantiles[0] = 0;
#     # return quantile locations and marginal p(new)
#     return rem_quantiles,know_quantiles,new_quantiles,sum(p_remember),sum(p_know),sum(p_new);

def predicted_proportions(c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT,use_fftw=True):
    # make c (the confidence levels) an array in case it is a scalar value
    c = array(c,ndmin=1);
    n = len(c);
    # form an array consisting of the appropriate (upper) integration limits
    clims = hstack(([INF_PROXY],c,[-INF_PROXY]));
    # compute process SD
    sigma_r = sqrt(2*d_r*DELTA_T);
    sigma_f = sqrt(2*d_f*DELTA_T);
    sigma = sqrt(sigma_r**2+sigma_f**2);

    # compute the correlation for r given r+f
    rho = sigma_r/sigma;

    t = linspace(DELTA_T,MAX_T,NR_TSTEPS); # this is the time axis
    bound = exp(-tc_bound*t); # this is the collapsing bound
     
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
    
    # Construct arrays to hold RT distributions
    p_old = zeros(shape(t));
    p_new = zeros(shape(t));
    p_rem_conf = zeros((n+1,size(t))); 
    p_know_conf = zeros((n+1,size(t)));
    
    
    ############################################
    ## take care of the first timestep #########
    ############################################
    tx[0] = stats.norm.pdf(x,mu+z0,sigma)*delta_s;
    p_old[0] = sum(tx[0][x>=bound[0]]);
    p_new[0] = sum(tx[0][x<=-bound[0]]);
    # compute STD(r) for the current time
    s_r = sigma_r;
    s_f = sigma_f;
    # compute STD(r+f) for the current time
    s_comb = sigma;
    # compute E[r|(r+f)]
    #mu_r_cond = mu_r*t[0]+rho*s_r*(bound[0]-t[0]*(mu_r+mu_f))/s_comb;
    mu_r_cond = mu_r*t[0]+(bound[0]-t[0]*(mu_r+mu_f)-z0)*rho**2;
    # compute STD[r|(r+f)]
    s_r_cond = s_r*sqrt(1-rho**2);
    s_f_cond = s_f*sqrt(1-(sigma_f/sigma)**2);
    
    # remove from consideration any particles that already hit the bound
    tx[0]*=(abs(x)<bound[0]);
    
    ############################################################################
    # compute the parameters of the bivariate distribution of particle locations
    # deltaT seconds after old/new decision
    
    mu_r_delta = mu_r_cond+mu_r*deltaT;
    mu_comb_delta = (mu_r+mu_f)*deltaT+bound[0];
    s2_r_delta = s_r_cond**2+2*d_r*deltaT;
    s2_f_delta = s_f_cond**2+2*d_f*deltaT;
    s2_comb_delta = s2_r_delta+s2_f_delta;
    #s2_comb_delta = 2*deltaT*(d_r+d_f);
    #s2_comb_delta = s_r_cond**2+s_f_cond**2+2*deltaT*(d_r+d_f);
    cov_delta = s2_r_delta;
    mu_mvn = array([mu_r_delta,mu_comb_delta]);
    sigma_mvn = array([[s2_r_delta,cov_delta],[cov_delta,s2_comb_delta]]);
    ############################################################################
    for j in range(1,len(clims)):
        # Note that the clims appear in descending order, from highest to lowest value
        KLL = array([-INF_PROXY,clims[j]]);     # lower limit for 'know' class
        KUL = array([r_bound,clims[j-1]]);      # upper limit for 'know' class
        RLL = array([r_bound,clims[j]]);        # lower limit for 'remember' class
        RUL = array([INF_PROXY,clims[j-1]]);    # upper limit for 'remember' class
        p_know_conf[j-1,0] = p_old[0]*stats.mvn.mvnun(KLL,KUL,mu_mvn,sigma_mvn)[0];
        p_rem_conf[j-1,0] = p_old[0]*stats.mvn.mvnun(RLL,RUL,mu_mvn,sigma_mvn)[0];
        
    #######################################
    ## take care of subsequent timesteps ##
    #######################################
    
    for i in range(1,len(t)):
        #tx[i] = convolve(tx[i-1],kernel,'same');
        # convolve the particle distribution from the previous timestep
        # with the diffusion kernel (using Fourier domain convolution)
        if(use_fftw):
            tx[i] = abs(ifftshift(fftw.ifft(fftw.fft(tx[i-1])*ft_kernel)));
        else:
            tx[i] = abs(ifftshift(ifft(fft(tx[i-1])*ft_kernel)));

        p_pos = tx[i][x>=bound[i]]; # probability of each particle position above the upper bound
        x_pos = x[x>=bound[i]];     # location of each particle position above the upper bound
        
        # compute the expected value of a particle that just exceeded the bound
        # during the last time interval
        comb_est = (dot(p_pos,x_pos)+EPS)/(sum(p_pos)+EPS); 

        p_old[i] = sum(p_pos); # total probability that particle crosses upper bound
        p_new[i] = sum(tx[i][x<=-bound[i]]); # probability that particle crosses lower bound
        
        # compute STD(r) for the current time
        s_r = sqrt(2*d_r*t[i]);
        s_f = sqrt(2*d_f*t[i]);
        # compute STD[r|(r+f)]
        s_r_cond = s_r*sqrt(1-rho**2);
        s_f_cond = s_f*sqrt(1-(sigma_f/sigma)**2);
        # compute E[r|(r+f)]
        mu_r_cond = mu_r*t[i]+(comb_est-t[i]*(mu_r+mu_f)-z0)*rho**2;
        # remove from consideration any particles that already hit the bound
        tx[i]*=(abs(x)<bound[i]);
        
        # (6/23/2015) New method for computing p_know, p_remember, and confidences
        # simultaneously using cumulative bivariate normal integral.
        # The idea is to:
        #   1. model the bivariate distribution of (r,r+f) particle locations 'deltaT'
        #   seconds after the old/new decision
        #   2. use the multivariate normal integral function (stats.mvn.mvnun) to compute
        #   the probability that the particle falls into any of the relevant regions
        #   defined by the constant "r" and "conf" bounds
        
        ########################################################################
        # compute the parameters of the bivariate distribution of particle
        # locations deltaT seconds after old/new decision
        mu_r_delta = mu_r_cond+mu_r*deltaT;
        mu_comb_delta = (mu_r+mu_f)*deltaT+comb_est;
        s2_r_delta = s_r_cond**2+2*d_r*deltaT;
        s2_f_delta = s_f_cond**2+2*d_f*deltaT;
        s2_comb_delta = s2_r_delta+s2_f_delta;
        #s2_comb_delta = 2*deltaT*(d_r+d_f);
        #s2_comb_delta = s_r_cond**2+s_f_cond**2+2*deltaT*(d_r+d_f);
        cov_delta = s2_r_delta;
        mu_mvn = array([mu_r_delta,mu_comb_delta]);
        sigma_mvn = array([[s2_r_delta,cov_delta],[cov_delta,s2_comb_delta]]);
        ########################################################################
        # Test Code:
        #if(t[i]>0.5):
            #slim = 3.0;
            #xaxis = linspace(-slim,slim,200);
            #xsup,ysup = meshgrid(xaxis,flipud(xaxis));
            #supp = vstack([xsup.flatten(),ysup.flatten()]).T;
            #z = stats.multivariate_normal.pdf(supp,mu_mvn,sigma_mvn);
            #figure(); imshow(reshape(z,shape(xsup)),cmap=cm.gray,extent=[-slim,slim,-slim,slim]);
            #vlines(r_bound,-slim,slim,colors='g');
            #hlines(c,-3,3,colors='r');
            #1/0
        ########################################################################
        for j in range(1,len(clims)):
            # remember that clims contains the bin edges in descending order
            KLL = array([-INF_PROXY,clims[j]]);
            KUL = array([r_bound,clims[j-1]]);
            RLL = array([r_bound,clims[j]]);
            RUL = array([INF_PROXY,clims[j-1]]);
            p_know_conf[j-1,i] = p_old[i]*stats.mvn.mvnun(KLL,KUL,mu_mvn,sigma_mvn)[0];
            p_rem_conf[j-1,i] = p_old[i]*stats.mvn.mvnun(RLL,RUL,mu_mvn,sigma_mvn)[0];
        
    # compute the marginal distributions for remember and know (i.e., across all confidence levels)
    p_remember = p_rem_conf.sum(0);
    p_know = p_know_conf.sum(0);
    return p_rem_conf,p_know_conf,p_new,t;

#def predicted_proportions_sim(mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0):
def predicted_proportions_sim(c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT):
    # make c (the confidence levels) an array in case it is a scalar value
    c = array(c,ndmin=1);
    n = len(c);
    # form an array consisting of the appropriate (upper) integration limits
    clims = hstack(([INF_PROXY],c,[-INF_PROXY]));
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
    decisions = zeros(NR_SAMPLES);
    resp_times = zeros(NR_SAMPLES);
    final_pos = zeros((NR_SAMPLES,2))
    
    for i, pos in enumerate(positions):
        # Find the index where the position first crosses a boundary (i.e., 1 or -1)
        cross_indices = pl.find(abs(pos)>=bound);
        if len(cross_indices):
            cross_idx = cross_indices[0]; # take the first index
        else: # i.e., if no crossing was found
            cross_idx = NR_TSTEPS-1; # set it to the final index
        # 4. Now we can use this index to determine both the decision and the response time
        decisions[i] = pos[cross_idx]>0;
        resp_times[i] = t[cross_idx]-0.5*DELTA_T; #i.e., the midpoint of the crossing interval
        final_pos[i] = [pos[cross_idx],r_positions[i,cross_idx]];
        
    # compute the distribution of particle positions deltaT seconds after the
    # old/new decision and choose a random position from the resulting
    # distribution to add to the position at decision
    # the parameters below are for the post old/new decision interval
    
    sigma_r_deltaT = sqrt(2*d_r*deltaT);
    sigma_f_deltaT = sqrt(2*d_f*deltaT);
    r_deltaTs = stats.norm.rvs(mu_r*deltaT,sigma_r_deltaT,size=NR_SAMPLES);
    pos_deltaTs = stats.norm.rvs(mu_f*deltaT,sigma_f_deltaT,size=NR_SAMPLES)+r_deltaTs;
    final_pos[:,0]+=pos_deltaTs; final_pos[:,1]+=r_deltaTs;
    
    # now make the remember decision on the basis of the final position in the
    # r dimension
    remembers = logical_and(decisions,final_pos[:,1]>r_bound);
    
    p_remember  = [];
    p_know = [];
    for j in range(1,len(clims)):
        # remember that the confidence levels are arranged in decreasing order
        valid_confs = logical_and(final_pos[:,0]>clims[j],final_pos[:,0]<=clims[j-1]);
        conf_rems = logical_and(valid_confs,remembers);
        conf_knows = logical_and(valid_confs,logical_and(logical_not(remembers),decisions));
        rem_RTs = resp_times[conf_rems];
        know_RTs = resp_times[conf_knows];
        params_rem = stats.gamma.fit(rem_RTs,floc=0);
        params_know = stats.gamma.fit(know_RTs,floc=0);
        p_remember_conf = stats.gamma.pdf(t,*params_rem)*DELTA_T*len(rem_RTs)/float(NR_SAMPLES);
        p_know_conf = stats.gamma.pdf(t,*params_know)*DELTA_T*len(know_RTs)/float(NR_SAMPLES);
        
        p_remember.append(p_remember_conf);
        p_know.append(p_know_conf);
        
    p_remember = array(p_remember);
    p_know = array(p_know);
        
    new_RTs = resp_times[logical_not(decisions)];
    params_new = stats.gamma.fit(new_RTs,floc=0);
    p_new = stats.gamma.pdf(t,*params_new)*DELTA_T*len(new_RTs)/float(NR_SAMPLES);
    return p_remember,p_know,p_new,t;

def predicted_proportions_NC(mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT,use_fftw=True):
    # compute process SD
    sigma_r = sqrt(2*d_r*DELTA_T);
    sigma_f = sqrt(2*d_f*DELTA_T);
    sigma = sqrt(sigma_r**2+sigma_f**2);

    # compute the correlation for r given r+f
    rho = sigma_r/sigma;
    rhoF = sigma_f/sigma;

    t = linspace(DELTA_T,MAX_T,NR_TSTEPS); # this is the time axis
    bound = exp(-tc_bound*t); # this is the collapsing bound
     
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
    #p_know = zeros(shape(t));
    p_remember = zeros(shape(t));
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
    
    mu_r_delta = mu_r_cond+mu_r*deltaT;
    s_r_delta = sqrt(s_r_cond**2+2*d_r*deltaT);
    p_remember[0] = p_old[0]*stats.norm.sf(r_bound,mu_r_delta,s_r_delta);
    # remove from consideration any particles that already hit the bound
    tx[0]*=(abs(x)<bound[0]);
    for i in range(1,len(t)):
        # convolve the particle distribution from the previous timestep
        # with the diffusion kernel (using Fourier domain convolution)
        if(use_fftw):
            tx[i] = abs(ifftshift(fftw.ifft(fftw.fft(tx[i-1])*ft_kernel)));
        else:
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
        # remove from consideration any particles that already hit the bound
        tx[i]*=(abs(x)<bound[i]);
        
        # old method for computing p_remember
        # p_remember[i] = sum(p_pos*stats.norm.sf(r_bound,mu_r_cond,s_r_cond));
        # new method for computin p_remember
        mu_r_delta = mu_r_cond+mu_r*deltaT;
        s_r_delta = sqrt(s_r_cond**2+2*d_r*deltaT);
        p_remember[i] = sum(p_pos*stats.norm.sf(r_bound,mu_r_delta,s_r_delta));

    #p_remember = p_old-p_know;
    p_know = p_old-p_remember;
    
    # TO DO:
    ## Added by Melchi on 6/15/2015 ##
    # The conditional distribution of position for the 'recall' component of the
    # particle (conditioned on time of new/old decision and/or value of r+f) has
    # already been computed. It's a normal distribution with mean mu_r_cond and
    # standard deviation s_r_cond. What remains to be done is to compute the
    # distribtuion of positions for the recall component after deltaT additional
    # seconds have elapsed
    # 1. The resulting distribution should have mean = mu_r_cond + (mu_r*deltaT)
    # 2. The resulting distribution should have SD = sqrt(s_r_cond^2+2*d_r*deltaT)
    ######################################################################################
    # determine the proportion of new, remember and know responses by confidence
    # determine the time points corresponding to quartiles within the overall distribution of remember and know responses

    
    return p_remember,p_know,p_new,t;
