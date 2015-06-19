from pylab import *
import pylab as pl
import numpy
from scipy import stats
from scipy import optimize
import pyfftw
import fftw_test as fftw
from multinomial_funcs import multinom_loglike,chi_square_gof

## Start by reading in the data.
# the reason to do this first is that, in order to be efficient,
# we don't want to represent any more of the time axis than we have to.

remember_hit = numpy.loadtxt('data/remRT_hit.txt'); # load remember RTs for hits
know_hit = numpy.loadtxt('data/knowRT_hit.txt'); # load know RTs for hits
remember_fa = numpy.loadtxt('data/remRT_fa.txt'); # load remember RTs for false alarms
know_fa = numpy.loadtxt('data/knowRT_fa.txt');  # load know RTs for false alarms
CR = numpy.loadtxt('data/CR.txt');  # load CR RTs 
miss = numpy.loadtxt('data/miss.txt');  # load miss RTs


## read in collapsed data 
#remember_hit = numpy.loadtxt('data_collapsed/remRT_hit.txt'); # load remember RTs for hits
#know_hit = numpy.loadtxt('data_collapsed/knowRT_hit.txt'); # load know RTs for hits
#remember_fa = numpy.loadtxt('data_collapsed/remRT_fa.txt'); # load remember RTs for false alarms#
#know_fa = numpy.loadtxt('data_collapsed/knowRT_fa.txt');  # load know RTs for false alarms
#CR = numpy.loadtxt('data_collapsed/CR.txt');  # load CR RTs 
#miss = numpy.loadtxt('data_collapsed/miss.txt');  # load miss RTs 

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
param_bounds = [(0.2,1.0),(0.01,0.19),(0.0,1.0),(0.0,1.0),(EPS,1.0),(EPS,1.0),(0.0,10.0)];


fftw.fftw_setup(zeros(NR_SSTEPS),NR_THREADS);


def find_ml_params():
    return optimize.differential_evolution(compute_chi,param_bounds);

def compute_nll(parameters,remember_hit=remember_hit,know_hit=know_hit,remember_fa=remember_fa,know_fa=know_fa,miss=miss,CR=CR):
    
    c = array([0.64990402,parameters]);
    #c = parameters[0]
    mu_r = 0.13935456; 
    mu_f = 0.01447192;  
    d_r = 0.05391228; 
    d_f = 0.11416173; 
    tc_bound = 0.28585528; 
    r_bound = 0.35386679; 
    z0 = -0.09009103; 
    mu_r0 = -0.02782681;  
    mu_f0 = -0.14973519; 
    deltaT = 0.43406613;
    
    nll = 0;

    remH_RT,remH_conf = numpy.split(remember_hit,2,axis=1);
    knowH_RT,knowH_conf = numpy.split(know_hit,2,axis=1);
    remFA_RT,remFA_conf = numpy.split(remember_fa,2,axis=1);
    knowFA_RT,knowFA_conf = numpy.split(know_fa,2,axis=1);
    CR_RT,CR_conf = numpy.split(CR,2,axis=1);
    miss_RT,miss_conf = numpy.split(miss,2,axis=1);
        
    rem_quantiles_old,know_quantiles_old,data_old = predicted_proportions(c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT);
    rem_quantiles_new,know_quantiles_new,data_new = predicted_proportions(c,mu_r0,mu_f0,d_r,d_f,tc_bound,r_bound,z0,deltaT);
    
    for i in range(n+1):
        rem_old = remH_RT[remH_conf==n-i];
        know_old = knowH_RT[knowH_conf==n-i];
        miss_old = miss_RT[miss_conf==n-i];
       
        rem_new = remFA_RT[remFA_conf==n-i];
        know_new = knowFA_RT[knowFA_conf==n-i];
        CR_new = CR_RT[CR_conf==n-i];
          
        nll+= compute_nll_conf(rem_old,know_old,miss_old,rem_new,know_new,CR_new,rem_quantiles_old[i],know_quantiles_old[i],rem_quantiles_new[i],know_quantiles_new[i],data_old[i],data_new[i]);    
    return nll;


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

def compute_chi(parameters,remember_hit=remember_hit,know_hit=know_hit,remember_fa=remember_fa,know_fa=know_fa,miss=miss,CR=CR):
    #c = c = parameters[0:n];
    #mu_r = parameters[n+0]; 
    #mu_f = parameters[n+1]; 
    #d_r = parameters[n+2]; 
    #d_f = parameters[n+3]; 
    #tc_bound = parameters[n+4]; 
    #r_bound = parameters[n+5];
    #z0 = parameters[n+6]; 
    #mu_r0 = parameters[n+7]; 
    #mu_f0 = parameters[n+8]; 
    #deltaT = parameters[n+9];

    c = c = parameters[0:n];
    mu_r = parameters[n+0]; #0.13935456; 
    mu_f = parameters[n+1]; #0.01447192; 
    d_r = parameters[n+2]; #0.05391228; 
    d_f = parameters[n+3]; #0.11416173;  
    tc_bound = 0.28585528; 
    r_bound = 0.35386679; 
    z0 = -0.09009103; 
    mu_r0 = -0.02782681;  
    mu_f0 = -0.14973519; 
    deltaT = parameters[n+4];
    

    chi = 0;

    remH_RT,remH_conf = numpy.split(remember_hit,2,axis=1);
    knowH_RT,knowH_conf = numpy.split(know_hit,2,axis=1);
    remFA_RT,remFA_conf = numpy.split(remember_fa,2,axis=1);
    knowFA_RT,knowFA_conf = numpy.split(know_fa,2,axis=1);
    CR_RT,CR_conf = numpy.split(CR,2,axis=1);
    miss_RT,miss_conf = numpy.split(miss,2,axis=1);
        
    rem_quantiles_old,know_quantiles_old,data_old = predicted_proportions(c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT);
    rem_quantiles_new,know_quantiles_new,data_new = predicted_proportions(c,mu_r0,mu_f0,d_r,d_f,tc_bound,r_bound,z0,deltaT);
    
    for i in range(n+1):
        rem_old = remH_RT[remH_conf==n-i];
        know_old = knowH_RT[knowH_conf==n-i];
        miss_old = miss_RT[miss_conf==n-i];
        
 
        rem_new = remFA_RT[remFA_conf==n-i];
        know_new = knowFA_RT[knowFA_conf==n-i];
        CR_new = CR_RT[CR_conf==n-i];
    
        chi+= compute_chi_conf(rem_old,know_old,miss_old,rem_new,know_new,CR_new,rem_quantiles_old[i],know_quantiles_old[i],rem_quantiles_new[i],know_quantiles_new[i],data_old[i],data_new[i]);    
    return chi;


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

def predicted_proportions(c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT,use_fftw=True):
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
    
    #p_know[0] = p_old[0]*stats.norm.sf(f_bound,mu_f_cond,s_f_cond)+p_old[0]*stats.norm.cdf(r_bound,mu_r_cond,s_r_cond);
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

    rem = zeros((n+1,1));
    know = zeros((n+1,1));
    new = zeros((n+1,1));
    quant_rem = zeros((n+1,size(QUANT)));
    quant_know = zeros((n+1,size(QUANT)));
    quant_new = zeros((n+1,size(QUANT)));
    
    p_rem_conf = zeros((n+1,size(t))); 
    p_know_conf = zeros((n+1,size(t)));
    p_new_conf = zeros((n+1,size(t)));
                        
    sigma_r_conf = sqrt(2*d_r*deltaT);
    sigma_f_conf = sqrt(2*d_f*deltaT);
    sigma_conf = sqrt(sigma_r_conf**2+sigma_f_conf**2);
    
    mu_conf = (mu_r+mu_f)*deltaT;
    
    if (n==1):
        c = array([c]);

    for i in range(n): 
        
        if (i==0):
            p_rem_conf[i] = p_remember*stats.norm.sf(c[i],mu_conf+bound,sigma_conf);
            p_know_conf[i] = p_know*stats.norm.sf(c[i],mu_conf+bound,sigma_conf);
            p_new_conf[i] = p_new*stats.norm.sf(c[i],mu_conf+bound,sigma_conf);
            
        else:
            p_rem_conf[i] = p_remember*(stats.norm.sf(c[i],mu_conf+bound,sigma_conf)-stats.norm.sf(c[i-1],mu_conf+bound,sigma_conf));
            p_know_conf[i] = p_know*(stats.norm.sf(c[i],mu_conf+bound,sigma_conf)-stats.norm.sf(c[i-1],mu_conf+bound,sigma_conf));
            p_new_conf[i] = p_new*(stats.norm.sf(c[i],mu_conf+bound,sigma_conf)-stats.norm.sf(c[i-1],mu_conf+bound,sigma_conf));
        
        rem[i] = p_rem_conf[i].sum();
        know[i] = p_know_conf[i].sum()
        new[i] = p_new_conf[i].sum()
        
        # compute the CDF for this quantile
        prob_rem = cumsum(p_rem_conf[i]/(rem[i]+EPS));
        prob_know = cumsum(p_know_conf[i]/(know[i]+EPS));
        prob_new = cumsum(p_new_conf[i]/(new[i]+EPS));
        
        # find the index of the CDF value that most closely matches the desire quantile rank.
        # the time associated with that index is the quantile value
        quant_rem[i,:] = array([t[argmax(abs(prob_rem>q))] for q in QUANT]);
        quant_know[i,:] = array([t[argmax(abs(prob_know>q))] for q in QUANT]); 
        quant_new[i,:] = array([t[argmax(abs(prob_new>q))] for q in QUANT]);
        quant_rem[i,0] = 0;
        quant_know[i,0] = 0;
        quant_new[i,0] = 0;
    
    # for low confidence data
    p_rem_conf[n] = p_remember*stats.norm.cdf(c[n-1],mu_conf+bound,sigma_conf);
    p_know_conf[n] = p_know*stats.norm.cdf(c[n-1],mu_conf+bound,sigma_conf);
    p_new_conf[n] = p_new*stats.norm.cdf(c[n-1],mu_conf+bound,sigma_conf);
    
    rem[n] = p_rem_conf[n].sum();
    know[n] = p_know_conf[n].sum();
    new[n] = p_new_conf[n].sum();
      
    prob_rem = cumsum(p_rem_conf[n]/(rem[n]+EPS));
    prob_know = cumsum(p_know_conf[n]/(know[n]+EPS));
    prob_new = cumsum(p_new_conf[n]/(new[n]+EPS));
    quant_rem[n,:] = array([t[argmax(abs(prob_rem>q))] for q in QUANT]);
    quant_know[n,:] = array([t[argmax(abs(prob_know>q))] for q in QUANT]); 
    quant_new[n,:] = array([t[argmax(abs(prob_new>q))] for q in QUANT]);
    quant_rem[n,0] = 0;
    quant_know[n,0] = 0;
    quant_new[n,0] = 0;
           
    # To check predicted proportions against observed data
    #data = hstack((rem,know,new));
    #return quant_rem,quant_know,quant_new,data;

    # To minimize compute_chi/compute_nll (excluding the "new" data)
    data = hstack((rem,know));
    return quant_rem,quant_know,data;

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
    
    #p_know[0] = p_old[0]*stats.norm.sf(f_bound,mu_f_cond,s_f_cond)+p_old[0]*stats.norm.cdf(r_bound,mu_r_cond,s_r_cond);
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
