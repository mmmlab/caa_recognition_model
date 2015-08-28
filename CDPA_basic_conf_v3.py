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

rem_hit = numpy.loadtxt('neha/data/remRT_hit.txt'); # load remember RTs for hits
know_hit = numpy.loadtxt('neha/data/knowRT_hit.txt'); # load know RTs for hits
rem_fa = numpy.loadtxt('neha/data/remRT_fa.txt'); # load remember RTs for false alarms
know_fa = numpy.loadtxt('neha/data/knowRT_fa.txt');  # load know RTs for false alarms
CR = numpy.loadtxt('neha/data/CR.txt');  # load CR RTs 
miss = numpy.loadtxt('neha/data/miss.txt');  # load miss RTs

EPS         = 1e-10 # a very small value (used for numerical stability)
NR_THREADS  = 1;    # this is for multithreaded fft
DELTA_T     = 0.05;  # size of discrete time increment (sec.)
MAX_T       = 10; #ceil(percentile(all_RT,99.5));
NR_TSTEPS   = MAX_T/DELTA_T;
NR_SSTEPS   = 4096; #8192#4096#2048;
NR_SAMPLES  = 10000; # number of trials to use for MC likelihood computation
n = 2 # number of confidence critetion
NR_QUANTILES = 4;
R = 0.1; D = 0.05; L = 0.1; Z = 0.0;

fftw.fftw_setup(zeros(NR_SSTEPS),NR_THREADS);

params_est = [0.7233685,0.08959858,0.50357936,0.96055488,0.29500501,0.00464397,1.0,0.67084697,-0.74701904,-0.25206785,0.84902461,0.22969753];

param_bounds = [(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0),(EPS,1.0),(EPS,1.0),(0.05,1.0),(0.0,1.0),(-1.0,1.0),(-1.0,1.0),(-1.0,1.0),(EPS,1.0)];
# c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,mu_r0,mu_f0,delta_t = params;

def find_ml_params():
    #return optimize.differential_evolution(compute_model_gof,param_bounds);
	return optimize.differential_evolution(chi_square,param_bounds);
	
# plotting the data
def plot_data(parameters):
    c = parameters[0:n];
    mu_r = parameters[n+0]; 
    mu_f = parameters[n+1]; 
    d_r = parameters[n+2]; 
    d_f = parameters[n+3]; 
    tc_bound = parameters[n+4]; 
    r_bound = parameters[n+5];
    z0 = parameters[n+6]; 
    mu_r0 = parameters[n+7]; 
    mu_f0 = parameters[n+8]; 
    deltaT = parameters[n+9]; 
    #f_bound = parameters[n+10];

    params_old = array([c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT]);
    params_new = array([c,mu_r0,mu_f0,d_r,d_f,tc_bound,r_bound,z0,deltaT]);
    p_remember,p_know,p_new,t = predicted_proportions(*params_old);
    p_rememberN,p_knowN,p_newN,t = predicted_proportions(*params_new);
    
    # plot the histogram for observed data
    #hist(rem_hit[:,0],bins=40,range = [0,10],histtype='step',color='0.5',lw=2,normed=True);
    
    # compute empirical p(remember|old)
    p_rem_total = len(rem_hit)*1.0/(len(rem_hit)+len(know_hit)+len(miss));
    # compute density histogram
    hd,edges = histogram(rem_hit[:,0],bins=40,range=[0,10],density=True);
    hist_density = hstack([[0],hd]);
    step(edges,hist_density*p_rem_total,color='0.5',lw=2);
    # plot the PDF for observed data
    rem_old = stats.kde.gaussian_kde(rem_hit[:,0]);
    plot(t,rem_old(t)*p_rem_total,'r',lw=2)
    # generate the raster plot of raw RTs
    rmin = 0;
    rmax = max(rem_old(t))*0.1;
    vlines(rem_hit[:,0],rmin,rmax,color='k',alpha=0.15);
    # plot the PDF for predicted data
    # note: the division by DELTA_T below is to make sure that you are plotting
    # probability densities (rather than probability masses)
    plot(t,p_remember.sum(0)/DELTA_T,'g',lw=2)
    title('Remember Hits');
    axis([0,t.max(),None,None])
    show();
	
# compute the chi square
observed = numpy.loadtxt('neha/data/myData.txt');
def chi_square(parameters,observed=observed):
    c = parameters[0:n];
    mu_r = parameters[n+0]; 
    mu_f = parameters[n+1]; 
    d_r = parameters[n+2]; 
    d_f = parameters[n+3]; 
    tc_bound = parameters[n+4]; 
    r_bound = parameters[n+5];
    z0 = parameters[n+6]; 
    mu_r0 = parameters[n+7]; 
    mu_f0 = parameters[n+8]; 
    deltaT = parameters[n+9]; 
    #f_bound = parameters[n+10];

    old = sum(observed[0:n+1,0:3]);
    new = sum(observed[n+1:2*(n+1),0:3]);
    model_params_old = array([c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT]);
    model_params_new = array([c,mu_r0,mu_f0,d_r,d_f,tc_bound,r_bound,z0,deltaT]);
    rem_quantiles,know_quantiles,new_quantiles,p_r,p_k,p_n = compute_model_quantiles(model_params_old);
    rem_quantilesN,know_quantilesN,new_quantilesN,p_rN,p_kN,p_nN = compute_model_quantiles(model_params_new);
    data_old = vstack((p_r,p_k,array([0,0,p_n])));
    data_new = vstack((p_rN,p_kN,array([0,0,p_nN])));
    #data_old = vstack((p_r,p_k,array([0,p_n])));
    #print data_old;
    predicted = vstack((data_old,data_new)); 
    #print predicted;
    chi = 0;
    for i in range(size(observed,0)): 
        for j in range(size(observed,1)):
            if predicted[i,j]>0:
                if i <= n:
                    #print "old",i,j;
                    predicted[i,j]=predicted[i,j]*old;
                else:
                    #print "new",i,j;
                    predicted[i,j]=predicted[i,j]*new;
                chi = chi + ((predicted[i,j]-observed[i,j])**2)/predicted[i,j];
    return chi;

def compute_model_gof(parameters):
    # computes the chi square fit of the model to the data

    c = parameters[0:n];
    mu_r = parameters[n+0]; 
    mu_f = parameters[n+1]; 
    d_r = parameters[n+2]; 
    d_f = parameters[n+3]; 
    tc_bound = parameters[n+4]; 
    r_bound = parameters[n+5];
    z0 = parameters[n+6]; 
    mu_r0 = parameters[n+7]; 
    mu_f0 = parameters[n+8]; 
    deltaT = parameters[n+9]; 
    
    model_params_old = array([c,mu_r,mu_f,d_r,d_f,tc_bound,r_bound,z0,deltaT]);
    model_params_new = array([c,mu_r0,mu_f0,d_r,d_f,tc_bound,r_bound,z0,deltaT]);
    chi_old = compute_gof(model_params_old,rem_hit[:,0],know_hit[:,0],miss[:,0],rem_hit[:,1],know_hit[:,1]);
    chi_new = compute_gof(model_params_new,rem_fa[:,0],know_fa[:,0],CR[:,0],rem_fa[:,1],know_fa[:,1]);
    return chi_old+chi_new;
    #return chi_old
	
def compute_gof(model_params,rem_RTs,know_RTs,new_RTs,rem_conf,know_conf):
    # computes the chi square fit of the model to the data
    # compute N, the total number of trials
    N = len(rem_RTs)+len(know_RTs)+len(new_RTs);
    # compute x, the observed frequency for each category

    rem_quantiles,know_quantiles,new_quantiles,p_r,p_k,p_n = compute_model_quantiles(model_params);
    # determine the number of confidence levels being used in the model
    nr_conf_levels = len(rem_quantiles);
    #print nr_conf_levels;
    # adjust the number of confidence levels in the data to match
    rem_conf = clip(rem_conf,0,nr_conf_levels-1);
    know_conf = clip(know_conf,0,nr_conf_levels-1);
    ## compute the number of RTs falling into each quantile bin
    rem_freqs = array([-diff([sum(rem_RTs[rem_conf==i]>q) for q in rem_quantiles[i]]+[0]) for i in range(nr_conf_levels)]);
    know_freqs = array([-diff([sum(know_RTs[know_conf==i]>q) for q in know_quantiles[i]]+[0]) for i in range(nr_conf_levels)]);
    new_freqs = -diff([sum(new_RTs>q) for q in new_quantiles]+[0]);
    x = hstack([rem_freqs.flatten(),know_freqs.flatten(),new_freqs]);
    
    # compute p, the probability associated with each category in the model
    p_rem = p_r[:,newaxis]*ones((nr_conf_levels,NR_QUANTILES))/float(NR_QUANTILES);
    p_know = p_k[:,newaxis]*ones((nr_conf_levels,NR_QUANTILES))/float(NR_QUANTILES);
    p_new = p_n*ones(NR_QUANTILES)/float(NR_QUANTILES);
    p = hstack([p_rem.flatten(),p_know.flatten(),p_new]);
    return chi_square_gof(x,N,p)
	
def compute_model_quantiles(params):
    # This function is set up to deal with multiple confidence levels
    quantile_increment = 1.0/NR_QUANTILES;
    quantiles = arange(0,1,quantile_increment);
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

        #p_know[i] = sum(p_pos*stats.norm.sf(f_bound,mu_f_cond,s_f_cond))+sum(p_pos*stats.norm.cdf(r_bound,mu_r_cond,s_r_cond));
        p_remember[i] = sum(p_pos*stats.norm.sf(r_bound,mu_r_cond,s_r_cond));
        # remove from consideration any particles that already hit the bound
        tx[i]*=(abs(x)<bound[i]);

    #p_remember = p_old-p_know;
    p_know = p_old-p_remember;
    
    ######################################################################################
    # determine the proportion of new, remember and know responses by confidence
    # determine the time points corresponding to quartiles within the overall distribution of remember and know responses
    
    p_rem_conf = zeros((n+1,size(t)));
    p_know_conf = zeros((n+1,size(t)));
                        
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
            
        else:
            p_rem_conf[i] = p_remember*(stats.norm.sf(c[i],mu_conf+bound,sigma_conf)-stats.norm.sf(c[i-1],mu_conf+bound,sigma_conf));
            p_know_conf[i] = p_know*(stats.norm.sf(c[i],mu_conf+bound,sigma_conf)-stats.norm.sf(c[i-1],mu_conf+bound,sigma_conf));        
    
    # for low confidence data
    p_rem_conf[n] = p_remember*stats.norm.cdf(c[n-1],mu_conf+bound,sigma_conf);
    p_know_conf[n] = p_know*stats.norm.cdf(c[n-1],mu_conf+bound,sigma_conf);
    
    p_remember = p_rem_conf.sum(0); 
    p_know = p_know_conf.sum(0);
    return p_rem_conf,p_know_conf,p_new,t;
