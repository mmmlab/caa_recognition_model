# sinha_cpd_model.py
from scipy import stats
from scipy.integrate import fixed_quad 
from scipy.optimize import fmin_cg
import numpy
import pylab as pl
from pylab import array,vstack,hstack,size,zeros
# fixed parameters
mu_F0 = 0;
mu_R0 = 0;
sigma_R0 = 1;
sigma_F0 = 1;
sigma_F1 = 1;

parameters = array([0.92,0.81,0.78,1.48,0.08,0.69,1.15,1.67,2.32]);
observed = numpy.loadtxt('neha/observed.txt'); # load the observed data

# data predicted by the CDP model
def cdppred(parm, total_old, total_new):
    nc = size(parm) - 4;
    #decision criteria
    c = parm[-nc:];
    #r = recollection criterion
    r = parm[0];
    # muR1 = mean of recollection distribution for targets 
    mu_R1 = parm[1];
    #muF1 = mean of familiarity distribution for targets
    mu_F1 = parm[2];
    # sR1 = standard deviation of recollection distribution for targets
    sigma_R1 = parm[3];

    rem_hit = zeros((size(c),1));
    know_hit = zeros((size(c),1));
    rem_fa = zeros((size(c),1));
    know_fa = zeros((size(c),1));
    # Define limits of integration (for numerical integration below)
    upper_lim = mu_R1+10*sigma_R1;
    lower_lim = mu_R1-10*sigma_R1;
    
    integrand = lambda R,mu_F,s_F,mu_R,s_R,conf: (stats.norm.cdf(((mu_F+R)-conf)/s_F))*stats.norm.pdf(R,mu_R,s_R);
    
    for i in range(size(c)):
        remember = fixed_quad(integrand,r,upper_lim,args=(mu_F1,sigma_F1,mu_R1,sigma_R1,c[i]),n=20);
        know = fixed_quad(integrand,lower_lim,r,args=(mu_F1,sigma_F1,mu_R1,sigma_R1,c[i]),n=20);
        rememberFA = fixed_quad(integrand,r,upper_lim,args=(mu_F0,sigma_F0,mu_R0,sigma_R0,c[i]),n=20);
        knowFA = fixed_quad(integrand,lower_lim,r,args=(mu_F0,sigma_F0,mu_R0,sigma_R0,c[i]),n=20);
        
        rem_hit[i] = remember[0]*total_old;
        know_hit[i] = know[0]*total_old;
        rem_fa[i] = rememberFA[0]*total_new;
        know_fa[i] = knowFA[0]*total_new;

    for i in range(size(c)-1):
        rem_hit[i] = rem_hit[i]-rem_hit[i+1];
        know_hit[i] = know_hit[i]-know_hit[i+1];
        rem_fa[i] = rem_fa[i]-rem_fa[i+1];
        know_fa[i] = know_fa[i]-know_fa[i+1];
    miss = total_old - (rem_hit.sum()+know_hit.sum());
    cr = total_new - (rem_fa.sum()+know_fa.sum());
    
    temp = zeros((size(c),1));
    data_old= vstack((hstack((rem_hit[::-1],know_hit[::-1],temp)),array([0,0,miss])));
    data_new= vstack((hstack((rem_fa[::-1],know_fa[::-1],temp)),array([0,0,cr])));
    predicted_data = vstack((data_old,data_new));
    return predicted_data;
    
def cdppred_prop(parm):
    nc = size(parm) - 4;
    #decision criteria
    c = parm[-nc:];
    #r = recollection criterion
    r = parm[0];
    # muR1 = mean of recollection distribution for targets 
    mu_R1 = parm[1];
    #muF1 = mean of familiarity distribution for targets
    mu_F1 = parm[2];
    # sR1 = standard deviation of recollection distribution for targets
    sigma_R1 = parm[3];

    rem_hit = zeros((size(c),1));
    know_hit = zeros((size(c),1));
    rem_fa = zeros((size(c),1));
    know_fa = zeros((size(c),1));
    # Define limits of integration (for numerical integration below)
    upper_lim = mu_R1+10*sigma_R1;
    lower_lim = mu_R1-10*sigma_R1;
    
    integrand = lambda R,mu_F,s_F,mu_R,s_R,conf: (stats.norm.cdf(((mu_F+R)-conf)/s_F))*stats.norm.pdf(R,mu_R,s_R);
    
    for i in range(size(c)):
        remember = fixed_quad(integrand,r,upper_lim,args=(mu_F1,sigma_F1,mu_R1,sigma_R1,c[i]),n=20);
        know = fixed_quad(integrand,lower_lim,r,args=(mu_F1,sigma_F1,mu_R1,sigma_R1,c[i]),n=20);
        rememberFA = fixed_quad(integrand,r,upper_lim,args=(mu_F0,sigma_F0,mu_R0,sigma_R0,c[i]),n=20);
        knowFA = fixed_quad(integrand,lower_lim,r,args=(mu_F0,sigma_F0,mu_R0,sigma_R0,c[i]),n=20);
        
        rem_hit[i] = remember[0];
        know_hit[i] = know[0];
        rem_fa[i] = rememberFA[0];
        know_fa[i] = knowFA[0];

    for i in range(size(c)-1):
        rem_hit[i] = rem_hit[i]-rem_hit[i+1];
        know_hit[i] = know_hit[i]-know_hit[i+1];
        rem_fa[i] = rem_fa[i]-rem_fa[i+1];
        know_fa[i] = know_fa[i]-know_fa[i+1];
        
    miss = 1 - (sum(rem_hit)+sum(know_hit));
    cr = 1 - (sum(rem_fa)+sum(know_fa));
    
    temp = zeros((size(c),1));
    data_old= vstack((hstack((rem_hit[::-1],know_hit[::-1],temp)),array([0,0,miss])));
    data_new= vstack((hstack((rem_fa[::-1],know_fa[::-1],temp)),array([0,0,cr])));
    predicted_proportions = vstack((data_old,data_new));
    return predicted_proportions;
    
def cdppred_reduced(parm):
    #decision criteria
    c = parm[-1];
    #r = recollection criterion
    r = parm[0];
    # muR1 = mean of recollection distribution for targets 
    mu_R1 = parm[1];
    #muF1 = mean of familiarity distribution for targets
    mu_F1 = parm[2];
    # sR1 = standard deviation of recollection distribution for targets
    sigma_R1 = parm[3];


    # Define limits of integration (for numerical integration below)
    upper_lim = mu_R1+10*sigma_R1;
    lower_lim = mu_R1-10*sigma_R1;
    
    integrand = lambda R,mu_F,s_F,mu_R,s_R,conf: (stats.norm.cdf(((mu_F+R)-conf)/s_F))*stats.norm.pdf(R,mu_R,s_R);
    
    
    remember = fixed_quad(integrand,r,upper_lim,args=(mu_F1,sigma_F1,mu_R1,sigma_R1,c),n=20)[0];
    know = fixed_quad(integrand,lower_lim,r,args=(mu_F1,sigma_F1,mu_R1,sigma_R1,c),n=20)[0];
    rememberFA = fixed_quad(integrand,r,upper_lim,args=(mu_F0,sigma_F0,mu_R0,sigma_R0,c),n=20)[0];
    knowFA = fixed_quad(integrand,lower_lim,r,args=(mu_F0,sigma_F0,mu_R0,sigma_R0,c),n=20)[0];
        
    predicted_proportions = array([remember,know,rememberFA,knowFA]);
    return predicted_proportions;
    
def likelihood_function(parm,observed=observed):
    bparm = cdppred_prop(parm);
    n_old = observed[:6].sum();
    n_new = observed[6:].sum();
    
    p_old_rem = bparm[:5,0];
    p_old_know = bparm[:5,1];
    p_new_rem = bparm[7:-1,0];
    p_new_know = bparm[7:-1,1];
    
    k_old_rem = observed[:5,0];
    k_old_know = observed[:5,1];
    k_new_rem = observed[7:-1,0];
    k_new_know = observed[7:-1,1];
    
    LL_old = sum(stats.binom.logpmf(k_old_rem,n_old,p_old_rem))+sum(stats.binom.logpmf(k_old_know,n_old,p_old_know));
    
    LL_new = sum(stats.binom.logpmf(k_new_rem,n_new,p_new_rem))+sum(stats.binom.logpmf(k_new_know,n_new,p_new_know));

    return -LL_old-LL_new;
    
def likelihood_function_reduced(rparm,observed=observed):
    parm = hstack([[rparm[0]]*4,rparm]);
    p_old_rem,p_old_know,p_new_rem,p_new_know = cdppred_reduced(parm);
    n_old = observed[:6].sum();
    n_new = observed[6:].sum();
    
    k_old_rem = observed[:5,0].sum();
    k_old_know = observed[:5,1].sum();
    k_new_rem = observed[7:-1,0].sum();
    k_new_know = observed[7:-1,1].sum();
    
    LL_old = stats.binom.logpmf(k_old_rem,n_old,p_old_rem)+stats.binom.logpmf(k_old_know,n_old,p_old_know);
    
    LL_new = stats.binom.logpmf(k_new_rem,n_new,p_new_rem)+stats.binom.logpmf(k_new_know,n_new,p_new_know);

    return -LL_old-LL_new;