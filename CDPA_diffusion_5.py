
# In[1]:

from pylab import *
import numpy as np
import pylab as pl
from scipy import stats
from scipy import optimize
from scipy.integrate import fixed_quad 


# In[19]:

# Constants
DELTA_T     = 0.025;  # size of discrete time increment (sec.)
MAX_T       = 20
NR_TSTEPS   = MAX_T/DELTA_T;
NR_SSTEPS   = 8192#4096#2048;
NR_SAMPLES  = 10000; # number of trials to use for MC likelihood computation
n = 2; # number of confidence critetion
QUANT = array([25,50,75]);
#quant = array([20,40,60,80]);

R = 0.1; D = 0.05; L = 0.1; Z = 0.0;
#observed = numpy.loadtxt('myData.txt'); # load the observed data
#remember_hit = numpy.loadtxt('remRT_hit.txt'); # load remember RTs for hits
#know_hit = numpy.loadtxt('knowRT_hit.txt'); # load know RTs for hits
#remember_fa = numpy.loadtxt('remRT_fa.txt'); # load remember RTs for false alarms
#know_fa = numpy.loadtxt('knowRT_fa.txt');  # load know RTs for false alarms

observed = numpy.loadtxt('genData.txt'); # load the observed data
remember_hit = numpy.loadtxt('gen_rem_hit.txt'); # load remember RTs for hits
know_hit = numpy.loadtxt('gen_know_hit.txt'); # load know RTs for hits
remember_fa = numpy.loadtxt('gen_rem_fa.txt'); # load remember RTs for false alarms
know_fa = numpy.loadtxt('gen_know_fa.txt');  # load know RTs for false alarms
print observed;


# Out[19]:

#     [[  848.31626927   490.80124766     0.        ]
#      [  364.99685766   396.24921546     0.        ]
#      [   84.2443294    128.4317921      0.        ]
#      [    0.             0.          1645.62808806]
#      [  210.49635501   195.28630806     0.        ]
#      [  161.28810876   252.59096634     0.        ]
#      [   68.00027682   153.57653898     0.        ]
#      [    0.             0.          2910.29553155]]
#     

# In[3]:

remH_RT,remH_conf = numpy.split(remember_hit,2,axis=1);
knowH_RT,knowH_conf = numpy.split(know_hit,2,axis=1);
remFA_RT,remFA_conf = numpy.split(remember_fa,2,axis=1);
knowFA_RT,knowFA_conf = numpy.split(know_fa,2,axis=1);
print "rem hit high", remH_RT[remH_conf==2].mean();
print "rem hit med", remH_RT[remH_conf==1].mean();
print "rem hit low", remH_RT[remH_conf==0].mean();
print "know hit high", knowH_RT[knowH_conf==2].mean();
print "know hit med", knowH_RT[knowH_conf==1].mean();
print "know hit low", knowH_RT[knowH_conf==0].mean();


# Out[3]:

#     rem hit high 1.31647239264
#     rem hit med 2.93447204969
#     rem hit low 4.45642857143
#     know hit high 1.48516304348
#     know hit med 3.00610955056
#     know hit low 4.60579710145
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
    quant_rem = zeros((n+1,size(QUANT)));
    quant_know = zeros((n+1,size(QUANT)));
    
    quant_vals = QUANT/100;
    
    for i,hi_edge in enumerate(t_C):
        # find the distribution bounds for the current confidence level
        if(i==0):
            lo_edge = 0;
        else:
            lo_edge = t_C[i-1];
            
        
        # compute the CDF for this quantile
        prob_rem = p_remember[logical_and(t>=lo_edge,t<hi_edge)];
        rem[i] = prob_rem.sum()
        prob_rem = cumsum(prob_rem/rem[i]);
        
        prob_know = p_know[logical_and(t>=lo_edge,t<hi_edge)];
        know[i] = prob_know.sum();
        prob_know = cumsum(prob_know/know[i]);
        
        # find the index of the CDF value that most closely matches the desire quantile rank.
        # the time associated with that index is the quantile value
        
        quant_rem[i,:] = array([t[argmin(abs(prob_rem-q))] for q in quant_vals]);
        quant_know[i,:] = array([t[argmin(abs(prob_know-q))] for q in quant_vals]); 
    
    temp = zeros((n+1,1));
    data = vstack((hstack((rem,know,temp)),array([0,0,sum(p_new)])));
    data_quant = vstack((quant_rem,quant_know));
    
    #return data;
    return data,data_quant;
    #return sum(p_remember),sum(p_know),sum(p_new),t;


# In[5]:

data,dataquant= predicted_proportions(array([0.77527381,0.62883984]),0.04369458,0.07381991,0.09110745,0.0506929,0.08987588,0.52394113,3.27516992,-0.15628818);
print data;
print dataquant;

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


# In[7]:

param = array([0.8,0.5,3*R/4,R/4,D/2,D/2,L,0.6,0.9,0.2,3*R/4,R/4,D/2,D/2]);
prop,dataold,datanew = compute_proportion(param);
print prop;
print dataold;
print datanew;


# Out[7]:

#     [[ 0.03597732  0.24335071  0.        ]
#      [ 0.10122512  0.39883889  0.        ]
#      [ 0.01872873  0.05082803  0.        ]
#      [ 0.          0.          0.13279225]
#      [ 0.03597732  0.24335071  0.        ]
#      [ 0.10122512  0.39883889  0.        ]
#      [ 0.01872873  0.05082803  0.        ]
#      [ 0.          0.          0.13279225]]
#     [[ 1.35   1.675  1.975]
#      [ 3.05   3.95   5.15 ]
#      [ 7.5    8.175  9.2  ]
#      [ 1.2    1.55   1.9  ]
#      [ 2.9    3.725  4.875]
#      [ 7.45   8.075  9.075]]
#     [[ 1.35   1.675  1.975]
#      [ 3.05   3.95   5.15 ]
#      [ 7.5    8.175  9.2  ]
#      [ 1.2    1.55   1.9  ]
#      [ 2.9    3.725  4.875]
#      [ 7.45   8.075  9.075]]
#     

# In[8]:

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
        chi[i]= (((total*rem_prop[i]*QUANT[0]/100.00-sum(rem_data<rem_edge[i,0]))**2)/total*rem_prop[i]*QUANT[0]/100.00)+(((total*know_prop[i]*QUANT[0]/100.00-sum(know_data<know_edge[i,0]))**2)/total*know_prop[i]*QUANT[0]/100.00);
        total_predicted_rem = total*rem_prop[i]*QUANT[0]/100.00;
        total_predicted_know = total*know_prop[i]*QUANT[0]/100.00;
        total_observed_rem = sum(rem_data<rem_edge[i,0]);
        total_observed_know = sum(rem_data<know_edge[i,0]);
        
        for j in range(1,size(QUANT)):
            observed_rem = sum(logical_and(rem_data<rem_edge[i,j],rem_data>=rem_edge[i,j-1]));#sum(rem_data<rem_edge[i,j])-sum(rem_data<rem_edge[i,j-1]);
            predicted_rem = total*rem_prop[i]*QUANT[j]/100.00-total*rem_prop[i]*QUANT[j-1]/100.00;
            observed_know = sum(know_data<know_edge[i,j])-sum(know_data<know_edge[i,j-1]);
            predicted_know = total*know_prop[i]*QUANT[j]/100.00-total*know_prop[i]*QUANT[j-1]/100.00;
            total_predicted_rem += predicted_rem;
            total_predicted_know += predicted_know;
            total_observed_rem += observed_rem;
            total_observed_know += observed_know
            chi[i] += ((predicted_rem-observed_rem)**2)/predicted_rem + ((predicted_know-observed_know)**2)/predicted_know;
        chi[i] += (((total*rem_prop[i]-total_predicted_rem)-(size(rem_data)-total_observed_rem))**2)/(total*rem_prop[i]-total_predicted_rem)+(((total*know_prop[i]-total_predicted_know)-(size(know_data)-total_observed_know))**2)/(total*know_prop[i]-total_predicted_know);
    return sum(chi);            


# In[30]:

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


# In[31]:

#param = array([0.8,0.5,3*R/4,R/4,D/2,D/2,L,0.6,0.9,0.2,3*R/4,R/4,D/2,D/2]);
#param = array([0.77527381,0.62883984,0.04369458,0.07381991,0.09110745,0.0506929,0.08987588,0.52394113,3.27516992,-0.15628818,-0.06139583,-0.01106722,0.05114411,0.03976751]);
#param = array([0.84115819,0.74704381,0.04310687,0.07412053,0.09164065,0.0519376,0.09200852,0.54037748,3.23358512,-0.15311715,-0.06078642,-0.01088571,0.05144551,0.0405709]);
#param = array([0.78541181,0.63970898,0.03955972,0.07311988,0.1072794,0.06329539,0.10958039,0.51742639,3.1245284,-0.12552528,-0.08099549,-0.01066264,0.06626993,0.04743122]);
#param = array([0.78683601,0.63960793,0.05422867,0.07451649,0.10709069,0.06478986,0.11458067,0.51158948,2.79327592,-0.1263059,-0.09388634,-0.01071401,0.06806819,0.05255354]);
#**param = array([0.7869721,0.63700509,0.05435808,0.0740718,0.10759311,0.07045798,0.11545176,0.50618527,2.753773,-0.12775451,-0.0934401,-0.01072134,0.06861823,0.05195974]);
#*param = array([0.78459859,0.63952242,0.04978944,0.07352597,0.10670458,0.06402103,0.11025439,0.51524347,2.7886897,-0.12626302,-0.09405035,-0.01071137,0.06782216,0.05254717]);
#*param = array([0.78296261,0.63997263,0.04632127,0.07460343,0.10695214,0.06320797,0.11019207,0.51554405,2.97408063,-0.12514898,-0.08946891,-0.01047399,0.06747498,0.05241066]);
#*param = array([0.78259688,0.64005155,0.04426498,0.07455727,0.10692587,0.06322004,0.11017429,0.5155061,2.96924978,-0.12505542,-0.08874669,-0.0104581,0.06760646,0.05290787]);
#param = array([0.7844855,0.63812949,0.04160994,0.07422986,0.10679761,0.06319589,0.11031198,0.51604383,3.02626311,-0.12463057,-0.0848216,-0.01059258,0.06564244,0.05272451]);
#param = array ([0.78451135,0.6381526,0.03983549,0.0741186,0.1066705,0.06339201,0.11026772,0.51609682,3.04724215,-0.12461062,-0.08430678,-0.01051689,0.06568695,0.05271524]);
#param = array([0.78455148,0.63818336,0.03982505,0.07405859,0.10665008,0.06347432,0.10989405,0.51688956,3.0489146,-0.12468622,-0.08440509,-0.01050146,0.06502361,0.05080061]);
#param = array([0.78541181,0.63970898,0.03955972,0.07311988,0.1072794,0.06329539,0.10958039,0.51742639,3.1245284,-0.12552528,-0.08099549,-0.01066264,0.06626993,0.04743122]);
#param = array([0.78459678,0.63952194,0.05422463,0.07452259,0.10671192,0.06402805,0.11025454,0.5152371,2.7871542,-0.12625914,-0.09405059,-0.01071414,0.06782042,0.05254881]);
#param = array([0.77418006,0.64760922,0.04470405,0.05987517,0.11397229,0.06405119,0.11128166,0.47274559,2.46608167,-0.13583415,-0.11935047,-0.01123566,0.06999618,0.05581485]);
#param = array([0.77380524,0.64736149,0.04495912,0.05991637,0.11904682,0.06406311,0.1114897,0.47119221,2.39503682,-0.13841441,-0.12783232,-0.0108359,0.06677878,0.05729744]);
#param = array([0.77370433,0.64656344,0.04382392,0.06021717,0.13038513,0.06539262,0.11125704,0.46793749,2.38316611,-0.14153438,-0.12914206,-0.01057224,0.06711001,0.05758781]);
#param = array([0.78217143,0.59825178,0.0402868,0.05977743,0.13962314,0.06859625,0.10912826,0.47256909,2.3039337,-0.1413971,-0.1354701,-0.01075517,0.06819155,0.0610058]);
#param = array([0.78318031,0.60190985,0.02972369,0.07107757,0.16312199,0.06712262,0.10838358,0.49642148,1.34586881,-0.15470904,-0.14647127,-0.01224913,0.07327821,0.06133563]);
#param = array([0.7824175,0.60426427,0.02899263,0.07380202,0.16343247,0.06748316,0.10833204,0.49467424,1.33786291,-0.1560848,-0.14645571,-0.01244893,0.07380844,0.06138794]);
#param = array([0.7831307,0.60253825,0.02923674,0.07357183,0.16362669,0.06709437,0.10826691,0.49595259,1.34030801,-0.1552733,-0.14614531,-0.0123076,0.07349555,0.06135621]);
param = array([0.78459678,0.63952194,0.05422463,0.07452259,0.10671192,0.06402805,0.11025454,0.5152371,2.7871542,-0.12625914,-0.09405059,-0.01071414,0.06782042,0.05254881]);
chi = chi_square(param);
chi


# Out[31]:

#     448.49595089855364

# In[ ]:

#param_initial = array([0.78459678,0.63952194,0.05422463,0.07452259,0.10671192,0.06402805,0.11025454,0.5152371,2.7871542,-0.12625914,-0.09405059,-0.01071414,0.06782042,0.05254881]);
#param_initial = array([0.77370433,0.64656344,0.04382392,0.06021717,0.13038513,0.06539262,0.11125704,0.46793749,2.38316611,-0.14153438,-0.12914206,-0.01057224,0.06711001,0.05758781]);
#param_initial = array([0.78217143,0.59825178,0.0402868,0.05977743,0.13962314,0.06859625,0.10912826,0.47256909,2.3039337,-0.1413971,-0.1354701,-0.01075517,0.06819155,0.0610058]);
#param_initial = array([0.78318031,0.60190985,0.02972369,0.07107757,0.16312199,0.06712262,0.10838358,0.49642148,1.34586881,-0.15470904,-0.14647127,-0.01224913,0.07327821,0.06133563]);
#param_initial = array([0.7824175,0.60426427,0.02899263,0.07380202,0.16343247,0.06748316,0.10833204,0.49467424,1.33786291,-0.1560848,-0.14645571,-0.01244893,0.07380844,0.06138794]);
#param_initial = array([0.7831307,0.60253825,0.02923674,0.07357183,0.16362669,0.06709437,0.10826691,0.49595259,1.34030801,-0.1552733,-0.14614531,-0.0123076,0.07349555,0.06135621]);
#param_initial = array([0.7831307,0.60253825,0.02923674,0.07357185,0.1636267,0.06709437,0.10826691,0.49595259,1.34030798,-0.15527331,-0.14614542,-0.0123076,0.07349555,0.06135621]);
param_initial = array([0.78459678,0.63952194,0.05422463,0.07452259,0.10671192,0.06402805,0.11025454,0.5152371,2.7871542,-0.12625914,-0.09405059,-0.01071414,0.06782042,0.05254881]);
for i in range(10):
    output = optimize.minimize(chi_square,param_initial,method='Nelder-Mead'); # estimate parameters that minimize the negative log likelihood
    print output.x;
    print output.message;
    param_initial=output.x;
    chi = chi_square(output.x); # check the value of chi square with estimated parameters (sanity check?)
    print chi;
    print "end"


# In[9]:

output = optimize.brute(chi_square,((0.01,1.0),(0.01,1.0),(0.01,1.0),(0.01,1.0),(-0.5,0.5),)); 
print output;
print chi_square(output);


# Out[9]:




