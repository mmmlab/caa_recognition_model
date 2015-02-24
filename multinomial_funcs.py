import pylab as pl
import scipy as sp
from scipy.special import gamma, gammaln

def multinom_loglike(x,n,p):
    """
    where x is a k-length vector representing the number of observed events in
    each category, n is a scalar representing the number of trials, and p is a
    k-length vector representing the proportion parameter for each category
    
    returns the log likelihood of obtaining x events in each category
    """
    return gammaln(n+1)-pl.sum(gammaln(x+1))+pl.sum(x*gammaln(p));

def chi_square_gof(x,n,p):
    return sum((x-n*p)**2/(n*p));