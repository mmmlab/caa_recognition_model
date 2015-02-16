import pymc as pm
import neha.simple_d2b as d2b
import pylab as pl

# generate data
observed_data = pl.array([d2b.generate_sample(0.5,0.2,0) for i in range(40)]).T;
# set priors
r = pm.Normal('r', mu=0, tau=2);
z = pm.Uniform('z', lower=-1, upper=1);
D = pm.Gamma('D', alpha=2, beta=0.25);

decision = pm.Stochastic(logp = d2b.compute_likelihood,
                doc = 'decision data',
                name = 'decision',
                parents = {'r': r, 'D': D, 'z': z},
                random = d2b.generate_sample,
                trace = True,
                value = observed_data,
                dtype=None,
                rseed = 1.,
                observed = True,
                cache_depth = 2,
                plot=False,
                verbose = 0)

