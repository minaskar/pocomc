.. _faq:

FAQ
===

This page contains a collection of common questions and answers regarding
the use of ``pocoMC``.

What is the philosophy behind pocoMC?
-------------------------------------
The philosophy behind pocoMC is to provide a simple, easy-to-use, and efficient
sampler for Bayesian inference. The sampler is designed to be easy to use and
requires minimal tuning, while still providing efficient sampling for a wide
range of problems. The sampler is designed to be a drop-in replacement for
nested sampling, providing similar functionality with improved efficiency. The
main target audience for pocoMC is researchers who want to perform Bayesian
inference on complex models, but do not want to spend a lot of time tuning the
sampler. Furthermore, pocoMC recognizes that many researchers are not experts in
Bayesian inference, and aims to provide a simple and intuitive interface for
performing Bayesian inference. Finally, pocoMC is designed to be efficient and tailored
to computationally expensive models that often arise in scientific research and engineering.


How does pocoMC compare to other samplers?
------------------------------------------
pocoMC is a novel sampler that achieves the efficiency of gradient-based samplers
(e.g., Hamiltonian Monte Carlo) without requiring gradients. pocoMC uses a normalizing
flow to map the target distribution to a simple distribution, and then applies
t-preconditioned Crank-Nicolson to sample from the simple distribution. This allows
pocoMC to explore the target distribution more efficiently than traditional samplers
such as Metropolis-Hastings and nested sampling. In practice, we have found that pocoMC
often outperforms gradient-based samplers in terms of efficiency and robustness, especially
for complex target distributions. Compared to nested sampling, pocoMC generally requires 
fewer iterations to go from the prior to the posterior, and is often more efficient in terms
of computational cost. However, for low dimensional problems or simple target distributions,
nested sampling may still be a good choice. In general, pocoMC is a versatile and efficient
sampler that is well-suited for a wide range of problems.


What is Preconditioned Monte Carlo?
-----------------------------------
Preconditioned Monte Carlo (PMC) is a general framework for sampling from complex target distributions
using simple distributions. The idea behind PMC is to use Persistent Sampling (i.e., a generalization of
Sequential Monte Carlo) combined with normalizing flow preconditioning and a novel gradient-free Markov
kernel called t-preconditioned Crank-Nicolson. The normalizing flow is used to map the target distribution
to a simple distribution, and then t-preconditioned Crank-Nicolson is used to sample from the simple distribution.
Persistent sampling is used to maintain a set of active particles that explore the target distribution efficiently, 
starting from the prior and gradually moving towards the posterior. The combination of normalizing flow preconditioning,
t-preconditioned Crank-Nicolson, and persistent sampling allows PMC to efficiently explore complex target distributions
without requiring gradients. PMC is a general framework that can be applied to a wide range of problems, and pocoMC is
an implementation of PMC that is tailored to Bayesian inference in science and engineering.


Does the sampler scale well to high dimensions?
-----------------------------------------------
Yes, the sampler scales well to high dimensions. The sampler uses a normalizing
flow to scale to high dimensions. The normalizing flow is a bijective transformation
that maps a simple distribution to a complex distribution. By turning the complex target 
distribution into a simple distribution, the sampler can take advantage of the symmetries
of the simple distribution to explore the target distribution more efficiently even in
high dimensions. Training the normalizing flow in high dimensions requires an increased
number of particles, which in turn can increase the computational cost. This means that 
in high-dimensional problems, there are some dimininishing returns in terms of the number
of particles used. However, we were able to sample from targets with more than 100 dimensions
very efficiently, often outperforming gradient-based samplers.


Does the sampler use gradients to scale to high dimensions?
-----------------------------------------------------------
No, the sampler does not use gradients to scale to high dimensions. The way that pocoMC is able to 
scale to high dimensions is by taking advantage of the geometry of the target distribution. The sampler
uses a normalizing flow to map the target distribution to a simple distribution. Then, instead of applying
gradient-based samplers (e.g., Hamiltonian Monte Carlo) to the target distribution, the sampler applies
t-preconditioned Crank-Nicolson to the simple distribution. This method is able to scale to extremely high
dimensions, often outperforming gradient-based samplers, assuming that the target distribution can be
efficiently mapped to a simple distribution using a normalizing flow.


Is the normalizing flow used as an emulator for the posterior? 
--------------------------------------------------------------
No, the normalizing flow is used as a preconditioner, meaning that it is used to transform the target distribution
into a simple distribution. The normalizing flow is not used as an emulator for the posterior. The sampler still 
samples from the target distribution, but it does so by sampling from the simple distribution and then transforming
the samples back to the target distribution using the inverse of the normalizing flow. This allows the sampler to
take advantage of the symmetries of the simple distribution to explore the target distribution more efficiently.


When does the sampling terminate?
---------------------------------
The sampling terminates when the effective inverse temperature parameter ``beta`` reaches 1.0 and the effective 
sample size (ESS) exceeds the predefined threshold (``n_total=4096`` by default). The effective inverse temperature 
parameter ``beta`` is a measure of how close the sampler is to the posterior distribution.


Can I use pocoMC to sample from a target distribution without normalizing flow preconditioning?
-----------------------------------------------------------------------------------------------
Yes, you can use pocoMC to sample from a target distribution without normalizing flow preconditioning. In this case,
the sampler will sample directly from the target distribution using t-preconditioned Crank-Nicolson. This can be useful
if the target distribution is already simple and does not require normalizing flow preconditioning. However, in general,
we recommend using normalizing flow preconditioning, as it can significantly improve the efficiency of the sampler.


How many effective particles should I use?
------------------------------------------
It depends. The number of effective particles that you should use depends on the complexity of the target distribution
and the computational resources available. In general, we recommend using as many effective particles as possible, as this
will improve the efficiency of the sampler. However, the number of effective particles that you can use is limited by the
computational resources available. In practice, we have found that using 512 effective particles is often sufficient
to sample from most target distributions efficiently. However, you may need to experiment with different numbers of 
effective particles to find the optimal number for your problem.


How many active particles should I use?
---------------------------------------
No more than half of the effective particles. The number of active particles that you should use depends on the number of
effective particles that you are using. In general, we recommend using no more than half of the effective particles as active
particles. For example, if you are using 512 effective particles, then you should use no more than 256 active particles. Using
more active particles introduces correlations between the particles, which can reduce the efficiency of the sampler.


How do I know if the sampler is working correctly?
--------------------------------------------------
There are several ways to check if the sampler is working correctly. One way is to run the sampler with two sets of settings,
one more conservative than the other. If the results are consistent between the two runs, then the sampler is likely working
correctly. For instance, you can run the sampler with 512 effective particles and 256 active particles, and then run the sampler
with 256 effective particles and 128 active particles. If the results are consistent between the two runs, then the sampler is
likely working correctly.


Are there any indications that the sampler is not working correctly?
--------------------------------------------------------------------
Yes, there are a few indications that the sampler is not working correctly. One indication is that the acceptance rate of the 
Markov kernel is too low. If the acceptance rate is too low, then the sampler is not exploring the target distribution efficiently.
Under normal circumstances, the acceptance rate (``acc`` in the progress bar) should be around 0.2-0.8. Another indication that the 
sampler is not working correctly is that the efficiency of the sampler is too low. If the efficiency of the sampler is too low, then 
the sampler is not exploring the target distribution efficiently. Under normal circumstances, the efficiency (``eff`` in the progress 
bar) of the sampler should be around 0.1-1.0. Finally, another indication that the sampler is not working correctly is that the samples 
are not consistent between runs. If the samples are not consistent between runs, then the sampler is not exploring the target distribution efficiently.


Where does the name pocoMC come from?
-------------------------------------
The name pocoMC comes from the Spanish and Italian word "poco", which means "little" or "few". The name pocoMC was chosen because the sampler
uses a small number of particles to explore the target distribution efficiently. The name pocoMC is also a play on the word "poco",
which shares some common sounds with the word "preconditioned". Finally, the name was inspired by the name of the developer's cat, Poco.