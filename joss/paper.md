---
title: 'pocoMC: A Python package for accelerated Bayesian inference in astronomy and cosmology'
tags:
  - Python
  - astronomy
authors:
  - name: Minas Karamanis
    orcid: 0000-0001-9489-4612
    corresponding: true
    affiliation: 1
  - name: David Nabergoj
    orcid: 0000-0001-6882-627X
    affiliation: 2
  - name: Florian Beutler
    orcid: 0000-0003-0467-5438
    affiliation: 1
  - name: John A. Peacock
    orcid: 0000-0002-1168-8299
    affiliation: 1
  - name: Uros Seljak
    orcid: 0000-0003-2262-356X
    affiliation: 3

affiliations:
 - name: Institute for Astronomy, University of Edinburgh, Royal Observatory, Blackford Hill, Edinburgh EH9 3HJ, UK
   index: 1
 - name: Faculty of Computer and Information Science, University of Ljubljana, Ve\v{c}na pot 113, 1000 Ljubljana, Slovenia
   index: 2
 - name: Physics Department, University of California and Lawrence Berkeley National Laboratory Berkeley, CA 94720, USA
   index: 3
date: 29 June 2022
bibliography: paper.bib

---

# Summary

`pocoMC` is a Python package for accelerated Bayesian inference in astronomy and 
cosmology. It facilitates both tasks of parameter estimation and model comparison. 
It allows fitting arbitrary models defined as a log-likelihood function and a 
log-prior probability density function in Python. `pocoMC` is designed to sample 
efficiently from  posterior distributions with non-trivial geometry, including 
strong multimodality and non-linearity. Parallelisation to computing clusters 
manifests linear scaling.

# Statement of need

Over the past few decades the volume of astronomical and cosmological data has 
increased substantially. At the same time, theoretical and phenomenological models
in these fields have grown even more complex. As a response to that, a number of methods
aiming at efficient Bayesian computation have been developed with the sole task of
comparing those models to the available data [@trotta2017bayesian; @sharma2017markov]. 
In the Bayesian context, scientific inference proceeds though the use of Bayes' theorem:
\begin{equation}\label{eq:bayes}
\mathcal{P}(\theta) = \frac{\mathcal{L}(\theta)\pi(\theta)}{\mathcal{Z}}
\end{equation}
where the posterior $\mathcal{P}(\theta)\equiv p(\theta\vert d,\mathcal{M})$ is the
probability of the parameters $\theta$ given the data $d$ and the model $\mathcal{M}$.
The other components of this equation are: the likelihood function 
$\mathcal{L}(\theta)\equiv p(d\vert \theta,\mathcal{M})$, the prior $\pi(\theta) \equiv p(\theta\vert \mathcal{M})$,
and the model evidence $\mathcal{Z}=p(d\vert \mathcal{M})$. The prior and the
likelihood are usually provided as input in this equation and one seeks to estimate the 
posterior and the evidence. Knowledge of the posterior, in the form of samples, 
is paramount for the task of parameter estimation whereas the ratio of model 
evidences yields the Bayes factor which is the cornerstone of Bayesian model comparison.

Markov chain Monte Carlo (MCMC) has been established as the standard tool for 
Bayesian computation in astronomy and cosmology, either as a standalone algorithm
or as part of another method (e.g. nested sampling [@skilling2006nested]). However, 
as MCMC relies on the local exploration of the posterior, the presense of non-linear
correlation between parameters and multimodality can at best hinder its performance
and at worst violate its theoretical guarantees of convergence (i.e. ergodicity). Usually
those challenges are partially addressed by reparameterising the model using a common change-of-variables parameter transformation. However, the practical effect of this often limited and usually
requires expert knowledge. These problems are usually complicated further by the substantial
computational cost of evaluating astronomical and cosmological models. `pocoMC` is 
designed to tackle exactly these kinds of difficulties by automatically reparameterising
the model such that the parameters of the model are approximately uncorrelated and standard techniques 
can be applied. As a result, `pocoMC` produces both samples from the posterior distribution and an
unbiased estimate of the model evidence thus facilitating both scientific tasks with excellent 
efficiency and robustness.

![Logo of `pocoMC`.](./../logo.png)

# Method

`pocoMC` implements the Preconditioned Monte Carlo (PMC) algorithm. PMC combines
the popular Sequential Monte Carlo (SMC) [@del2006sequential] method with a Normalising Flow (NF) [@papamakarios2021normalizing]. 
The latter works as a preconditioner for the target distribution of the former. 
As SMC evolves a population of particles, starting from the prior distribution 
and gradually approaching the posterior distribution, the NF transforms the 
parameters of the target distribution such that any correlation between parameters
or presence of multimodality is removed. The effect of this bijective transformation
is the substantial rise in the sampling efficiency of the algorithm as the particles
are allowed to sample freely from the target without being hindered by its locally-curved 
geometry. The method is explained in detail in the accompanying publication [@karamanis2022pmc]
and we provide only a short summary here.

## Sequential Monte Carlo

The basic idea of basic SMC is to sample from the posterior distribution $\mathcal{P}(\theta)$ by first
defining a path of intermediate distributions starting from the prior $\pi(\theta)$. In the
case of `pocoMC` the path has the form:
\begin{equation}\label{eq:path}
p_{t}(\theta) = \pi(\theta)^{1-\beta_{t}} \mathcal{P}(\theta)^{\beta_{t}}
\end{equation}
where $0=\beta_{1}<\beta_{2}<\dots<\beta_{T}=1$. Starting from the prior, each distribution with density $p_{t}(\theta)$ is
sampled in turn using a collection of particles propaged by a number of MCMC steps. Prior to MCMC sampling,
the particles are re-weighted using importance sampling and then re-sampled to account for the transition from
$p_{t}(\theta)$ to $p_{t+1}(\theta)$. `pocoMC` utilises the importance weights of this step to define an estimator
for the effective sample size (ESS) of the population of particles. Maintaining a fixed value of ESS during the run
allows `pocoMC` to adaptively specify the $\beta_{t}$ schedule.

## Preconditioned Monte Carlo

In vanilla SMC, standard MCMC methods (e.g. Metropolis-Hastings) are used to update the positions
of the particles during each iteration. This however can become highly inefficient if the distribution
$p_{t}(\theta)$ is characterised by a non-trivial geometry. `pocoMC`, which is based on PMC, utilises
a NF to learn an invertible transformation that simplifies
the geometry of the distribution by mapping $p_{t}(\theta)$ into a zero-mean unit-variance normal distribution.
Sampling then proceeds in the latent space in which correlations are substantially reduced. The positions of
the particles are transformed back to the original parameter space at the end of each iteration. This way,
PMC and `pocoMC` are able to sample from very challenging posteriors very efficiently using simple Metropolis-Hastings
updates in the preconditioned/uncorrelated latent space.

# Features

- User-friendly black-box API
- Only the log-likelihood, log-prior and some prior samples are required from the user
- No tuning is required but is possible for experienced users
- Default configuration sufficient for most applications
- Posterior corner, trace, and run plotting tools
- Support for both MAF and RealNVP normalising flows with added regularisation [@papamakarios2017masked; @dinh2016density]
- Straightforward parallelisation using MPI or multiprocessing
- Wide range of examples of using the code
- Continuous integration and unit tests available
- [Extensive documentation available](http://pocomc.readthedocs.io)

# Acknowledgements

We acknowledge contributions from X, Y, Z.

# References