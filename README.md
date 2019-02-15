# Augusta

Incorporating adversarial objective in quest for empowerment.

In addition, this repo serves as a way of reproducing the paper entitled [Variational Information Maximization  for Intrinsically Motivated Reinforcement Learning](http://papers.nips.cc/paper/5668-variational-information-maximisation-for-intrinsically-motivated-reinforcement-learning.pdf).

## TODO

### MINE
* [X] Unbiased gradient estimate
* [X] Reproduce the source policy from var-info-max
    * [X] 2 step
    * [X] 5 step
* [ ] Use policy gradient to derive source policy


### Variational Information Maximization
* [X] Make basic environments that will allow for testing
* [X] Implement basic empowerment estimation from paper
    * [X] Decoder to predict which sequence of actions was taken (log-likelihood)
    * [X] Source network
    * [X] Verify the empowerment computed is correct for at least one single state
* [X] Implement true empowerment algorithm (Blahut-Arimoto)
* [X] Expand the documentation in the repo
* [X] Run the maps with longer sequences

* [ ] Implement in the continuous case
    * inverted pendulum could be a good start

## Objective

Reproduce some of the results posted in order to be able to take it as a starting point to build a new approach to compute the empowerment.

## Proposed approach

1. inverse model of action sequence (_tl;dr predict action out of states w/ log-likelihood_)
1. normalized distribution prediction with two models (_tl;dr predict log-likelihood w/ MSE_)
1. compute empowerment with the normalization constant estimator $\phi$ from step \#2.

**They also train a feature extractor on top of all this, however it doesn't change much of the general approach and isn't required for discrete states**

## Environments

### Discrete and deterministic environments

Some simplified environments were used in the original paper. These same environments are implemented in [this file](src/custom_envs/static_envs.py) using an api like OpenAI's `gym`.

## Blahut-Arimoto algorithm

As a way of comparing the estimates of the empowerment, we need a true measure of it. To achieve this, the Blahut-Arimoto (BA) algorithm can be used to compute it.

Conceptually speaking, the BA algorithm is an EM algorithm that initializes the underlying action distribution (behavior policy) to be uniform and iteratively maximizes it in order to find the one that maximizes the mutual information. After _running_ the algorithm, we are left with both an estimate of the empowerment value as well as the underlying action distribution which maximizes the mutual information.

The paper [Empowerment for Continuous Agent-Environment Systems](https://arxiv.org/abs/1201.6583) details grealy how to run the algorhtm. The implementation used in this repo can be found in [here](src/blahut_arimoto.py).

### Deterministic environment

Given a simplified environment with deterministic state transitions, the computation of the Blahut-Arimoto algorithm can be simplified greatly. In particular, when running the expectation step of the Blahut-Arimoto algorithm, the marginal distribution of reachable states given the current location needs to be computed over the possible sequences of actions.

Computing that marginal could be very expensive in an environment with stochastic state transitions, as all of the possible ways of reaching another state would need to be explored during the computation. Thankfully, in this simplified setting with deterministic (static) environments, the dynamics of the transitions can be exploited to make the marginal easier to compute.

Indeed, given the ultimate state to be reached `s'`, the marginal probability of reaching that state from origin state `s` is the sum of the probabilities of the action sequences (under the behavior policy, i.e. source distribution) that allow to reach that ultimate state `s'`. This simplifies greatly the computatation since we only need to make deterministic rollouts at each state and use that information to find which state can lead to others.

Considering the simplified environment with a reasonable state and action size, all possible rollouts can be computed with moderate effort. This in turn makes it easy to determine all the possible paths that lead to a given state, which is the most computationally intensive part of the algorithm.
