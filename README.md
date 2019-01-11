# Variational Information Maximization

This serves as a repo for reproducing the paper entitled [Variational Information Maximization  for Intrinsically Motivated Reinforcement Learning](http://papers.nips.cc/paper/5668-variational-information-maximisation-for-intrinsically-motivated-reinforcement-learning.pdf)

# TODO
* [x] Make basic environments that will allow for testing
* [ ] Implement basic empowerment estimation from paper
    * [X] Decoder to predict which sequence of actions was taken (log-likelihood)
    * [X] Source network
    * [ ] Verify the empowerment computed is correct for at least one single state
* [ ] Set up makefile for easy launch and clean of runs
* [ ] Implement true empowerment algorithm [see this section](#blahut-arimoto-algorithm)

# Objective

Reproduce some of the results posted in order to be able to take it as a starting point to build a new approach to compute the empowerment.

# Proposed approach

1. inverse model of action sequence (_tl;dr predict action out of states w/ log-likelihood_)
1. normalized distribution prediction (_tl;dr predict log-likelihood w/ MSE_)

Compute empowerment with resulting function which is learned inside step \#2 above.

**They also train a feature extractor on top of all this, however it doesn't change much of the general approach**


# Blahut-Arimoto algorithm

As a way of comparing the estimates of the empowerment, we need a true measure of it. To achieve this, the Blahut-Arimoto (BA) algorithm can be used to compute it.

Conceptually speaking, the BA algorithm initializes the underlying action distribution to be uniform and iteratively maximizes it in order to find the one that maximizes the mutual information.

After _running_ the algorithm, we are left with both an estimate of the empowerment value as well as the action distribution which maximizes the mutual information.
