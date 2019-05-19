# Integrating Theory-Driven and Data-Driven Approaches to Affective Computing via Deep Probabilistic Programming

Instructor: [Desmond C. Ong](https://web.stanford.edu/~dco)

Contributors: Zhi-Xuan Tan, Harold Soh, Jamil Zaki, & Noah D. Goodman.

### Preamble:
This [tutorial](https://desmond-ong.github.io/pplAffComp) will be taught at <i>[Affective Computing and Intelligent Interaction 2019](http://acii-conf.org/2019/)</i>, in September 2019 in Cambridge, UK. This tutorial is in-turn based off material introduced in the following paper:

Ong, D. C., Soh, H., Zaki, J., & Goodman, N. D. (in press). Applying Probabilistic Programming to Affective Computing. <i>IEEE Transactions on Affective Computing</i> <br> [ [arXiv](https://arxiv.org/abs/1903.06445) ]



## Abstract

Research in affective computing has traditionally fallen into either theory-driven approaches that do not scale well to naturalistic data, or atheoretic, data-driven approaches that learns only to recognize patterns. In this tutorial, we introduce deep probabilistic programming, a new paradigm that models psychologically-grounded theories of emotion using stochastic programs. Specifically, this framework is flexible enough to combine the benefits of probabilistic models with those of deep learning models, marrying the advantages of each approach. For example, when modelling someone’s emotions in context, we may choose to compose a probabilistic model of emotional appraisal with a deep learning model for recognizing emotions from faces—all within a single unified framework for training and performing inference. By leveraging modern advances in deep probabilistic programming languages, researchers can easily scale these models up to larger, naturalistic datasets. Additionally, the lowered cost of building these models will allow rigorous theory-building and hypothesis testing between competing theories of emotion. We will be teaching from a webbook that contains worked examples and executable code on a previously-collected dataset, which tutorial participants will be able to download and run on their local machine. We hope that participants will be inspired to adopt approaches that merge theory- and data-driven approaches in affective computing.




## Table of Contents


- [Getting Started](gettingStarted.md)
- Introduction to Probabilistic Programming (Taken from the Pyro tutorials)
    - Intro to Pyro [Part 1](http://pyro.ai/examples/intro_part_i.html) and [Part 2](http://pyro.ai/examples/intro_part_ii.html).
    - [Intro to Variational Inference](http://pyro.ai/examples/svi_part_i.html)
- Rational Speech Acts examples:
    - [Implicature](code/RSA-implicature.ipynb)
    - [Hyperbole](code/RSA-hyperbole.ipynb)

Affective Computing relevant examples:

- [Example 1: Linear Regression](code/LinearRegression.ipynb) as a model of Appraisal.
- [Example 2: Variational Autoencoder](code/VAE.ipynb) to generate emotional faces
- [Example 3: Semi-supervised VAE](code/SemiSupervisedVAE.ipynb) to learn to recognize emotions from faces
- Example 4: Multimodal VAE to model latent affect (Under construction! Will be up soon)


---
### Links to External Resources


- [Pyro Github](https://github.com/uber/pyro)
- [Pyro Tutorials](http://pyro.ai/)
- [WebPPL](http://webppl.org/): a probabilistic programming language written in Javascript
- [Probabilistic Models of Cognition](http://probmods.org/): a webbook on applying probabilistic modeling to human cognition




---
## Acknowledgments

This work was supported indirectly by a number of grants:

- the A\*STAR Human-Centric Artificial Intelligence Programme (SERC SSF Project No. A1718g0048)
- Singapore MOE AcRF Tier 1 (No. 251RES1709) to HS
- NIH Grant 1R01MH112560-01 to JZ.
- This material is based on research sponsored by DARPA under agreement number FA8750-14-2-0009.

