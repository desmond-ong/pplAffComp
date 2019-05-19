# Integrating Theory-Driven and Data-Driven Approaches to Affective Computing via Deep Probabilistic Programming

Instructor: [Desmond C. Ong](https://web.stanford.edu/~dco)

Contributors: [Desmond C. Ong](https://web.stanford.edu/~dco), Zhi-Xuan Tan, Harold Soh, Jamil Zaki, & Noah D. Goodman.

### Preamble:
This [tutorial](https://desmond-ong.github.io/pplAffComp) will be taught at <i>[Affective Computing and Intelligent Interaction 2019](http://acii-conf.org/2019/)</i>, in September 2019 in Cambridge, UK. This tutorial is in turn based off material introduced in the following paper:

Ong, D. C., Soh, H., Zaki, J., & Goodman, N. D. (in press). Applying Probabilistic Programming to Affective Computing. <i>IEEE Transactions on Affective Computing</i> <br> [ [arXiv](https://arxiv.org/abs/1903.06445) ]



## Abstract

Research in affective computing has traditionally fallen into either theory-driven approaches that may not scale well to the complexities of naturalistic data, or atheoretic, data-driven approaches that learn only to recognize patterns but fall short of being able to reason about emotions.

In this tutorial, we introduce deep probabilistic programming, a new paradigm that models psychologically-grounded theories of emotion using stochastic programs. Specifically, this flexible framework combines the benefits of probabilistic models with those of deep learning models, marrying the advantages of both approaches. For example, when modelling someone’s emotions in context, we may choose to compose a probabilistic model of emotional appraisal with a deep learning model for recognizing emotions from faces&mdash;and we can do this all within a single unified framework for training and performing inference. By leveraging modern advances in deep probabilistic programming languages, researchers can easily scale these models up to larger, naturalistic datasets. Additionally, the lowered cost of model-building will accelerate rigorous theory-building and hypothesis-testing between competing theories of emotion.


The target audience will comprise researchers from two groups. The first group includes researchers, such as cognitive psychologists, who tend to favor theory-grounded models of affective phenomena, and who wish to scale their models up to more naturalistic datasets. The second group includes computer scientists, especially those who primarily use deep learning, who wish to add more emotion theory into their deep learning models, and in a principled manner. Deep probabilistic programming offers a way to combine the benefits of these two approaches to affective computing.


We will be learning from a webbook, using Jupyter notebooks. We will start with introductory primers to probabilistic programming concepts, such as stochastic primitives; compositionality and recursion; and stochastic variational inference. We will then transition into worked examples on previously-collected affective computing datasets. We will be using the open-sourced deep probabilistic programming language [Pyro](https://pyro.ai), first released in 2017. Tutorial participants will be able to download and run the code on their local machines as they follow along the material. We hope that by the end of this short tutorial, participants will be inspired&emdash;and equipped with some basic skills&emdash;to adopt approaches like deep probabilistic programming that merge theory- and data-driven approaches, and that this effort will lead to greater collaboration between these historically-distinct paradigms in affective computing.




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

