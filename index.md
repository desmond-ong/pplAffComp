---
layout: single
classes: wide
title: Integrating Theory-Driven and Data-Driven Approaches to Affective Computing via Deep Probabilistic Programming
---


### Preamble:
This [tutorial](https://desmond-ong.github.io/pplAffComp) will be taught at <i>[Affective Computing and Intelligent Interaction 2019](http://acii-conf.org/2019/tutorials_accepted/)</i>, in September 2019 in Cambridge, UK. This tutorial is in turn based off material introduced in the following paper:

Ong, D. C., Soh, H., Zaki, J., & Goodman, N. D. (in press). Applying Probabilistic Programming to Affective Computing. <i>IEEE Transactions on Affective Computing</i> <br> [ [arXiv](https://arxiv.org/abs/1903.06445) ]


Tutorial Details.

- Location: Computer Laboratory, University of Cambridge. William Gates Building, 15 JJ Thomson Ave.
- Date and Time: Tuesday 3 September 2019, 9am-12pm.


Presenter: [Desmond C. Ong](https://web.stanford.edu/~dco)

Contributors:

- [Desmond C. Ong](https://web.stanford.edu/~dco), National University of Singapore (NUS) & Agency for Science, Technology and Research (A\*STAR)
- Zhi-Xuan Tan, Agency for Science, Technology and Research (A\*STAR); now at Massachusetts Institute of Technology (MIT)
- [Harold Soh](https://haroldsoh.com/), National University of Singapore (NUS)
- [Jamil Zaki](http://ssnl.stanford.edu/people), Stanford University
- [Noah D. Goodman](http://cocolab.stanford.edu/ndg.html), Stanford University



## Abstract

Research in affective computing has traditionally fallen into either theory-driven approaches that may not scale well to the complexities of naturalistic data, or atheoretic, data-driven approaches that learn to recognize complex patterns but still fall short of reasoning about emotions. In this tutorial, we introduce deep probabilistic programming, a new paradigm that models psychologically-grounded theories of emotion using stochastic programs. Specifically, this flexible framework combines the benefits of probabilistic models with those of deep learning models, marrying the advantages of both approaches. For example, when modelling someone’s emotions in context, we may choose to compose a probabilistic model of emotional appraisal with a deep learning model for recognizing emotions from faces&mdash;and we can do this all within a single unified framework for training and performing inference. By leveraging modern advances in deep probabilistic programming languages, researchers can easily scale these models up to larger, naturalistic datasets. Additionally, the lowered cost of model-building will accelerate rigorous theory-building and hypothesis-testing between competing theories of emotion.


The target audience comprises two groups of researchers. The first group includes researchers, such as cognitive psychologists, who favor theoretically-grounded models of affective phenomena, and who wish to scale their models up to complex, naturalistic datasets. The second group includes computer scientists, especially those who primarily use deep learning, who wish to add more emotion theory into their deep learning models, and in a principled manner. Deep probabilistic programming offers a way to combine the benefits of these two approaches to affective computing.


We will be learning from a webbook, using Jupyter notebooks. We will start with introductory primers to probabilistic programming concepts, such as stochastic primitives; compositionality and recursion; and stochastic variational inference. We will then transition into worked examples on previously-collected affective computing datasets. We will be using the open-sourced deep probabilistic programming language [Pyro](https://pyro.ai), first released in 2017. Tutorial participants will be able to download and run the code on their local machines as they follow along the material. We hope that by the end of this short tutorial, participants will be inspired&mdash;and equipped with some basic skills&mdash;to adopt approaches like deep probabilistic programming that merge theory- and data-driven approaches, and that such efforts will lead to greater collaboration between these historically-distinct paradigms in affective computing.

_Edit_: We will now be using Google Colaboratory (Colab), which provides a Jupyter environment on the cloud, hosted by Google. This means that participants do not need to download any code onto their computers to follow along.



## Table of Contents


- If you want to set up Pyro and this tutorial code on your laptop, refer to [[Getting Started](https://github.com/desmond-ong/pplAffComp/blob/master/gettingStarted.md)]. Otherwise, we'll be using Google Colab which will let you run all the Pyro code in the browser!
- Introduction to Probabilistic Programming (Taken from the Pyro tutorials)
    - Intro to Pyro [Part 1](http://pyro.ai/examples/intro_part_i.html) and [Part 2](http://pyro.ai/examples/intro_part_ii.html).
    - [Intro to Variational Inference](http://pyro.ai/examples/svi_part_i.html)
- Rational Speech Acts examples:
    - [Implicature](code/RSA-implicature.ipynb)
    - [Hyperbole](code/RSA-hyperbole.ipynb)

Affective Computing relevant examples:

- Example 1 [[Colab Notebook](https://colab.research.google.com/github/desmond-ong/pplAffComp/blob/master/Colab/PPLTutorial_1_LinearRegression.ipynb)]: Linear Regression as a model of Appraisal.
- Example 2 [[Colab Notebook](https://colab.research.google.com/github/desmond-ong/pplAffComp/blob/master/Colab/PPLTutorial_2_VAE.ipynb)]: Variational Autoencoder to generate emotional faces.
<!-- - Example 3 [[Colab Notebook](https://colab.research.google.com/github/desmond-ong/pplAffComp/blob/master/Colab/PPLTutorial_3_SSVAE.ipynb)]: Semi-supervised VAE to learn to recognize emotions from faces. -->
- Example 3 Colab Notebook: Semi-supervised VAE to learn to recognize emotions from faces. (Currently buggy, sorry!)
- Example 4 [[Colab Notebook](https://colab.research.google.com/github/desmond-ong/pplAffComp/blob/master/Colab/PPLTutorial_4_MVAE.ipynb)]: Multimodal VAE to model latent affect.


These examples were tested on Pyro version 0.4.1


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

