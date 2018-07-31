import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from custom_mlp import MLP, Exp

def BernoulliMLP(in_dim, out_dim, hidden_dims, allow_broadcast, use_cuda):
    """MLP that outputs the mean for a Bernoulli distribution."""
    return MLP([in_dim] + hidden_dims + [out_dim],
               activation=nn.ReLU,
               output_activation=nn.Sigmoid,
               allow_broadcast=allow_broadcast,
               use_cuda=use_cuda)

def NormalMLP(in_dim, out_dim, hidden_dims, allow_broadcast, use_cuda):
    """MLP that outputs the mean and variance of a Normal distribution."""
    return MLP([in_dim] + hidden_dims + [[out_dim, out_dim]],
               activation=nn.ReLU,
               output_activation=[None, Exp],
               allow_broadcast=allow_broadcast,
               use_cuda=use_cuda)

class SSVAE(nn.Module):
    """
    This class encapsulates the parameters (neural networks),
    models & guides needed to train a semi-supervised variational
    auto-encoder.
    
    Modified from
    https://github.com/uber/pyro/blob/dev/examples/vae/ss_vae_M2.py

    :param output_size: size of the tensor representing the outputs
    :param input_size: size of the tensor representing the inputs
    :param z_dim: size of the tensor representing the latent random variable z
    :param hidden_layers: a tuple (or list) of MultiLayer Perceptron (MLP)
                          layers to be used in the neural networks
                          representing the parameters of the distributions
                          in our model
    :param use_cuda: use GPUs for faster training
    :param aux_loss_multiplier: the multiplier to use with the auxiliary loss
    :param output_dist: either 'normal' or 'bernoulli' distribution
    :param input_dist: either 'normal' or 'bernoulli' distribution
    :param output_prior: parameters for output prior distribution
    """
    def __init__(self, output_size, input_size, z_dim, hidden_layers=[200],
                 config_enum=None, use_cuda=False, aux_loss_multiplier=None,
                 output_dist="normal", input_dist="normal",
                 output_prior=[0.5, 0.2]):
        super(SSVAE, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.allow_broadcast = config_enum == 'parallel'
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier
        self.output_dist = output_dist
        self.input_dist = input_dist
        self.output_prior = output_prior
        
        # encoder_y goes from inputs to outputs
        if output_dist == "normal":
            self.encoder_y =\
                NormalMLP(input_size, output_size, hidden_layers, 
                          self.allow_broadcast, use_cuda)
        else:
            self.encoder_y =\
                BernoulliMLP(input_size, output_size, hidden_layers, 
                             self.allow_broadcast, use_cuda)
            
        # encoder_z goes from [inputs, outputs] to z
        self.encoder_z =\
            NormalMLP(input_size + output_size, z_dim, hidden_layers,
                      self.allow_broadcast, self.use_cuda)

        # decoder goes from [z, outputs] to the inputs.
        if input_dist == "normal":
            self.decoder =\
                NormalMLP(z_dim + output_size, input_size,
                          hidden_layers[::-1], self.allow_broadcast, use_cuda)
        else:
            self.decoder =\
                BernoulliMLP(z_dim + output_size, input_size,
                             hidden_layers[::-1], self.allow_broadcast,
                             use_cuda)
        
        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()
            
    def model(self, xs, ys=None, beta=1.0):
        """
        The model corresponds to the following generative process:
        p(z)     = normal(0,I)      # Prior on the latent variable z
        p(y)     = normal(.5, .2)   # Default prior on the outputs
        p(x|y,z) = normal(decoder(y,z))

        :param xs: a batch of word embeddings
        :param ys: (optional) a batch of emotion ratings.
                   if ys is not provided, will treat as unsupervised,
                   sample from prior.
        :param beta: scale parameter that weights the KL divergence in the ELBO
                     also sometimes called annealing.
        :return: None
        """
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("ssvae", self)

        batch_size = xs.size(0)
        # inform Pyro that the variables in the batch of xs, ys
        # are conditionally independent
        with pyro.iarange("data"):

            # sample the latent z from the (constant) prior, z ~ Normal(0,I)
            z_prior_mean  = torch.zeros(size=[batch_size, self.z_dim])
            z_prior_scale = torch.ones(size=[batch_size, self.z_dim])
            if self.use_cuda:
                z_prior_mean = z_prior_mean.cuda()
                z_prior_scale = z_prior_scale.cuda()
            z_prior = dist.Normal(z_prior_mean, z_prior_scale).independent(1)
            with poutine.scale(scale=beta):
                zs = pyro.sample("z", z_prior)

            # if the output is not provided, sample from prior,
            # otherwise, observe the value (i.e. score it against the prior)
            if self.output_dist == "normal":
                y_prior_mean  = torch.ones(size=[batch_size, self.output_size])
                y_prior_mean *= self.output_prior[0]
                y_prior_scale = torch.ones(size=[batch_size, self.output_size])
                y_prior_scale *= self.output_prior[1]
                if self.use_cuda:
                    y_prior_mean = y_prior_mean.cuda()
                    y_prior_scale = y_prior_scale.cuda()
                y_prior = dist.Normal(y_prior_mean,
                                      y_prior_scale).independent(1)
            else:
                y_prior_mean  = torch.ones(size=[batch_size, self.output_size])
                y_prior_mean *= self.output_prior[0]
                if self.use_cuda:
                    y_prior_mean = y_prior_mean.cuda()
                y_prior = dist.Bernoulli(y_prior_mean).independent(1)
            if ys is None:
                ys = pyro.sample("y", y_prior)
            else:
                ys = pyro.sample("y", y_prior, obs=ys)
                
            # Finally, we can condition on observing the input,
            #    using the latent z and emotion rating y in the 
            #    parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
            #    where decoder is a neural network

            if self.input_dist == "normal":
                x_mean, x_scale = self.decoder.forward([zs, ys])
                x_dist = dist.Normal(x_mean, x_scale).independent(1)
            else:
                x_mean = self.decoder.forward([zs, ys])
                x_dist = dist.Bernoulli(x_mean).independent(1)
            pyro.sample("x", x_dist, obs=xs)
            
            # return the parameters of the input distribution
            if self.input_dist == "normal":
                return x_mean, x_scale
            else:
                return x_mean

    def guide(self, xs, ys=None, beta=1.0):
        """
        The guide corresponds to the following:
        q(y|x)   = normal(encoder_y(x))  
        q(z|x,y) = normal(encoder_z(x,y)) 

        :param xs: a batch of word vectors
        :param ys: (optional) a batch of emotion ratings.
                   if ys is not provided, will treat as unsupervised
        :param beta: not used here, but left to match 
                     the call signature of self.model()
        :return: None
        """
        # inform Pyro that the variables in the batch of xs, ys
        # are conditionally independent
        with pyro.iarange("data"):

            # if the output is not provided, 
            #    sample with the variational distribution
            if ys is None:
                if self.output_dist == "normal":
                    y_mean, y_scale = self.encoder_y.forward(xs)
                    y_dist = dist.Normal(y_mean, y_scale).independent(1)
                else:
                    y_mean = self.encoder_y.forward(xs)
                    y_dist = dist.Bernoulli(y_mean).independent(1)
                ys = pyro.sample("y", y_dist)
                
            # Sample (and score) the latent z with the variational
            #   distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            #   where loc(.), scale(.) are given by encoder_z()
                        
            z_mean, z_scale = self.encoder_z.forward([xs, ys])
            with poutine.scale(scale=beta): 
                pyro.sample("z", dist.Normal(z_mean, z_scale).independent(1))

    def model_rating(self, xs, ys=None, beta=None):
        """
        This model is used to add an auxiliary (supervised) loss as
        described in Kingma et al. (2014),
        "Semi-Supervised Learning with Deep Generative Models".
        
        This is to ensure that the model learns from the supervised examples.
        q(y|x) = normal(encoder_y(x))
        
        :param xs:   word embedding
        :param ys:   emotion rating
        :param beta: not used here, but left to match the call
                     signature of self.model()
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("ssvae", self)

        # inform Pyro that the variables in the batch of xs, ys
        # are conditionally independent
        with pyro.iarange("data"):
            # this is the extra term to yield an auxiliary loss that
            # we do gradient descent on
            if ys is not None:
                if self.output_dist == "normal":
                    y_mean, y_scale = self.encoder_y.forward(xs)
                    y_dist = dist.Normal(y_mean, y_scale).independent(1)
                else:
                    y_mean = self.encoder_y.forward(xs)
                    y_dist = dist.Bernoulli(y_mean).independent(1)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample("y_aux", y_dist, obs=ys)

    def guide_rating(self, xs, ys=None, beta=None):
        """
        Dummy guide function to accompany model_rating() in inference
        This guide function is empty, because model_rating()
        has no latent random variables
        (i.e., model_rating() has no pyro.sample() calls that
        are not conditioned on observations)
        """
        pass
    
    def rate(self, xs):
        """
        Assign outputs (ys) to inputs (xs)

        :param xs: a batch of inputs
        :return:   a batch of the corresponding outputs (ys)
        """
        # use the trained model q(y|x) = normal(encoder_y(x))
        if self.output_dist == "normal":
            y_mean, y_scale = self.encoder_y.forward(xs)
        else:
            y_mean = self.encoder_y.forward(xs)
        return y_mean
    
    def reconstruct(self, x):
        """
        A helper function for reconstructing the input.
        """
        # This function assumes that x is a single vector, but since the
        # encoders and decoders take in batches, we have to resize x:
        xs = x.view(1, self.input_size)
        if self.output_dist == "normal":
            y_mean, y_scale = self.encoder_y.forward(xs)
            y_dist = dist.Normal(y_mean, y_scale).independent(1)
        else:
            y_mean = self.encoder_y.forward(xs)
            y_dist = dist.Bernoulli(y_mean).independent(1)
        ys = y_dist.sample()
        z_mean, z_scale = self.encoder_z.forward([xs, ys])
        # Sample in latent space
        zs = dist.Normal(z_mean, z_scale).sample()
        # Decode the word
        if self.input_dist == "normal":
            x_mean, x_scale = self.decoder.forward([zs, ys])
        else:
            x_mean = self.decoder.forward([zs, ys])
        return x_mean
