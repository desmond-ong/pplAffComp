import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import pyro
import pyro.distributions as dist

from networks import ImageEncoder, ImageDecoder, Encoder, Decoder

class ProductOfExperts(nn.Module):
    """
    Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param loc: M x D for M experts
    @param scale: M x D for M experts
    """
    def forward(self, loc, scale, eps=1e-8):
        scale = scale + eps # numerical constant for stability
        # precision of i-th Gaussian expert (T = 1/sigma^2)
        T = 1. / scale
        product_loc = torch.sum(loc * T, dim=0) / torch.sum(T, dim=0)
        product_scale = 1. / torch.sum(T, dim=0)
        return product_loc, product_scale

class MVAE(nn.Module):
    """
    This class encapsulates the parameters (neural networks), models & guides
    needed to train a multimodal variational auto-encoder.
    Modified from https://github.com/mhw32/multimodal-vae-public
    Multimodal Variational Autoencoder.

    @param z_dim: integer
                  size of the tensor representing the latent random variable z
                  
    Currently all the neural network dimensions are hard-coded; 
    in a future version will make them be inputs into the constructor
    """
    def __init__(self, z_dim, img_width, embed_dim,
                 rating_dim, outcome_dim, use_cuda=False):
        super(MVAE, self).__init__()
        self.z_dim = z_dim
        self.img_width = img_width
        self.image_encoder = ImageEncoder(z_dim)
        self.image_decoder = ImageDecoder(z_dim)
        self.embed_dim = embed_dim
        self.word_encoder = Encoder(z_dim, embed_dim)
        self.word_decoder = Decoder(z_dim, embed_dim)
        self.rating_dim = rating_dim
        self.rating_encoder = Encoder(z_dim, rating_dim)
        self.rating_decoder = Decoder(z_dim, rating_dim)
        self.outcome_dim = outcome_dim
        self.outcome_encoder = Encoder(z_dim, outcome_dim)
        self.outcome_decoder = Decoder(z_dim, outcome_dim)
        self.experts = ProductOfExperts()
        self.use_cuda = use_cuda
        
        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()

    def prior_expert(self, size, use_cuda=False):
         """Universal prior expert. Here we use a spherical
         Gaussian: N(0, 1).
         @param size: integer
                      dimensionality of Gaussian
         @param use_cuda: boolean [default: False]
                          cast CUDA on variables
         """
         mu = Variable(torch.zeros(size))
         logvar = Variable(torch.log(torch.ones(size)))
         if use_cuda:
             mu, logvar = mu.cuda(), logvar.cuda()
         return mu, logvar
            
    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            return mu

    def forward(self, word=None, image=None, rating=None, outcome=None):
        mu, logvar  = self.infer(word, image, rating, outcome)
        # reparametrization trick to sample
        z  = self.reparametrize(mu, logvar)
        # reconstruct inputs based on that gaussian
        word_recon = self.word_decoder(z)
        image_recon = self.image_decoder(z)
        rating_recon = self.rating_decoder(z)
        outcome_recon = self.outcome_decoder(z)
        return word_recon, image_recon, rating_recon, outcome_recon, mu, logvar

    def infer(self, word=None, image=None, rating=None, outcome=None):
        if word is not None:
            batch_size = word.size(0)
        elif image is not None:
            batch_size = image.size(0)
        elif rating is not None:
            batch_size = rating.size(0)
        elif outcome is not None:
            batch_size = outcome.size(0)

        batch_size = 1

        # initialize the universal prior expert
        mu, logvar = self.prior_expert((1, batch_size, self.z_dim),
                                       use_cuda=self.use_cuda)
        if word is not None:
            word_mu, word_logvar = self.word_encoder(word)
            mu = torch.cat((mu, word_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, word_logvar.unsqueeze(0)), dim=0)
        
        if image is not None:
            image_mu, image_logvar = self.image_encoder(image)
            mu = torch.cat((mu, image_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, image_logvar.unsqueeze(0)), dim=0)

        if rating is not None:
            rating_mu, rating_logvar = self.rating_encoder(rating)
            mu = torch.cat((mu, rating_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, rating_logvar.unsqueeze(0)), dim=0)

        if outcome is not None:
            outcome_mu, outcome_logvar = self.outcome_encoder(outcome)
            mu     = torch.cat((mu, outcome_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, outcome_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar
    
    def model(self, words=None, images=None, ratings=None, outcomes=None):
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("mvae", self)
        
        batch_size = 0
        if words is not None:
            batch_size = words.size(0)
        elif images is not None:
            batch_size = images.size(0)
        elif ratings is not None:
            batch_size = ratings.size(0)
        elif outcomes is not None:
            batch_size = outcomes.size(0)
        
        with pyro.iarange("data", batch_size):
            if outcomes is not None:
                # sample from outcome prior, compute p(z|outcome)
                outcome_prior_loc = torch.zeros(\
                        torch.Size((batch_size, self.outcome_dim)))
                outcome_prior_scale = torch.ones(\
                        torch.Size((batch_size, self.outcome_dim)))
                pyro.sample("obs_outcome",
                            dist.Normal(outcome_prior_loc,
                                        outcome_prior_scale).independent(1),
                            obs=outcomes.reshape(-1, self.outcome_dim))
                
                z_loc, z_scale = self.outcome_encoder.forward(outcomes)
            else:
                # setup hyperparameters for prior p(z)
                z_loc = torch.zeros(torch.Size((batch_size, self.z_dim)))
                z_scale = torch.ones(torch.Size((batch_size, self.z_dim)))
            
            # sample from prior
            # (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent",
                            dist.Normal(z_loc, z_scale).independent(1))
            # decode the latent code z

            word_loc, word_scale = self.word_decoder.forward(z)
            # score against actual words
            if words is not None:
                pyro.sample("obs_word",
                            dist.Normal(word_loc, word_scale).independent(1), 
                            obs=words.reshape(-1, self.embed_dim))
            
            img_loc = self.image_decoder.forward(z)
            # score against actual images
            if images is not None:
                pyro.sample("obs_img",
                            dist.Bernoulli(img_loc).independent(1), 
                            obs=images.reshape(-1, 3,
                                               self.img_width, self.img_width))
            
            rating_loc, rating_scale = self.rating_decoder.forward(z)
            # score against actual ratings
            if ratings is not None:
                pyro.sample("obs_rating",
                            dist.Normal(rating_loc,
                                        rating_scale).independent(1), 
                            obs=ratings.reshape(-1, self.rating_dim))

            # return the loc so we can visualize it later
            return word_loc, img_loc, rating_loc
    
    def guide(self, words=None, images=None, ratings=None, outcomes=None):
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("mvae", self)
        
        batch_size = 0
        if words is not None:
            batch_size = words.size(0)
        elif images is not None:
            batch_size = images.size(0)
        elif ratings is not None:
            batch_size = ratings.size(0)
        elif outcomes is not None:
            batch_size = outcomes.size(0)
            
        with pyro.iarange("data", batch_size):
            # use the encoder to get the parameters used to define q(z|x)
                        
            # initialize the prior expert
            # the additional dimension (1) is to 
            z_loc = torch.zeros(torch.Size((1, batch_size, self.z_dim)))
            z_scale = torch.ones(torch.Size((1, batch_size, self.z_dim)))
            if self.use_cuda:
                z_loc, z_scale = z_loc.cuda(), z_scale.cuda()
            
            # figure out the elbo loss? encoder/decoder?
            if outcomes is not None:
                outcome_z_loc, outcome_z_scale =\
                    self.outcome_encoder.forward(outcomes)
                z_loc = torch.cat((z_loc, outcome_z_loc.unsqueeze(0)), dim=0)
                z_scale = torch.cat((z_scale, outcome_z_scale.unsqueeze(0)),
                                    dim=0)

            if words is not None:
                word_z_loc, word_z_scale =\
                        self.word_encoder.forward(words)
                z_loc = torch.cat((z_loc, word_z_loc.unsqueeze(0)), dim=0)
                z_scale = torch.cat((z_scale, word_z_scale.unsqueeze(0)),
                                    dim=0)                
                
            if images is not None:
                image_z_loc, image_z_scale =\
                    self.image_encoder.forward(images)
                z_loc = torch.cat((z_loc, image_z_loc.unsqueeze(0)), dim=0)
                z_scale = torch.cat((z_scale, image_z_scale.unsqueeze(0)),
                                    dim=0)
            
            if ratings is not None:
                rating_z_loc, rating_z_scale =\
                    self.rating_encoder.forward(ratings)
                z_loc = torch.cat((z_loc, rating_z_loc.unsqueeze(0)), dim=0)
                z_scale = torch.cat((z_scale, rating_z_scale.unsqueeze(0)),
                                    dim=0)
            
            z_loc, z_scale = self.experts(z_loc, z_scale)
            # sample the latent z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))
    
