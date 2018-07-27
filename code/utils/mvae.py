import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

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
                 rating_dim, outcome_dim, use_cuda=False, lambdas={}):
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
        
        self.lambdas = lambdas
        if "image" not in self.lambdas:
            self.lambdas["image"] = 1.0
        if "word" not in self.lambdas:
            self.lambdas["word"] = 1.0
        if "rating" not in self.lambdas:
            self.lambdas["rating"] = 50.0
         if "outcome" not in self.lambdas:
            self.lambdas["outcome"] = 100.0
       
        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()
            self.image_encoder.cuda()
            self.image_decoder.cuda()
            self.word_encoder.cuda()
            self.word_decoder.cuda()
            self.rating_encoder.cuda()
            self.rating_decoder.cuda()
            self.outcome_encoder.cuda()
            self.outcome_decoder.cuda()
            self.experts.cuda()
            
    def forward(self, word=None, image=None, rating=None, outcome=None):
        z_loc, z_scale  = self.infer(word, image, rating, outcome)
        z = pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))
        word_recon = self.word_decoder(z)
        image_recon = self.image_decoder(z)
        rating_recon = self.rating_decoder(z)
        outcome_recon = self.outcome_decoder(z)
        return (word_recon, image_recon, rating_recon, outcome_recon,
                z_loc, z_scale)

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

        # initialize the universal prior
        z_loc = torch.zeros(torch.Size((1, batch_size, self.z_dim))) + 0.5
        z_scale = torch.ones(torch.Size((1, batch_size, self.z_dim))) * 0.1
        if self.use_cuda:
            z_loc, z_scale = z_loc.cuda(), z_scale.cuda()
        
        if word is not None:
            word_z_loc, word_z_scale = self.word_encoder(word)
            z_loc = torch.cat((z_loc, word_z_loc.unsqueeze(0)), dim=0)
            z_scale = torch.cat((z_scale, word_z_scale.unsqueeze(0)), dim=0)
        
        if image is not None:
            image_z_loc, image_z_scale = self.image_encoder(image)
            z_loc = torch.cat((z_loc, image_z_loc.unsqueeze(0)), dim=0)
            z_scale = torch.cat((z_scale, image_z_scale.unsqueeze(0)), dim=0)

        if rating is not None:
            rating_z_loc, rating_z_scale = self.rating_encoder(rating)
            z_loc = torch.cat((z_loc, rating_z_loc.unsqueeze(0)), dim=0)
            z_scale = torch.cat((z_scale, rating_z_scale.unsqueeze(0)), dim=0)

        if outcome is not None:
            outcome_z_loc, outcome_z_scale = self.outcome_encoder(outcome)
            z_loc = torch.cat((z_loc, outcome_z_loc.unsqueeze(0)), dim=0)
            z_scale = torch.cat((z_scale, outcome_z_scale.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        z_loc, z_scale = self.experts(z_loc, z_scale)
        return z_loc, z_scale
    
    def model(self, words=None, images=None, ratings=None, outcomes=None,
              annealing_beta=1.0):
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
            z_loc = torch.zeros(torch.Size((1, batch_size, self.z_dim))) + 0.5
            z_scale = torch.ones(torch.Size((1, batch_size, self.z_dim))) * 0.1
            if self.use_cuda:
                z_loc, z_scale = z_loc.cuda(), z_scale.cuda()
            
            # sample from prior
            # (value will be sampled by guide when computing the ELBO)
            z_dist = dist.Normal(z_loc, z_scale).independent(1)
            with poutine.scale(scale=annealing_beta):
                z = pyro.sample("latent", z_dist)

            # decode the latent code z
            
            word_loc, word_scale = self.word_decoder.forward(z)
            word_dist = dist.Normal(word_loc, word_scale).independent(1)
            # score against actual words
            if words is not None:
                with poutine.scale(scale=self.lambdas["word"]):
                    pyro.sample("obs_word", word_dist
                                obs=words.reshape(-1, self.embed_dim))
            
            img_loc = self.image_decoder.forward(z)
            img_dist = dist.Bernoulli(img_loc).independent(1)
            # score against actual images
            if images is not None:
                with poutine.scale(scale=self.lambdas["image"]):
                    pyro.sample("obs_img", img_dist,
                                obs=images.reshape(-1, 3, self.img_width,
                                                   self.img_width))
            
            rating_loc, rating_scale = self.rating_decoder.forward(z)
            rating_dist = dist.Normal(rating_loc, rating_scale).independent(1)
            # score against actual ratings
            if ratings is not None:
                with poutine.scale(scale=self.lambdas["rating"]):
                    pyro.sample("obs_rating", rating_dist
                                obs=ratings.reshape(-1, self.rating_dim))

            outcome_loc, outcome_scale = self.outcome_decoder.forward(z)
            outcome_dist = dist.Normal(outcome_loc,
                                       outcome_scale).independent(1)
            # score against actual outcomes
            if outcomes is not None:
                with poutine.scale(scale=self.lambdas["outcome"]):
                    pyro.sample("obs_outcome", outcome_dist,
                                obs=outcomes.reshape(-1, self.outcome_dim))
                
            # return the loc so we can visualize it later
            return word_loc, img_loc, rating_loc, outcome_loc
    
    def guide(self, words=None, images=None, ratings=None, outcomes=None,
              annealing_beta=1.0):
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
            with poutine.scale(scale=annealing_beta):
                pyro.sample("latent",
                            dist.Normal(z_loc, z_scale).independent(1))
    
