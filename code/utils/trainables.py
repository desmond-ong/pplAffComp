"""Trainables for hyperparameter optimization via Ray Tune."""
import os

from itertools import cycle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

import ray.tune as tune

from glove import load_embeddings
from datasets import *
from mvae import MVAE
from ssvae import SSVAE

DEFAULT_EMBED_DIM = 50
IMG_WIDTH = 64

class MVAETrainable(tune.Trainable):
    """Trainable for the MVAE. Relevant hyperparameters:

    dataset -- either a "face" or "word" dataset
    batch_size -- number of samples per batch
    lr -- learning rate for Adam optimizer
    z_dim -- number of latent dimensions
    embed_path -- path to glove word embeddings (dimension is deduced)
    normalize_embeddings -- whether to normalize word embeddings after loading
    use_cuda -- whether to train using CUDA
    """

    def _setup(self):
        self.epochs = 0
        # Clear previous pyro parameters
        pyro.clear_param_store()
        # Load the data
        if self.config['dataset'] == "face":
            self.embeddings = None
            self.config['embed_dim'] = DEFAULT_EMBED_DIM
            self.dataset, self.loader =\
                load_face_outcome_emotion_data(self.config['batch_size'])
        elif self.config['dataset'] == "word":
            self.embeddings, self.config['embed_dim'] = \
                load_embeddings(self.config['embed_path'],
                                self.config['normalize_embeddings'])
            self.dataset, self.loader =\
                load_word_outcome_emotion_data(self.config['batch_size'],
                                               self.embeddings)
        # Setup the MVAE
        self.mvae = MVAE(z_dim=self.config['z_dim'], img_width=IMG_WIDTH,
                         embed_dim=self.config['embed_dim'],
                         rating_dim=EMOTION_VAR_DIM,
                         outcome_dim=OUTCOME_VAR_DIM,
                         use_cuda=self.config['use_cuda'],)
        # Setup the optimizer
        optimizer = Adam({'lr': self.config['lr']})
        # Setup the inference algorithm
        self.svi = SVI(self.mvae.model, self.mvae.guide,
                       optimizer, loss=Trace_ELBO())
                    
    def _train(self):
        # Initialize loss accumulator
        epoch_loss = 0.
        # Do a training epoch over each mini-batch
        for batch_num, batch_data in enumerate(self.loader):
            # Discard the word string
            batch_data = list(batch_data[1:])
            for i in range(len(batch_data)):
                if len(batch_data[i].shape) == 1:
                    # If data modality is missing, set to None
                    batch_data[i] = None
                elif self.config['use_cuda']:
                    # Store in CUDA memory
                    batch_data[i] = batch_data[i].cuda()
            # Run SVI optimization step
            epoch_loss += self.svi.step(*batch_data)
        # Report training diagnostics
        n_samples = len(self.dataset)
        mean_loss = epoch_loss / n_samples
        self.epochs += 1
        return tune.TrainingResult(mean_loss=mean_loss,
                                   timesteps_this_iter=1)

    def _save(self, checkpoint_dir):
        filename = "mvae_model_epoch_{}.save".format(self.epochs)
        path = os.path.join(checkpoint_dir, filename) 
        pyro.get_param_store().save(path)
        return path

    def _restore(self, checkpoint_path):
        pyro.get_param_store().load(checkpoint_path)
        pyro.module("mvae", self.mvae, update_module_params=True)


class SSVAETrainable(tune.Trainable):
    """Trainable for the SSVAE. Relevant hyperparameters:

    dataset -- either a "face" or "word" dataset
    batch_size -- number of samples per batch
    unsup_ratio -- ratio of unsupervised to supervised batches
    lr -- learning rate for Adam optimizer
    z_dim -- number of latent dimensions
    hidden_layers -- number of hidden layers and units, e.g. [100, 100]
    aux_loss_mult -- multiplier for auxiliary loss
    beta_fn -- annealing function for ELBO beta multiplier, epoch num is arg
    embed_path -- path to glove word embeddings (dimension is deduced)
    normalize_embeddings -- whether to normalize word embeddings after loading
    use_cuda -- whether to train using CUDA
    """
    
    def _setup(self):
        self.epochs = 0
        # Clear previous pyro parameters
        pyro.clear_param_store()
        # Load the data
        if self.config['dataset'] == "face":
            self.embeddings = None
            self.config['embed_dim'] = DEFAULT_EMBED_DIM
            self.sup_dataset, self.sup_loader =\
                load_face_outcome_emotion_data(self.config['batch_size'])
            self.unsup_dataset, self.unsup_loader =\
                load_face_only_data(self.config['batch_size'])
            self.input_size = np.prod(self.sup_dataset.image_shape())
        elif self.config['dataset'] == "word":
            self.embeddings, self.input_size = \
                load_embeddings(self.config['embed_path'],
                                self.config['normalize_embeddings'])
            self.sup_dataset, self.sup_loader =\
                load_word_outcome_emotion_data(self.config['batch_size'],
                                               self.embeddings)
            self.unsup_dataset, self.unsup_loader =\
                load_word_only_data(self.config['batch_size'],
                                    self.embeddings)
        # Setup the SSVAE
        self.ssvae = SSVAE(output_size=EMOTION_VAR_DIM,
                           input_size=self.input_size,
                           z_dim=self.config['z_dim'],
                           hidden_layers=self.config['hidden_layers'],
                           use_cuda=self.config['use_cuda'],
                           aux_loss_multiplier=self.config['aux_loss_mult'])
        # Setup the optimizer
        optimizer = Adam({'lr': self.config['lr']})
        # Setup the inference algorithm
        self.loss = SVI(self.ssvae.model, self.ssvae.guide,
                        optimizer, loss=Trace_ELBO())
        self.aux_loss = SVI(self.ssvae.model_rating, self.ssvae.guide_rating,
                            optimizer, loss=Trace_ELBO())
                    
    def _train(self):
        # Initialize loss accumulator
        sup_loss, sup_aux_loss = 0.0, 0.0
        unsup_loss, unsup_aux_loss = 0.0, 0.0
        # Setup alternating between supervised and unsupervised examples
        sup_iter = iter(self.sup_loader)
        unsup_iter = cycle(self.unsup_loader)
        # Ratio of unsupervised to supervised batches is unsup_ratio
        sup_flags = ([True]*len(self.sup_loader) +
                     [False]*len(self.sup_loader) * self.config['unsup_ratio'])
        sup_flags = np.random.permutation(sup_flags)
        # Compute beta parameter for this epoch
        beta_fn= self.config.get("beta_fn", lambda x : 1.0)
        beta = beta_fn(self.epochs)
        # Do a training epoch over each mini-batch
        for batch_num, is_supervised in enumerate(sup_flags):
            if is_supervised:
                batch_data = next(sup_iter)
            else:
                batch_data = next(unsup_iter)
            # Extract the relevant modalities
            if self.config['dataset'] == "face":
                xs, ys = batch_data[2], batch_data[3]
            elif self.config['dataset'] == "word":
                xs, ys = batch_data[1], batch_data[3]
            if len(ys.shape) == 1:
                ys = None
            if self.config['use_cuda']:
                # Store in CUDA memory
                xs = xs.cuda()
                if ys is not None:
                    ys = ys.cuda()
            # Run optimization step
            if is_supervised:
                sup_loss += self.loss.step(xs, ys, beta=beta)
                sup_aux_loss += self.aux_loss.step(xs, ys)
            else:
                unsup_loss += self.loss.step(xs, beta=beta)
                unsup_aux_loss += self.aux_loss.step(xs)
        # Compute training loss
        n_sup = len(self.sup_dataset)
        n_unsup = len(self.sup_dataset) * self.config['unsup_ratio']
        mean_sup_loss = sup_loss / n_sup
        mean_sup_aux_loss = sup_aux_loss / n_sup
        mean_unsup_loss = unsup_loss / n_unsup
        mean_unsup_aux_loss = unsup_aux_loss / n_unsup
        mean_loss = (mean_sup_loss +
                     mean_sup_aux_loss * self.config['aux_loss_mult'] +
                     mean_unsup_loss * self.config['unsup_ratio'])
        self.epochs += 1
        return tune.TrainingResult(mean_loss=mean_loss,
                                   timesteps_this_iter=1)

    def _save(self, checkpoint_dir):
        filename = "ssvae_model_epoch_{}.save".format(self.epochs)
        path = os.path.join(checkpoint_dir, filename) 
        pyro.get_param_store().save(path)
        return path

    def _restore(self, checkpoint_path):
        pyro.get_param_store().load(checkpoint_path)
        pyro.module("ssvae", self.ssvae, update_module_params=True)
