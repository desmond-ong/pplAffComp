"""Trainables for hyperparameter optimization via Ray Tune."""
import os

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

OUTCOME_VAR_NAMES = ['payoff1', 'payoff2', 'payoff3', 
                     'prob1', 'prob2', 'prob3', 
                     'win', 'winProb', 'angleProp']
EMOTION_VAR_NAMES = ['happy', 'sad', 'anger', 'surprise', 
                     'disgust', 'fear', 'content', 'disapp']

OUTCOME_VAR_DIM = len(OUTCOME_VAR_NAMES)
EMOTION_VAR_DIM = len(EMOTION_VAR_NAMES)
EMBED_DIM = 50
IMG_WIDTH = 64

class MVAETrainable(tune.Trainable):
    def _setup(self):
        self.epochs = 0
        # Clear previous pyro parameters
        pyro.clear_param_store()
        # Load the data
        if self.config['dataset'] == "face":
            self.embeddings = None
            self.config['embed_dim'] = EMBED_DIM
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
        optimizer = Adam(lr=self.config['lr'])
        # Setup the inference algorithm
        self.svi = SVI(self.mvae.model, self.mvae.guide,
                       optimizer, loss=Trace_ELBO())
                    
    def _train(self):
        # Initialize loss accumulator
        epoch_loss = 0.
        # Do a training epoch over each mini-batch
        for (batch_num, batch_data in enumerate(self.loader)):
            # Discard the word string
            batch_data = list(batch_data[1:])
            for i in range(len(batch_data)):
                if len(batch_data[i].shape) == 1:
                    # If data modality is missing, set to None
                    batch_data[i] = None
                elif self.config['use_cuda']:
                    # Store in CUDA memory
                    batch_data[i].cuda()
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
