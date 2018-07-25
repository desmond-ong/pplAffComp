"""Helper functions for using GloVe word embeddings."""
import os
import numpy as np
import torch

def normalize(v):
    norm = np.sqrt(v.dot(v))
    return v / norm

def cosine_sim_np(a, b):
    return np.dot(normalize(a), normalize(b))

def cosine_sim_torch(a, b):
    a_norm = a / a.norm()
    b_norm = b / b.norm()    
    return torch.dot(a_norm, b_norm)
    
def load_embeddings(path, normalize_embeddings=False):
    with open(path,'r') as f:
        embeddings = {}
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embed = np.array([float(val) for val in split_line[1:]],
                             dtype=np.float32)
            if normalize_embeddings:
                embed = normalize(embed)
            embeddings[word] = embed
        embed_dim = len(embeddings[word])
        return embeddings, embed_dim
