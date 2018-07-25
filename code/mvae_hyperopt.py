"""Uses population based training to tune hyperparameters."""

import os
import ray
from ray.tune import register_trainable, run_experiments, grid_search

from utils.trainables import MVAETrainable

USE_CUDA = False
EMBED_PATH = os.path.join(os.path.abspath('..'), "glove", "glove.6B.50d.txt")

if __name__ == "__main__":
    ray.init()
    register_trainable("mvae_trainable", MVAETrainable)
    # Use population based training to tune hyperparameters
    run_experiments(
    {
        "mvae_train": {
            "run": "mvae_trainable",
            "stop": {
                "training_iteration": 10000
            },
            "config": {
                "dataset": "word",
                "batch_size": 32,
                "lr": grid_search([5e-5, 1e-5, 5e-6, 1e-6]),
                "z_dim": grid_search(range(10,51,5)),
                "use_cuda": USE_CUDA,
                "embed_path": EMBED_PATH,
                "normalize_embeddings": False
            },
        }
    })
