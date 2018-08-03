"""Tests words trained with the SSVAE"""

import os
import json
import torch
import ray
from ray.tune import register_trainable

from utils.trainables import SSVAETrainable

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_GPUS = int(torch.cuda.device_count())
USE_CUDA = torch.cuda.is_available()
MODEL_DIR = os.path.join(CODE_DIR, "..", "ssvae_best_results")
MODEL_PATH = os.path.join(MODEL_DIR, "ssvae_model_best.save")
EMBED_PATH = os.path.join(CODE_DIR, "..", "glove", "glove.6B.50d.txt")

if __name__ == "__main__":
    ray.init(num_gpus=NUM_GPUS)
    with open(os.path.join(MODEL_DIR, "params.json")) as config_f:
        config = json.load(config_f)
        if "embed_path" in config:
            config["embed_path"] = EMBED_PATH
    # Load config and model
    ssvae_trainable = SSVAETrainable(config=config)
    ssvae_trainable._restore(MODEL_PATH)
    # Run word test
    ssvae_trainable.word_test(test_words=['amazing', 'shit', 'gah'])
