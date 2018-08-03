"""Tests words trained with the SSVAE"""

import os
import sys
import json

from utils.trainables import SSVAETrainable

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CODE_DIR, "..", "ssvae_best_results")
MODEL_PATH = os.path.join(MODEL_DIR, "ssvae_model_best.save")
EMBED_PATH = os.path.join(CODE_DIR, "..", "glove", "glove.6B.50d.txt")

if __name__ == "__main__":
    # Get test words from command line, else default to None
    test_words = None
    if len(sys.argv) > 1:
        test_words = sys.argv[1:]
    with open(os.path.join(MODEL_DIR, "params.json")) as config_f:
        config = json.load(config_f)
        if "embed_path" in config:
            config["embed_path"] = EMBED_PATH
    # Create dummy logger so trainable initialization won't create log files
    dummy_logger = lambda: None
    dummy_logger.logdir = None
    dummy_logger_creator = lambda c : dummy_logger
    # Load config and model
    ssvae_trainable = SSVAETrainable(config, dummy_logger_creator)
    ssvae_trainable._restore(MODEL_PATH)
    # Run word test
    ssvae_trainable.word_test(test_words)
