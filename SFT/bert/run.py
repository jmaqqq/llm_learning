
"""
    usage: python run.py --model bert --data intent
"""

import time
import torch
import numpy as np
from train_eval import train, test
from importlib import import_module
import argparse
from data_helper import build_dataset, build_iterator, get_time_dif
from utils import logger
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--data', type=str, required=True)
args = parser.parse_args()


if __name__ == '__main__':
    dataset = args.data
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    logger.info("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    logger.info(f"Time usage: {time_dif}")

    # t1 & eval
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
