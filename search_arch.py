import numpy as np
from search.model_eval_ddp import start_eval_processes, evaluate_arch
from utils import *
from mplog import init_log
import logging
import yaml
import pickle
import os

logger = logging.getLogger(__name__)


def gen_random_paths(num_layers, num_choices, num_paths=1):
    paths = np.random.randint(0, num_choices, size=(num_paths, num_layers))
    return path_idx2name(paths)


if __name__ == "__main__":
    init_log()
    with open('config.yaml', mode='r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    np.random.seed(42)
    pickle_file = './search/sample.pickle'
    if not os.path.exists(pickle_file):
        th = start_eval_processes(config)
        paths = gen_random_paths(14, 12, 200)

        acc = {}
        for path in paths:
            logger.info(path)
            top1 = evaluate_arch(path)
            logger.info(top1.avg)
            acc[str(path)] = top1.avg

        with open(pickle_file, 'wb') as f:
            pickle.dump(acc, f)
    else:
        with open(pickle_file,'rb') as f:
            acc=pickle.load(f)

    
