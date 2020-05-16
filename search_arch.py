import numpy as np
from search.model_eval_ddp import start_eval_processes, evaluate_arch,close_eval_processes
from utils import *
from mplog import MPLog
import yaml
import pickle
import os
import argparse
import time

logger = None


def gen_random_paths(num_layers, num_choices, num_paths=1):
    paths = np.random.randint(0, num_choices, size=(num_paths, num_layers))
    return path_idx2name(paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', default='search_debug.log',
                        type=str, help='log file path',)
    args = parser.parse_args()
    logger=MPLog(args.log_path,0)
    with open('config.yaml', mode='r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    np.random.seed(42)
    
    pickle_file = './search/{}_{}2.pickle'.format(config['search_config']['model'],config['search_config']['dataset'])
    if not os.path.exists(pickle_file):
        th = start_eval_processes(config)
        paths = gen_random_paths(14, 12, 200)

        acc = {}
        for path in paths:
            begin_time=time.time()
            top1,top5 = evaluate_arch(path)
            logger.log(path)
            logger.log(f'time: {time.time()-begin_time} s top1: {top1.avg} top5: {top5.avg}')
            acc[str(path)] = {
                'top1':top1.avg,
                'top5':top5.avg
            }

        

        with open(pickle_file, 'wb') as f:
            pickle.dump(acc, f)

        close_eval_processes()
    else:
        with open(pickle_file,'rb') as f:
            acc=pickle.load(f)

    

    
