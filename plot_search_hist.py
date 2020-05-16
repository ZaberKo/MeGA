import numpy as np
import matplotlib.pyplot as plt
import os.path
import pickle

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path',type=str, help="log file path")
    args=parser.parse_args()


    with open(args.log_path,'rb') as f:
        acc_list=pickle.load(f)

    top1=[]
    for arch,score in acc_list.items():
        # if 'skip' not in arch:
            top1.append(score['top1'])
    
    directory=os.path.dirname(args.log_path)
    filename=os.path.splitext(os.path.basename(args.log_path))[0]

    top1=np.array(top1)
    print(f'min: {top1.min()} max: {top1.max()} avg: {top1.mean()}')
    

    
    fig=plt.figure(dpi=300)
    

    

    # ax.set_ylim(0,100)
    ax=plt.gca()
    # ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax.hist(top1,bins=20)
    ax.set_title('model accuracy distribution',fontsize=16)
    ax.set_xlabel('accuracy',fontsize=16)
    ax.set_ylabel('count',fontsize=16)

    
    plt.savefig(os.path.join(directory,'{}.png'.format(filename)))
