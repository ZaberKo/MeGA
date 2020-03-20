import numpy as np
import matplotlib.pyplot as plt
import os.path

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path',type=str, help="log file path")
    args=parser.parse_args()


    train_loss=[]
    with open(args.log_path,'r',encoding='utf-8') as f:
        for line in f:
            if 'Train:' in line:
                # print(float(line.split(':')[6].split(' ')[0]))
                train_loss.append(float(line.split(':')[6].split(' ')[0]))

    
    directory=os.path.dirname(args.log_path)
    filename=os.path.splitext(os.path.basename(args.log_path))[0]

    train_loss=np.array(train_loss)


    
    x=np.linspace(0,len(train_loss)-1,len(train_loss))

    
    plt.figure()
    ax=plt.gca()

    # ax.yaxis.set_major_locator(plt.MultipleLocator(10))

    # ax.set_ylim(0,100)
    # plt.ylim(0,100)
    plt.plot(x,train_loss,label='train loss')
    plt.legend()
    plt.savefig(os.path.join(directory,'{}.png'.format(filename)))
