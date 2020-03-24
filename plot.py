import numpy as np
import matplotlib.pyplot as plt
import os.path

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path',type=str, help="log file path")
    args=parser.parse_args()


    train_loss=[]
    train_acc=[]
    with open(args.log_path,'r',encoding='utf-8') as f:
        for line in f:
            if 'Train:' in line:
                train_loss.append(float(line.split(':')[6].split(' ')[0]))
                train_acc.append(float(line.split(':')[9].split(' ')[0]))

    
    directory=os.path.dirname(args.log_path)
    filename=os.path.splitext(os.path.basename(args.log_path))[0]

    train_loss=np.array(train_loss)
    train_acc=np.array(train_acc)

    print('iterations:{} min_loss:{} max_acc:{}'.format(len(train_loss),train_loss.min(),train_acc.max()))
    
    x=np.linspace(0,len(train_loss)-1,len(train_loss))

    
    fig=plt.figure(figsize=(12,12),dpi=300)
    

    # ax.yaxis.set_major_locator(plt.MultipleLocator(10))

    # ax.set_ylim(0,100)
    
    ax1=fig.add_subplot(2,1,1)
    ax1.plot(x,train_loss)
    ax1.set_title('train loss',fontsize=16)
    ax1.set_xlabel('iterations',fontsize=16)
    ax1.set_ylabel('loss',fontsize=16)

    ax2=fig.add_subplot(2,1,2)
    ax2.plot(x,train_acc)
    ax2.set_title('train acc')
    ax2.set_ylim(0,100)
    ax2.set_xlabel('iterations',fontsize=16)
    ax2.set_ylabel('accuracy',fontsize=16)
    
    plt.savefig(os.path.join(directory,'{}.png'.format(filename)))
