import numpy as np
import matplotlib.pyplot as plt
import os.path

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path',type=str, help="log file path")
    args=parser.parse_args()


    train_loss=[]
    train_acc1=[]

    train_acc=[]
    val_acc=[]    
    with open(args.log_path,'r',encoding='utf-8') as f:
        for line in f:
            if 'Train:' in line:
                train_loss.append(float(line.split(':')[6].split('[')[0]))
                train_acc1.append(float(line.split(':')[8].split('[')[0]))

            elif 'val acc:' in line:
                val_acc.append(float(line.split(':')[2].split(' ')[1]))
            elif 'train acc' in line:
                train_acc.append(float(line.split(':')[1]))

    
    directory=os.path.dirname(args.log_path)
    filename=os.path.splitext(os.path.basename(args.log_path))[0]

    train_loss=np.array(train_loss)
    train_acc1=np.array(train_acc1)

    train_acc=np.array(train_acc)
    val_acc=np.array(val_acc)

    print('iterations:{} min_loss:{} max_acc:{}'.format(len(train_loss),train_loss.min(),train_acc1.max()))
    print(f'epochs:{len(train_acc)} max_val_acc:{val_acc.max()}[epoch {val_acc.argmax()}] max_train_acc:{train_acc.max()}[epoch {train_acc.argmax()}]')
    
    x1=np.linspace(0,len(train_acc1)-1,len(train_acc1))
    x2=np.linspace(0,len(train_acc)-1,len(train_acc))

    
    fig=plt.figure(figsize=(12,12),dpi=300)
    # ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    # ax.set_ylim(0,100)
    ax1=fig.add_subplot(2,1,1)
    ax1.plot(x1,train_loss)
    ax1.set_title('train loss',fontsize=16)
    ax1.set_xlabel('iterations',fontsize=12)
    ax1.set_ylabel('loss',fontsize=12)

    ax2=fig.add_subplot(2,1,2)
    ax2.plot(x1,train_acc1)
    ax2.set_title('train acc',fontsize=16)
    ax2.set_ylim(0,100)
    ax2.set_xlabel('iterations',fontsize=12)
    ax2.set_ylabel('accuracy',fontsize=12)
    
    plt.savefig(os.path.join(directory,'{}.png'.format(filename)))

    fig=plt.figure(dpi=300)
    # ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    # ax.set_ylim(0,100)
    plt.plot(x2,train_acc,label='tain_acc')
    plt.plot(x2,val_acc,label='val_acc')
    plt.legend()
    plt.xlabel('epochs',fontsize=16)
    plt.ylabel('accuracy',fontsize=16)
    
    plt.savefig(os.path.join(directory,'{}_acc.png'.format(filename)))
