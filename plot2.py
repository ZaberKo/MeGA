import numpy as np
import matplotlib.pyplot as plt
import os.path

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path',type=str, help="log file path")
    args=parser.parse_args()


    
    train_acc=[]
    val_acc=[]
    with open(args.log_path,'r',encoding='utf-8') as f:
        for line in f:
            if 'val acc:' in line:
                val_acc.append(float(line.split(':')[2].split(' ')[1]))
            elif 'train acc' in line:
                train_acc.append(float(line.split(':')[1]))

    
    directory=os.path.dirname(args.log_path)
    filename=os.path.splitext(os.path.basename(args.log_path))[0]

    train_acc=np.array(train_acc)
    val_acc=np.array(val_acc)

    print(f'epochs:{len(train_acc)} max_val_acc:{val_acc.max()}[epoch {val_acc.argmax()}] max_train_acc:{train_acc.max()}[epoch {train_acc.argmax()}]')
    
    x=np.linspace(0,len(train_acc)-1,len(train_acc))

    
    fig=plt.figure(dpi=300)
    

    # ax.yaxis.set_major_locator(plt.MultipleLocator(10))

    # ax.set_ylim(0,100)


    plt.plot(x,train_acc,label='tain_acc')
    plt.plot(x,val_acc,label='val_acc')
    plt.legend()
    plt.xlabel('epochs',fontsize=16)
    plt.ylabel('accuracy',fontsize=16)
    
    plt.savefig(os.path.join(directory,'{}_acc.png'.format(filename)))
