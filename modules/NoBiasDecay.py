import torch.nn as nn
from .hypernet import Hypernet

def noBiasDecay(model, lr, weight_decay):
    '''
    no bias decay : only apply weight decay to the weights in convolution and fully-connected layers
    In paper [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)
    Ref: https://github.com/weiaicunzai/Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks/blob/master/utils.py
    '''
    decay, bias_no_decay, weight_no_decay = [], [], []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)
            if m.bias is not None:
                bias_no_decay.append(m.bias)    
        else: 
            if hasattr(m, 'weight'):
                weight_no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                bias_no_decay.append(m.bias)
        
    assert len(list(model.parameters())) == len(decay) + len(bias_no_decay) + len(weight_no_decay)
    
    # bias using 2*lr
    return [{'params': bias_no_decay, 'lr': 2*lr, 'weight_decay': 0.0}, {'params': weight_no_decay, 'lr': lr, 'weight_decay': 0.0}, {'params': decay, 'lr': lr, 'weight_decay': weight_decay}]

def noBiasDecay_hypernet(model, lr, weight_decay,num_choices):
    '''
    no bias decay : only apply weight decay to the weights in convolution and fully-connected layers
    In paper [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)
    Ref: https://github.com/weiaicunzai/Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks/blob/master/utils.py
    '''
    assert isinstance(model,Hypernet), 'this method is only for hypernet training'
    
    decay1, bias_no_decay1, weight_no_decay1 = [], [], []
    decay2, bias_no_decay2, weight_no_decay2 = [], [], []

    def update(modules,decay, bias_no_decay, weight_no_decay):
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    decay.append(m.weight)
                    if m.bias is not None:
                        bias_no_decay.append(m.bias)    
                else: 
                    if hasattr(m, 'weight'):
                        weight_no_decay1.append(m.weight)
                    if hasattr(m, 'bias'):
                        bias_no_decay.append(m.bias)

    update(model.fixed_modules,decay1,bias_no_decay1,weight_no_decay1)
    update(model.choice_modules,decay2,bias_no_decay2,weight_no_decay2)
        
    assert len(list(model.parameters())) == len(decay1) + len(bias_no_decay1) + len(weight_no_decay1)+len(decay2) + len(bias_no_decay2) + len(weight_no_decay2)
    
    # bias using 2*lr
    return [
        {'params': bias_no_decay1, 'lr': 2*lr/num_choices, 'weight_decay': 0.0}, 
        {'params': weight_no_decay1, 'lr': lr/num_choices, 'weight_decay': 0.0}, 
        {'params': decay1, 'lr': lr/num_choices, 'weight_decay': weight_decay},
        {'params': bias_no_decay2, 'lr': 2*lr, 'weight_decay': 0.0}, 
        {'params': weight_no_decay2, 'lr': lr, 'weight_decay': 0.0}, 
        {'params': decay2, 'lr': lr, 'weight_decay': weight_decay}        
        ]
