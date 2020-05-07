import torch 
import torch.nn as nn

from .dropblock import DropBlock2D,DropBlock2D_Channel

class LinearScheduler(nn.Module):
    def __init__(self, module):
        super(LinearScheduler, self).__init__()
        self.module = module
        self.iter = 0
        

    def init(self,start_value, stop_value, nr_steps):
        assert self.iter==0
        self.drop_values = torch.linspace(start=start_value, end=stop_value, steps=nr_steps)
        self.module.drop_prob = self.drop_values[self.iter]

    def forward(self, x):
        return self.module(x)

    def step(self):
        assert hasattr(self,'drop_values'), 'LinearScheduler must call init() first'
        self.iter += 1
        if self.iter < len(self.drop_values):
            self.module.drop_prob = self.drop_values[self.iter]

        


class ScheduleDropBlock(nn.Module):
    def __init__(self,block_size, per_channel=False):
        super(ScheduleDropBlock,self).__init__()
        # use other method to init dropout_rate
        start_dropout_rate=0
        if per_channel:
            self.dropblock=DropBlock2D_Channel(start_dropout_rate,block_size)
        else:
            self.dropblock=DropBlock2D(start_dropout_rate,block_size)
        self.schduler=LinearScheduler(self.dropblock)


    def forward(self,x):
        return self.schduler(x)

def _check_and_update(m):
    if isinstance(m,LinearScheduler):
        m.step()

def _check_and_init(m,start_value, stop_value, nr_steps):
    if isinstance(m,LinearScheduler):
        m.init(start_value, stop_value, nr_steps)

def _custom_apply(module,fn,*args):
    for sub_module in module.children():
        _custom_apply(sub_module,fn,*args)
    fn(module,*args)

def update_dropout_schedule(module:nn.Module):    
    module.apply(_check_and_update)



def init_dropout_schedule(module:nn.Module,start_value, stop_value, nr_steps):    
    # module.apply(_check_and_init)
    _custom_apply(module,_check_and_init,start_value, stop_value, nr_steps)



    