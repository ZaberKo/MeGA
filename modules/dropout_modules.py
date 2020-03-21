import torch 
import torch.nn

class LinearScheduler(nn.Module):
    def __init__(self, module, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.module = module
        self.iter = 0
        self.drop_values = torch.linspace(start=start_value, end=stop_value, steps=nr_steps)


    def forward(self, x):
        return self.module(x)

    def step(self):
        if self.iter < len(self.drop_values):
            self.module.drop_prob = self.drop_values[self.iter]

        self.iter += 1


def _check_and_update(m):
    if isinstance(m,LinearScheduler):
        m.step()


def update_dropout_schedule(module:nn.Module):    
    module.apply(_check_and_update)