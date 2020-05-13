import math

from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupLR(_LRScheduler):
    '''
    Cosine lr decay function with warmup.
    Ref: https://github.com/PistonY/torch-toolbox/blob/master/torchtoolbox/optimizer/lr_scheduler.py
         https://github.com/Randl/MobileNetV3-pytorch/blob/master/cosine_with_warmup.py
    Lr warmup is proposed by 
        `Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour`
        `https://arxiv.org/pdf/1706.02677.pdf`
    Cosine decay is proposed by 
        `Stochastic Gradient Descent with Warm Restarts`
        `https://arxiv.org/abs/1608.03983`
    Args:
        optimizer (Optimizer): optimizer of a model.
        iter_in_one_epoch (int): number of iterations in one epoch.
        epochs (int): number of epochs to train.
        lr_min (float): minimum(final) lr.
        warmup_epochs (int): warmup epochs before cosine decay.
        last_epoch (int): init iteration. In truth, this is last_iter
    Attributes:
        niters (int): number of iterations of all epochs.
        warmup_iters (int): number of iterations of all warmup epochs.
        cosine_iters (int): number of iterations of all cosine epochs.
    '''

    def __init__(self, optimizer, epochs, iter_in_one_epoch, lr_min=0, warmup_epochs=0, last_epoch=-1):
        self.lr_min = lr_min
        self.niters = epochs * iter_in_one_epoch
        self.warmup_iters = iter_in_one_epoch * warmup_epochs
        self.cosine_iters = iter_in_one_epoch * (epochs - warmup_epochs)
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return [(self.lr_min + (base_lr - self.lr_min) * self.last_epoch / self.warmup_iters) for base_lr in self.base_lrs]
        else:
            return [(self.lr_min + (base_lr - self.lr_min) * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_iters) / self.cosine_iters)) / 2) for base_lr in self.base_lrs]


class CosineAnnealingWarmRestarts(_LRScheduler):
    '''
    see: https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
    '''
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, T_warm=0, decay=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(
                "Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(
                "Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_warm < 0 or not isinstance(T_warm, int):
            raise ValueError(
                "Expected positive integer T_warm, but got {}".format(T_warm))
        self.T_0 = T_0  # the first number of epochs between two warm restarts in SGDR
        self.T_i = T_0  # the number of epochs between two warm restarts in SGDR
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_warm = T_warm
        self.decay = decay
        self.cycle = 0

        super(CosineAnnealingWarmRestarts, self).__init__(
            optimizer, last_epoch)
        self.T_cur = self.last_epoch

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_warm:
            return [self.eta_min+(base_lr*(self.decay**self.cycle)-self.eta_min)*self.T_cur / self.T_warm
                    for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr*(self.decay**self.cycle)-self.eta_min) * (1 + math.cos(math.pi * (self.T_cur-self.T_warm) / (self.T_i - self.T_warm))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_warm) * self.T_mult + self.T_warm
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    T_sum=self.T_0
                    T_i=self.T_0
                    n=0
                    while epoch>T_sum:
                        n+=1
                        T_i=self.T_warm+(self.T_0-self.T_warm)*self.T_mult**n
                        T_sum+=T_i


                    self.cycle = n
                    self.T_cur = T_i-(T_sum-epoch)
                    self.T_i = T_i
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]



class CosineAnnealingWarmRestarts2(_LRScheduler):
    '''
    copied from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingWarmRestarts.
    We modify it.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    '''

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0,  warmup_epochs=0, decay_rate=0.5, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(
                "Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(
                "Expected integer T_mult >= 1, but got {}".format(T_mult))
        if warmup_epochs < 0 or not isinstance(warmup_epochs, int):
            raise ValueError(
                "Expected positive integer warmup_epochs, but got {}".format(warmup_epochs))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.decay_power = 0
        super(CosineAnnealingWarmRestarts2, self).__init__(
            optimizer, last_epoch)
        self.T_cur = self.last_epoch

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [(self.eta_min + (base_lr - self.eta_min) * self.T_cur / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr * (self.decay_rate**self.decay_power) - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        '''
            Step could be called after every batch update
        '''
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            # self.T_cur = self.T_cur + 1
            # if self.T_cur >= self.T_i:
            #     self.T_cur = self.T_cur - self.T_i
            #     self.T_i = self.T_i * self.T_mult
        
        if epoch < 0:
            raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            
        if epoch < self.warmup_epochs:
            self.T_cur = epoch
        else:
            epoch_cur = epoch - self.warmup_epochs
            if epoch_cur >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch_cur % self.T_0
                    self.decay_power = epoch_cur // self.T_0
                else:
                    n = int(math.log((epoch_cur / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch_cur - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
                    self.decay_power = n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch_cur
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
