import torch

class NoamOpt(torch.optim.lr_scheduler._LRScheduler):
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, model_size, factor, warmup, last_epoch=-1):
        
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        super(NoamOpt, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        return self._rate

    def step(self, epoch=None):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()

        for p in self.optimizer.param_groups:
            p['lr'] = rate        
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        self._rate = rate
        #self.optimizer.step()

        
    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))