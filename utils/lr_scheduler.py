import math


class warmup_scheduler(object):
    def __init__(self,
                 base_lr,
                 iter_per_epoch,
                 max_epoch,
                 multi_step=[30, 60, 90],
                 gamma=.1,
                 warmup_epoch=5):
        super(warmup_scheduler, self).__init__()

        self.base_lr = base_lr
        self.warmup_iters = iter_per_epoch * warmup_epoch
        self.current_iter = 1
        if multi_step:
            print('Using multi-step learning rate decay')
            self.get_lr = self.step_get_lr
            self.multi_step = multi_step
            self.iter_per_epoch = iter_per_epoch
            self.gamma = gamma
        else:
            print('Using cosine learning rate decay')
            self.get_lr = self.cosine_get_lr
            self.cosine_iters = iter_per_epoch * (max_epoch - warmup_epoch)

    def step_get_lr(self):
        if self.current_iter < self.warmup_iters:
            lr_ratio = self.current_iter / self.warmup_iters
        else:
            num_epochs = (self.current_iter -
                          self.warmup_iters) / self.iter_per_epoch
            stage = sum([num_epochs > k for k in self.multi_step])
            lr_ratio = self.gamma**stage
        self.current_iter += 1
        return lr_ratio

    def cosine_get_lr(self):
        if self.current_iter < self.warmup_iters:
            lr_ratio = self.current_iter / self.warmup_iters
        else:
            process = (self.current_iter -
                       self.warmup_iters) / self.cosine_iters
            lr_ratio = .5 * (1 + math.cos(process * math.pi))
            lr_ratio = max(lr_ratio, 1e-5)
        self.current_iter += 1
        return lr_ratio

    def step(self, optimizer):
        if self.current_iter == 1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        lr_ratio = self.get_lr()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_ratio * param_group['initial_lr']
        return lr_ratio * self.base_lr
