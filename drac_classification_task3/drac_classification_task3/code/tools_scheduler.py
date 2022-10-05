import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(select_scheduler, optimizer, factor=0.2, patience=4, eps=1e-6, T_max=10, min_lr=1e-6, T_0=10, freeze_epo = 0, warmup_epo = 1, cosine_epo = 3):
    if select_scheduler =='ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True, eps=eps)
    elif select_scheduler =='CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr, last_epoch=-1)
    elif select_scheduler =='CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=min_lr, last_epoch=-1)
    elif select_scheduler =='GradualWarmupSchedulerV2':
        scheduler_cosine=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cosine_epo)
        scheduler_warmup=GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)
        scheduler=scheduler_warmup

    return scheduler

# def step_scheduler(select_scheduler, sheduler, val_loss, epoch):
#     if select_scheduler =='ReduceLROnPlateau':
#         scheduler.step(val_loss)
#     elif select_scheduler =='GradualWarmupSchedulerV2':
#         scheduler.step(epoch)
#     else:
#         scheduler.step()
    
#     return scheduler
def step_scheduler(select_scheduler, scheduler, val_loss, epoch):
    if select_scheduler =='ReduceLROnPlateau':
        scheduler.step(val_loss)
    elif select_scheduler =='GradualWarmupSchedulerV2':
        scheduler.step(epoch)
    else:
        scheduler.step()