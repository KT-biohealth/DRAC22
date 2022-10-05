import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score,  cohen_kappa_score
import numpy as np

import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    https://dacon.io/competitions/official/235585/codeshare/1796
    """

    def __init__(self, gamma=2.0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # print(self.gamma)
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

def Cohen_Kappa(label, result):
    return cohen_kappa_score(label, result, weights='quadratic')

def ROC_AUC(label, result):
    return roc_auc_score(label, result, average='macro', multi_class='ovr')

def Confusion_Matrix(label, result):
    return confusion_matrix(label, result)


class KappaLoss(nn.Module):
    def __init__(self, num_classes, y_pow=2, eps=1e-10):
        super(KappaLoss, self).__init__()
        self.num_classes = num_classes
        self.y_pow = y_pow
        self.eps = eps

    def kappa_loss(self, y_pred, y_true):
        num_classes = self.num_classes
        y = torch.eye(num_classes).cuda()
        y_true = y[y_true]

        y_true = y_true.float()
        repeat_op = torch.Tensor(list(range(num_classes))).unsqueeze(
            1).repeat((1, num_classes)).cuda()
        repeat_op_sq = torch.square((repeat_op - repeat_op.T))
        weights = repeat_op_sq / ((num_classes - 1) ** 2)

        pred_ = y_pred ** self.y_pow
        pred_norm = pred_ / \
            (self.eps + torch.reshape(torch.sum(pred_, 1), [-1, 1]))

        hist_rater_a = torch.sum(pred_norm, 0)
        hist_rater_b = torch.sum(y_true, 0)
        print('pred_ is :', pred_)
        print('pred_norm is :',pred_norm)
        print('a is :', hist_rater_a)
        print(hist_rater_b)


        conf_mat = torch.matmul(pred_norm.T, y_true)

        bsize = y_pred.size(0)
        nom = torch.sum(weights * conf_mat)
        expected_probs = torch.matmul(torch.reshape(
            hist_rater_a, [num_classes, 1]), torch.reshape(hist_rater_b, [1, num_classes]))
        denom = torch.sum(weights * expected_probs / bsize)

        return nom / (denom + self.eps)

    def forward(self, y_pred, y_true):
        return self.kappa_loss(y_pred, y_true)


class SAM(torch.optim.Optimizer):
    """
    from sam import SAM
    ...

    model = YourModel()
    --------1
    base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=200)

    --------2
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=2.0, adaptive=True, lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class EarlyStopping_for_mse:
    def __init__(self, patience=5):
        self.loss = np.inf
        self.patience = 0
        self.patience_limit = patience

    def step(self, loss):
        if self.loss > loss:
            self.loss = loss
            self.patience = 0
        else:
            self.patience += 1
            print('patience is up, now is : ', self.patience)

    def is_stop(self):
        return self.patience >= self.patience_limit


class EarlyStopping:
    def __init__(self, patience=5):
        self.cohen = 0
        self.patience = 0
        self.patience_limit = patience

    def step(self, now_cohen):
        if self.cohen < now_cohen:
            self.cohen = now_cohen
            self.patience = 0
        else:
            self.patience += 1
            print('patience is up, now is : ', self.patience)

    def is_stop(self):
        return self.patience >= self.patience_limit
