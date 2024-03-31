import copy
from utils.tools import save_model
from modules.overallmodel import _model

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SolverBase(object):
    def __init__(self, hyp_params, data_loaders):
        ####################################################################
        #                                                                  #
        #              Initialize all the config of model                  #
        #                                                                  #
        ####################################################################

        self.hp = hyp_params
        self.data_loaders = data_loaders

        # Initialize the device
        if self.hp.use_cuda:
            self.hp.device = self.device = torch.device("cuda")
        else:
            self.hp.device = self.device = torch.device("cpu")

        # Initialize the model
        self.model = _model(self.hp).to(self.device)
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)  # 统计整个模型的参数数量
        print('\n' + '=' * 7, f" Total number of parameters is {self.total_params}", '=' * 7 + '\n')

        # Initialize more the criterion and loss function
        if self.hp.dataset == "mosei_emo":    # mosei_emo are classification datasets
            self.criterion = nn.CrossEntropyLoss(reduction="mean")
        else:   # mosi and mosei_senti are regression datasets
            self.criterion = nn.L1Loss(reduction="mean")

        # 根据模型，初始化不同的优化器及学习率调整器
        bert_param, va_encoder_param, backbone_param = [], [], []    # 对 bert 的参数进行 fine-tune
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if 'bert' in name:
                    bert_param.append(p)
                elif 'encoder' in name:
                    va_encoder_param.append(p)
                else:
                    backbone_param.append(p)

            # 此for循环用于初始化部分参数；并且是否放在上一个for循环中，对结果影响很大
            for p in va_encoder_param + backbone_param:
                if p.dim() > 1:     # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                    nn.init.xavier_normal_(p)

        # Initialize the optimizer
        # optimizer_pretrained_group = [
        #     {'params': bert_param, 'weight_decay': self.hp.weight_decay_bert, 'lr': self.hp.lr_bert},
        #     {'params': va_encoder_param, 'weight_decay': self.hp.weight_decay_pretrained, 'lr': self.hp.lr_pretrained}
        # ]
        optimizer_main_group = [
            {'params': bert_param, 'weight_decay': self.hp.weight_decay_bert, 'lr': self.hp.lr_bert},  # 先设置为0，看一下效果    self.hp.lr_bert
            {'params': va_encoder_param, 'weight_decay': self.hp.weight_decay_model, 'lr': self.hp.lr_model},
            {'params': backbone_param, 'weight_decay': self.hp.weight_decay_model, 'lr': self.hp.lr_model}
        ]

        self.optimizer = {
            # 'pretrained': getattr(optim, self.hp.optim)(optimizer_pretrained_group),
            'model': getattr(optim, 'Adam')(optimizer_main_group)
        }

        # Initialize the scheduler
        self.scheduler = {
            # 'pretrained': ReduceLROnPlateau(self.optimizer['pretrained'], mode='min', patience=self.hp.when, factor=0.5, verbose=True),
            'main': ReduceLROnPlateau(self.optimizer['model'], mode='min', patience=self.hp.when, factor=0.5, verbose=True)
        }

        # Initalize the early stopping object
        self.earlyStop = self.hp.patience

        # Initialize the best model
        self.best_model = None

    def save_model(self):
        stats = {
            'args': self.hp,
            'best_stats': self.prev_stats["best"],
            'best_epoch': self.best_epoch
        }
        save_model(stats, self.best_model)
