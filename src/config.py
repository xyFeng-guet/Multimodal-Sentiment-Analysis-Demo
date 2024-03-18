import argparse
import pprint
from pathlib import Path
import torch.nn as nn
from torch import optim


# 获取当前工作路径
username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent.parent


class Config(object):
    def __init__(self, data, mode='train'):
        """Configuration Class: set kwargs as class attributes with setattr"""
        data_dir = project_dir.joinpath('Datasets/CMU-origin-datasets')    # 设置数据读取路径
        data_dict = {
            'mosi': data_dir.joinpath('MOSI'),
            'mosei': data_dir.joinpath('MOSEI'),
            'sims': data_dir.joinpath('CH-SIMS')
        }
        self.dataset_dir = data_dict[data.lower()]      # dataset_dir.joinpath('align_label') if params.aligned else dataset_dir.joinpath('no_align_label')
        self.sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')
        self.mode = mode
        self.dataset = data
        self.word_emb_path = project_dir.joinpath('pretrained-language-models/glove.840B.300d.txt')   # path to a pretrained word embedding file
        self.shuffle = bool(True) if mode == 'train' else False

    def train_tools_retrieve(self, dataset_name):
        # 设置训练工具字典
        optimizer_dict = {
            'RMSprop': optim.RMSprop,
            'Adam': optim.Adam
        }
        activation_dict = {
            'elu': nn.ELU,
            "hardshrink": nn.Hardshrink,
            "hardtanh": nn.Hardtanh,
            "leakyrelu": nn.LeakyReLU,
            "prelu": nn.PReLU,
            "relu": nn.ReLU,
            "rrelu": nn.RReLU,
            "tanh": nn.Tanh
        }
        output_dim_dict = {
            'mosi': 1,
            'mosei_senti': 1,
            'mosei_emotion': 6,
            'sims': 1
        }
        criterion_dict = {
            'mosi': 'L1Loss',
            'mosei_senti': 'L1Loss',
            'mosei_emotion': 'CrossEntropyLoss',
            'sims': 'L2Loss'
        }

        tool = None
        return tool

    def str2bool(v):
        """string to boolean"""
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


def get_config(dataset='mosi'):
    train_config = Config(data=dataset, mode='train')
    valid_config = Config(data=dataset, mode='valid')
    test_config = Config(data=dataset, mode='test')
    model_config = {'train': train_config, 'valid': valid_config, 'test': test_config}
    return model_config


def get_hyper_params(params, model_config, dataset):
    ####################################################################
    #                       Hyper_parameters
    ####################################################################
    hyp_params = params

    # addintional appending
    hyp_params.word2id = model_config['train'].word2id
    hyp_params.pretrained_emb = model_config['train'].pretrained_emb

    # architecture parameters
    hyp_params.origin_dim_t, hyp_params.origin_dim_v, hyp_params.origin_dim_a = model_config['train'].tva_dim
    hyp_params.len_t, hyp_params.len_a, hyp_params.len_v = model_config['train'].tva_len

    hyp_params.use_cuda = params.use_cuda
    hyp_params.when = params.when
    hyp_params.dataset = dataset
    hyp_params.n_class = model_config.train_tools_retrieve(dataset, 1)
    hyp_params.criterion = model_config.train_tools_retrieve(dataset, 'MSELoss')

    return hyp_params
