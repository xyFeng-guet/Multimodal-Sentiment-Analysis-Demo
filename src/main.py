import os
import torch
import time
import numpy as np
import random

from utils import *
from solvers.solver import Solver
from config import get_config, get_hyper_params
from dataset_argparse import get_args, get_args_mosi, get_args_mosei, get_args_sims
from data_loader import get_loader


def get_all_params(dataset='mosi'):
    ####################################################################
    #                                                                  #
    #                      Parameters of Dataset                       #
    #                                                                  #
    ####################################################################

    args = get_args(dataset=f'{dataset}')

    if dataset == 'mosi':
        params = get_args_mosi(args)
    elif dataset in ['mosei_senti', 'mosei_emo']:
        params = get_args_mosei(args)
    elif dataset == 'sims':
        params = get_args_sims(args)
    else:
        raise ValueError("You must choose one of {mosi/mosei_senti/mosei_emo/ur_funny} as your dataset.")

    return params


def os_env_devices(params):
    params.use_cuda = False

    # set manual seed
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    os.environ['PYTHONHASHSEED'] = str(params.seed)
    torch.autograd.set_detect_anomaly(True)  # 检测梯度正向与反向传播异常

    if torch.cuda.is_available():
        torch.cuda.manual_seed(params.seed)
        torch.cuda.manual_seed_all(params.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        params.use_cuda = True
    else:
        torch.set_default_tensor_type('torch.FloatTensor')


def get_all_data_loaders(params):
    # configurations for data_loader
    dataset = str.lower(params.dataset.strip())
    batch_size = params.batch_size

    ####################################################################
    #                                                                  #
    #             Load the dataset (aligned or non-aligned)            #
    #                                                                  #
    ####################################################################

    print('\n' + '=' * 10 + ' ' * 6, "Start loading the data", ' ' * 6 + '=' * 10 + '\n')
    model_config = get_config(dataset)
    data_loaders = get_loader(params, model_config)
    hyp_params = get_hyper_params(params, model_config, dataset)  # 获取所有超参数（根据不同 model）
    print('\n' + '=' * 10 + ' ' * 6, 'Finish loading the data', ' ' * 5 + '=' * 10 + '\n')

    return hyp_params, data_loaders


if __name__ == '__main__':
    start_time = time.time()

    params = get_all_params(dataset='mosi')     # 获取所有参数
    os_env_devices(params)      # 设置随机种子、GPU等环境变量
    hyp_params, data_loaders = get_all_data_loaders(params)  # 获取训练、验证、测试数据

    solver = Solver(hyp_params, data_loaders=data_loaders)
    solver.process_model()

    end_time = time.time()
    print(f'Total time usage = {(end_time - start_time) / 3600:.2f} hours.')  # 总耗时
    exit()
