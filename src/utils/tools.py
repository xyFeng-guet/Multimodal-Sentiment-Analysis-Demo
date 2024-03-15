import torch
import os
import io
import pickle
import numpy as np
import pandas as pd


def search_fail_samples(predict, truth, ids, threshold=0.5):
    # predict: [batch_size, 1]
    # truth: [batch_size, 1]
    # ids: [batch_size, 1]
    predict = predict.cpu().detach().numpy()
    truth = truth.cpu().detach().numpy()

    fail_samples = {'acc2': {'vid': [],
                             'pre': [],
                             'tru': []},

                    'acc7': {'vid': [],
                             'pre': [],
                             'tru': []}}

    # 根据判断对错筛选样例
    for i in range(len(predict)):
        if predict[i] * truth[i] < 0:   # 相乘 < 0 说明预测值与真实值异号
            fail_samples['acc2']['vid'].append(ids[i])
            fail_samples['acc2']['pre'].append(predict[i])
            fail_samples['acc2']['tru'].append(truth[i])

    # 根据预测值与真实值的差距筛选样例
    for i in range(len(predict)):
        if abs(predict[i] - truth[i]) > threshold:
            fail_samples['acc7']['vid'].append(ids[i])
            fail_samples['acc7']['pre'].append(predict[i])
            fail_samples['acc7']['tru'].append(truth[i])

    # 两个list中共有的样例
    commen_samples = list(set(fail_samples['acc2']['vid']).intersection(set(fail_samples['acc7']['vid'])))

    return fail_samples, commen_samples


def receive_data(train_data, valid_data, test_data, metric):
    # 查看当前工作路径
    if not os.path.isfile(f'{os.path.join(os.getcwd(), f"picture/{metric}.csv")}'):
        df = pd.DataFrame(columns=['train', 'valid', 'test'])
        df.to_csv(f'{os.path.join(os.getcwd(), f"picture/{metric}.csv")}', index=True)

    df = pd.read_csv(f'{os.path.join(os.getcwd(), f"picture/{metric}.csv")}', index_col=0)
    data = [float(train_data), float(valid_data), float(test_data)]
    # 将data中的数据分别添加到df的三列中
    df = df.append(pd.Series(data, index=df.columns), ignore_index=True)
    df.to_csv(f'{os.path.join(os.getcwd(), f"picture/{metric}.csv")}', index=True)


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    # name = 'best_model'
    if not os.path.exists('pre_trained_models'):
        os.mkdir('pre_trained_models')
    torch.save(model.state_dict(), f'pre_trained_models/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    # name = 'best_model'
    with open(f'pre_trained_models/{name}.pt', 'rb') as f:
        buffer = io.BytesIO(f.read())
    model = torch.load(buffer)
    return model


def get_lens(sents):
    return [len(sent) for sent in sents]


def pad_sents(sents, pad_token):
    sents_padded = []
    lens = get_lens(sents)
    max_len = max(lens)
    sents_padded = [sents[i] + [pad_token] * (max_len - l) for i, l in enumerate(lens)]
    return sents_padded, lens


def get_mask(sents, unmask_idx=1, mask_idx=0):
    lens = get_lens(sents)
    max_len = max(lens)
    mask = [([unmask_idx] * lenth + [mask_idx] * (max_len - lenth)) for lenth in lens]
    return mask
