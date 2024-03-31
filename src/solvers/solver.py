import sys
import copy
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

from .solver_base import SolverBase
from utils.eval_metrics import eval_mosi, eval_mosei_senti, eval_mosei_emo, eval_sims, plot_confuse_metrix
from utils.tools import save_model, search_fail_samples


class Solver(SolverBase):
    def __init__(self, hyp_params, data_loaders):
        super(Solver, self).__init__(hyp_params, data_loaders)
        self.hp = hyp_params
        self.best_epoch = 0

        self.eval_function = {
            "mosi": eval_mosi,
            "mosei_senti": eval_mosei_senti,
            "mosei_emo": eval_mosei_emo,
            "sims": eval_sims
        }

        if self.hp.dataset == "mosei_emo":
            pass
        else:
            #  指标共5个, train valid test 分别为三行
            self.headers = ["Phase", "MAE", "Corr", "Acc7", "Acc2", "Acc2_0", "F1", "F1_0", "Loss"]
            n = len(self.headers) - 1

            # prev_stats = [[-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n]
            prev_stats = [[99.0] * n, [99.0] * n, [99.0] * n]

            self.prev_stats = {
                "train": copy.deepcopy(prev_stats),
                "valid": copy.deepcopy(prev_stats),
                "test": copy.deepcopy(prev_stats),
                "best": copy.deepcopy(prev_stats)
            }

    ####################################################################
    #                                                                  #
    #                 Training and evaluation scripts                  #
    #                                                                  #
    ####################################################################

    def process_model(self):
        early_stop = self.earlyStop
        eval_function = self.eval_function[self.hp.dataset]
        threshold = self.hp.threshold   # 设定错误样例阈值
        scheduler_main = self.scheduler['main']

        for epoch in range(1, self.hp.num_epochs + 1):
            print('\n' + '=' * 10 + ' ' * 6, f'Epoch {epoch}', ' ' * 6 + '=' * 10 + '\n')
            start = time.time()     # 记录每个epoch的开始时间

            train_loss, results_train, truths_train = self.train_tmp(phase='normal')

            val_loss, results_val, truths_val, _ = self.eval_tmp(phase='valid')
            test_loss, results_test, truths_test, allids = self.eval_tmp(phase='test')

            scheduler_main.step(val_loss)    # Decay learning rate by validation loss

            train_stats = eval_function(results_train, truths_train, True) + [round(float(sum(train_loss)), 3)]
            valid_stats = eval_function(results_val, truths_val, True) + [val_loss]
            test_stats = eval_function(results_test, truths_test, True) + [test_loss]

            # 查看错误样例
            if epoch == self.hp.see_fail:
                fail, commen = search_fail_samples(results_test, truths_test, allids, threshold)    # 输出错误样本

                file_acc2 = pd.DataFrame(fail['acc2'], columns=['vid', 'pre', 'tru'])
                file_acc7 = pd.DataFrame(fail['acc7'], columns=['vid', 'pre', 'tru'])
                commen = pd.DataFrame(commen, columns=['vid'])

                # 对file_acc2、file_acc7、commen的vid列进行字典序排序
                file_acc2 = file_acc2.sort_values(by='vid', ascending=True)
                file_acc7 = file_acc7.sort_values(by='vid', ascending=True)
                commen = commen.sort_values(by='vid', ascending=True)

                file_acc2.to_csv(f'/opt/data/private/Experience/MMIM/Experience/{self.hp.dataset.upper()}/fail_acc2.csv', index=False)
                file_acc7.to_csv(f'/opt/data/private/Experience/MMIM/Experience/{self.hp.dataset.upper()}/fail_acc7.csv', index=False)
                commen.to_csv(f'/opt/data/private/Experience/MMIM/Experience/{self.hp.dataset.upper()}/commen.csv', index=False)

                print(f'\nsuccess save fail samples {threshold}\n')

            # 查看混淆矩阵
            if epoch == self.hp.plot_conf_met:
                plot_confuse_metrix(predict=results_test, truth=truths_test, path=self.hp.save_path)
                print('\nsuccess save confuse metrix\n')

            end = time.time()
            duration = end - start
            print('\n' + '=' * 10 + ' ' * 4, f'Use Time: {duration}', ' ' * 4 + '=' * 10 + '\n')

            # 区别多分类任务以及回归任务
            if self.hp.dataset in ['mosi', 'mosei_senti', 'sims']:
                # ======== 输出训练指标数据 ========
                train_stats_str = [str(s)[0:6] for s in train_stats]
                valid_stats_str = [str(s)[0:6] for s in valid_stats]
                test_stats_str = [str(s)[0:6] for s in test_stats]

                print(tabulate([
                    ['Train', *train_stats_str], '\n',
                    ['Valid', *valid_stats_str], '\n',
                    ['Test', *test_stats_str], '\n'
                ], headers=self.headers))

                # ======== 判断使用早停法 ========
                if valid_stats[-1] <= self.prev_stats["best"][1][-1]:     # 比较 valid loss 查看是否为最优模型
                    self.best_epoch = epoch
                    early_stop = self.earlyStop    # 记录早停次数
                    self.prev_stats["best"][1][-1] = valid_stats[-1]
                    self.best_model = copy.deepcopy(self.model.state_dict())
                else:
                    early_stop -= 1
                    if early_stop == 0:
                        print('\n' + '=' * 10 + ' ' * 2, f"Early stopping at epoch {epoch}!", ' ' * 2 + '=' * 10 + '\n')
                        break
            else:
                pass    # dataset == 'mosei_emo'

        '''
        print('=== Best performance ===')   # 输出 Best epoch 各项指标
        self.save_stats()
        self.save_model()
        print('Results and model are saved!')
        '''
        sys.stdout.flush()

    ####################################################################
    #                   do something for pre epoch                     #
    ####################################################################

    def train_tmp(self, phase='normal'):
        epoch_loss = 0.0
        results, truths = [], []

        model = self.model.train()
        data_loader = self.data_loaders['train']

        criterion = self.criterion  # criterion for downstream task
        optimizer = self.optimizer['pretrained'] if phase == 'pretrained' else self.optimizer['model']

        # i_batch 记录使用的第几组batch, 预处理时使用
        for i_batch, batch_data in enumerate(tqdm(data_loader, desc='train')):
            batch_data = [data.cuda() if type(data) != np.ndarray else data for data in batch_data]
            vlens, alens, y, text_sent_mask = batch_data[1], batch_data[3], batch_data[4], batch_data[-2]
            lengths = {'t': text_sent_mask.sum(1), 'v': vlens, 'a': alens}
            batch_size = y.size(0)

            model.zero_grad()
            preds = model(batch_data, lengths)

            if phase == 'pretrained':
                pass
            elif phase == 'normal':    # normal training with contrastive learning
                loss = criterion(preds, y)
                loss.backward()
            else:
                raise ValueError('stage index can either be 0 or 1')

            # Update the model
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
            optimizer.step()

            epoch_loss += loss.item() * batch_size
            results.append(preds)
            truths.append(y)

        avg_loss = epoch_loss / self.hp.n_train
        if phase == 'pretrained':
            results, truths = None, None
        else:
            results = torch.cat(results)
            truths = torch.cat(truths)

        return [avg_loss], results, truths

    def eval_tmp(self, phase='valid'):
        total_loss = 0.0
        results, truths, allids = [], [], []    # allids 用于错误样例分析
 
        model = self.model.eval()
        data_loader = self.data_loaders['test'] if phase == 'test' else self.data_loaders['valid']
        criterion = self.criterion      # criterion for downstream task

        with torch.no_grad():
            for i_batch, batch_data in enumerate(tqdm(data_loader, desc=f'{phase}')):
                batch_data = [data.cuda() if type(data) != np.ndarray else data for data in batch_data]
                vlens, alens, y, text_sent_mask, ids = batch_data[1], batch_data[3], batch_data[4], batch_data[-2], batch_data[-1]
                lengths = {'t': text_sent_mask.sum(1), 'v': vlens, 'a': alens}
                batch_size = y.size(0)

                preds = model(batch_data, lengths)
                total_loss += criterion(preds, y).item() * batch_size

                # Collect the results into ntest if test else self.hp.n_valid)
                results.append(preds)
                truths.append(y)
                allids.append(ids)

        avg_loss = total_loss / (self.hp.n_test if phase == 'test' else self.hp.n_valid)
        results = torch.cat(results)
        truths = torch.cat(truths)
        allids = [item for sublist in allids for item in sublist]   # see_fail 时查看错误样例 id

        return avg_loss, results, truths, allids
