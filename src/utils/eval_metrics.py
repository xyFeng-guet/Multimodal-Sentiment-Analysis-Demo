import os
import torch
import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score


def plot_confuse_metrix(predict, truth, path, emo=None):
    if emo is None:
        emo = ['-3', '-2', '-1', '0', '1', '2', '3']    # mosi mosei 回归任务 acc7
        # emo = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']    # mosei 情绪多分类任务

    test_preds = predict.view(-1).cpu().detach().numpy()
    test_truth = truth.view(-1).cpu().detach().numpy()

    preds_acc7 = np.clip(test_preds, a_min=-3., a_max=3.)
    truth_acc7 = np.clip(test_truth, a_min=-3., a_max=3.)

    # predict = predict.argmax(-1)
    # truth = truth.argmax(-1)
    preds_acc7 = np.round(preds_acc7) + 3
    truth_acc7 = np.round(truth_acc7) + 3

    cm = confusion_matrix(y_true=truth_acc7, y_pred=preds_acc7, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emo)

    disp.plot(include_values=True, cmap='YlOrRd', ax=None, xticks_rotation='horizontal')

    plt.savefig(os.path.join(path, 'confuse_metrix_mosei.png'))
    plt.show()


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo, verbose=False):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    w_acc = (tp * (n / p) + tn) / (2 * n)

    if verbose:
        fp = n - tn
        fn = p - tp
        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * recall * precision / (recall + precision + 1e-8)
        print('TP=', tp, 'TN=', tn, 'FP=', fp, 'FN=', fn, 'P=', p, 'N', n, 'Recall', recall, "f1", f1)

    return w_acc


def eval_mosi(results, truths, exclude_zero=False):
    eval_state = eval_mosei_senti(results, truths, exclude_zero)
    return eval_state


def eval_mosei_senti(result, truths, exclude_zero=False):
    test_preds = result.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    acc7 = multiclass_acc(test_preds_a7, test_truth_a7)
    acc5 = multiclass_acc(test_preds_a5, test_truth_a5)

    # 是否带0进行判断
    # non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

    # 以下所有指标都为 _non0
    # f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth_non0 = (test_truth[non_zeros] > 0)
    binary_preds_non0 = (test_preds[non_zeros] > 0)
    f1_non0 = f1_score(binary_truth_non0, binary_preds_non0, average='weighted')
    acc2_non0 = accuracy_score(binary_truth_non0, binary_preds_non0)

    # 以下所有指标都为 _has0
    binary_truth_has0 = (test_truth >= 0)
    binary_preds_has0 = (test_preds >= 0)
    f1_has0 = f1_score(binary_truth_has0, binary_preds_has0, average='weighted')
    acc2_has0 = accuracy_score(binary_truth_has0, binary_preds_has0)

    return [mae, corr, acc7, acc2_non0, acc2_has0, f1_non0, f1_has0]


def eval_mosei_emo(preds, truths, threshold, verbose=False):
    '''
    CMU-MOSEI Emotion is a multi-label classification task
    preds: (bs, num_emotions)
    truths: (bs, num_emotions)
    '''

    total = preds.size(0)
    num_emo = preds.size(1)

    preds = preds.cpu().detach()
    truths = truths.cpu().detach()

    preds = torch.sigmoid(preds)

    aucs = roc_auc_score(truths, preds, labels=list(range(num_emo)), average=None).tolist()
    aucs.append(np.average(aucs))

    preds[preds > threshold] = 1
    preds[preds <= threshold] = 0

    accs = []
    f1s = []
    for emo_ind in range(num_emo):
        preds_i = preds[:, emo_ind]
        truths_i = truths[:, emo_ind]
        accs.append(weighted_accuracy(preds_i, truths_i, verbose=verbose))
        f1s.append(f1_score(truths_i, preds_i, average='weighted'))

    accs.append(np.average(accs))
    f1s.append(np.average(f1s))

    acc_strict = 0
    acc_intersect = 0
    acc_subset = 0
    for i in range(total):
        if torch.all(preds[i] == truths[i]):
            acc_strict += 1
            acc_intersect += 1
            acc_subset += 1
        else:
            is_loose = False
            is_subset = False
            for j in range(num_emo):
                if preds[i, j] == 1 and truths[i, j] == 1:
                    is_subset = True
                    is_loose = True
                elif preds[i, j] == 1 and truths[i, j] == 0:
                    is_subset = False
                    break
            if is_subset:
                acc_subset += 1
            if is_loose:
                acc_intersect += 1

    acc_strict /= total  # all correct
    acc_intersect /= total  # at least one emotion is predicted
    acc_subset /= total  # predicted is a subset of truth

    return accs, f1s, aucs, [acc_strict, acc_subset, acc_intersect]


def eval_sims(results, truths, exclude_zero=False):
    return None
