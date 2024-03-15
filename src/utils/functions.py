import torch
import torch.nn as nn
import torch.autograd.functional as F
from torch.autograd import Function

"""
Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
"""


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class AttSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, dim):
        attn_weights = F.softmax(input, dim=dim)
        attn_weights_mask = attn_weights.isnan()

        ctx.attn_weights_mask = attn_weights_mask
        # save_for_backward only for saving inputs and outputs
        ctx.dim = dim
        output = torch.masked_fill(attn_weights, attn_weights_mask, 0.0)
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors    # read from tuple

        attn_weights_mask = ctx.attn_weights_mask
        dim = ctx.attn_weights_mask

        grad_output = grad_output * output

        # grad_non_mask = grad_output - grad_output.sum(dim=dim, keepdim=True)
        grad_non_mask = grad_output - grad_output * grad_output.sum(dim=-1).unsqueeze(-1)
        out_grad = torch.masked_fill(grad_non_mask, attn_weights_mask, 0.0)

        return out_grad, dim
