import math
import torch
import torch.nn as nn
from .cmt.cm_transformer import CrossModuleTransformer


def cos_similar(p, q):
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    return sim_matrix

def cosin(a, b):
    r = cos_similar(a, b)
    list1 = []
    for i in range(len(r)):
        list1.append([r[i][i]])
    r1 = torch.tensor(list1, dtype=torch.float32)
    return r1


class FeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, drop_out=0.1):
        super(FeatureProjector, self).__init__()
        self.feed_foward_size = int(output_dim / 2)
        self.project_size = output_dim - self.feed_foward_size
        self.proj1 = nn.Linear(input_dim, self.feed_foward_size, bias=True)

        self.proj2 = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.proj2.append(nn.Linear(input_dim, self.project_size, bias=False))
            else:
                self.proj2.append(nn.Linear(self.project_size, self.project_size, bias=False))
            self.proj2.append(nn.GELU())

        self.layernorm_ff = nn.LayerNorm(self.feed_foward_size)
        self.layernorm = nn.LayerNorm(self.project_size)
        self.MLP = nn.Sequential(*self.proj2)
        self.drop = nn.Dropout(p=drop_out)

    def forward(self, batch):
        # input: list of data samples with different seq length
        dropped = self.drop(batch)
        ff = self.proj1(dropped)
        x = self.MLP(dropped)
        x = torch.cat([self.layernorm(x), self.layernorm_ff(ff)], dim=-1)
        # return x.transpose(0, 1)  # return shape: [seq,batch,fea]
        return x


class TransformerFusion(nn.Module):
    def __init__(self, hp, dropout_type):
        super(TransformerFusion, self).__init__()
        self.hp = hp
        self.layers = hp.fusion_layers
        self.num_heads = hp.fusion_num_heads
        self.embed_dim = hp.dim_seq_out_v
        self.dropout_type = dropout_type

        self.modal_interaction = nn.ModuleDict({
            'lv': self.get_network(layers=self.layers),
            'la': self.get_network(layers=self.layers)
        })

    def get_network(self, self_type='l', layers=2):
        return CrossModuleTransformer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            layers=layers,
            attn_dropout=self.dropout_type['attn'],
            relu_dropout=self.dropout_type['relu'],
            res_dropout=self.dropout_type['res'],
            embed_dropout=self.dropout_type['embed'],
        )

    def forward(self, seq_l, seq_a, seq_v, lengths, mask):
        last_a2l, last_l2a = self.modal_interaction['la'](seq_l, seq_a, lengths, mask)
        last_v2l, last_l2v = self.modal_interaction['lv'](seq_l, seq_v, lengths, mask)

        # 取出第一个，即cls_token的信息
        cls_a2l = last_a2l.permute(1, 0, 2)[:, 0, :]
        cls_l2a = last_l2a.permute(1, 0, 2)[:, 0, :]
        cls_v2l = last_v2l.permute(1, 0, 2)[:, 0, :]
        cls_l2v = last_l2v.permute(1, 0, 2)[:, 0, :]

        last_hs = torch.cat([cls_a2l, cls_l2a, cls_v2l, cls_l2v], dim=1)

        return last_hs


class SubNet(nn.Module):
    '''The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, n_class):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.linear_project_fusion = nn.Linear(in_size, hidden_size)
        self.linear_project1 = nn.Linear(in_size, hidden_size)
        self.linear_project2 = nn.Linear(hidden_size, hidden_size)
        self.linear_project3 = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, n_class)

    def forward(self, complete_input, fusion_feature):
        '''Args:
            x: tensor of shape (batch_size, in_size)
            t, a, v: tensors of shape (batch_size, length, dim_size)
        '''
        y_cat = torch.cat([complete_input, fusion_feature], dim=1)

        y_fusion = torch.tanh(self.linear_project1(y_cat))
        y_fusion = torch.tanh(self.linear_project2(y_fusion))
        y_fusion = torch.tanh(self.linear_project3(y_fusion))

        pred = self.linear_out(y_fusion)

        return pred
