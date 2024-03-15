import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BertModel, BertConfig

from dataset_argparse import max_t_seq_len, max_va_seq_len


class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "glove" or "Bert" or other pretrained language model.
    """
    def __init__(self, hp):
        super(LanguageEmbeddingLayer, self).__init__()
        self.hp = hp

        if self.hp.text_encoder == 'bert':
            bertconfig = BertConfig.from_pretrained('./pretrained-language-models/bert-base-uncased', output_hidden_states=True)
            self.language_model = BertModel.from_pretrained('./pretrained-language-models/bert-base-uncased', config=bertconfig)
        else:
            # text_embedding == 'glove'
            self.embedding = nn.Embedding(self.hp.vocab_size, self.hp.emb_size)

    def forward(self, bert_sent, bert_sent_type, bert_sent_mask):
        if self.hp.text_encoder == 'bert':
            bert_output = self.language_model(
                input_ids=bert_sent,
                attention_mask=bert_sent_mask,
                token_type_ids=bert_sent_type
            )

            # bertmodel会输出 hidden states，以及last hidden state
            # hidden states 包含了13层 transformer encoder 的输出，最后一层为最终的语义表示
            # last hidden state 为最后一层 transformer encoder 的输出，所以通过 bert_output[0] 取出
            bert_output = bert_output[0]    # [batch_size, seq_len, dim(768)]

            # masked mean
            # masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            # mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
            # output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
            return bert_output   # return head (sequence representation)
        else:
            # text_embedding == 'glove'
            # output = self.embedding(sentences)
            # return output
            raise NotImplementedError


class SeqEncoder(nn.Module):
    """Encode all modalities with assigned network. The network will output encoded presentations
    of three modalities. The last hidden of LSTM/GRU generates as input to the control module,
    while separate sequence vectors are received by the transformer.
    TODO: Currently only use one component to encode (coded in "if...else..."). Try to preserve the
    interface of both CNN and LSTM/GRU part. In case one approach can not generate satisfying separate vectors.
    Then activate both components to generate all outputs.

    RNNEncoder(dropout=hp.dropout_v if hp.n_layer > 1 else 0.0)

    Args:
        in_size: input dimension
        hidden_size: hidden layer dimension
        num_layers: specify the number of layers of LSTMs.
        dropout: dropout probability
        bidirectional: specify usage of bidirectional LSTM
    Output:
        (return value in forward) a tensor of shape (batch_size, out_size)
    """
    def __init__(self, hyp_params):
        super(SeqEncoder, self).__init__()
        self.hp = hp = hyp_params

        self.origin_dim_a, self.origin_dim_v = hp.origin_dim_a, hp.origin_dim_v
        self.hidd_dim_a, self.hidd_dim_v = hp.dim_seq_hidden_a, hp.dim_seq_hidden_v
        self.out_dim_a, self.out_dim_v = hp.dim_seq_out_a, hp.dim_seq_out_v

        ############################
        # TODO: use compound mode ##
        ############################
        self.layers = hp.num_seq_layers
        self.bidirectional = hp.bidirectional
        self.drop_linear = hp.drop_linear
        self.drop_seq_a = self.drop_seq_v = hp.drop_seq

        #####################################################################
        # TODO: 1) Use double layer                                         #
        #       2) Keep language unchanged while encode video and accoustic #
        #####################################################################

        self.rnn_v = nn.LSTM(input_size=self.origin_dim_v, hidden_size=self.hidd_dim_v, num_layers=self.layers, bidirectional=self.bidirectional, dropout=self.drop_seq_v, batch_first=False)
        self.rnn_a = nn.LSTM(input_size=self.origin_dim_a, hidden_size=self.hidd_dim_a, num_layers=self.layers, bidirectional=self.bidirectional, dropout=self.drop_seq_a, batch_first=False)

        self.cnn_v = nn.Conv1d(in_channels=self.origin_dim_v, out_channels=self.hidd_dim_v, kernel_size=3, stride=1, padding=1)
        self.cnn_a = nn.Conv1d(in_channels=self.origin_dim_a, out_channels=self.hidd_dim_a, kernel_size=3, stride=1, padding=1)

        # dict that maps modals to corresponding networks
        self.linear_proj_v_h = nn.Linear((2 if self.bidirectional else 1) * self.hidd_dim_v, self.out_dim_v)
        self.linear_proj_a_h = nn.Linear((2 if self.bidirectional else 1) * self.hidd_dim_a, self.out_dim_a)

        self.linear_proj_v_seq = nn.Linear((2 if self.bidirectional else 1) * self.hidd_dim_v, self.out_dim_v)
        self.linear_proj_a_seq = nn.Linear((2 if self.bidirectional else 1) * self.hidd_dim_a, self.out_dim_a)

        self.layer_norm_v = nn.LayerNorm(self.out_dim_v)
        self.layer_norm_a = nn.LayerNorm(self.out_dim_a)

        self.activ = nn.ReLU()
        self.dropout = nn.Dropout(self.drop_linear)

        ##########################################
        #   TODO: add activations\dropout later  #
        ##########################################

        self.proj_v_h = nn.Sequential(self.linear_proj_v_h, self.layer_norm_v)
        self.proj_a_h = nn.Sequential(self.linear_proj_a_h, self.layer_norm_a)

        self.proj_v_seq = nn.Sequential(self.linear_proj_v_seq)
        self.proj_a_seq = nn.Sequential(self.linear_proj_a_seq)

        self.rnn_dict = {'v': self.rnn_v, 'a': self.rnn_a}
        self.proj_dict_h = {'v': self.proj_v_h, 'a': self.proj_a_h}
        self.proj_dict_seq = {'v': self.proj_v_seq, 'a': self.proj_a_seq}

    def forward_rnn_prj(self, input, length, modal):
        assert modal in "va"

        '''
        batch中各个数据的维度是相同的
        为了高效处理, 就需要对样本进行填充(pad_sequence), 使得batch中的各个数据的长度相同, 例如输入的数据x;
        *** 填充之后的样本序列，虽然保证长度相同，但是序列里面可能 padding 很多无效的 0 值，将无效的 padding 值喂给模型进行 forward 会影响模型的效果;
        *** 因此将数据进行padding之后, 送入模型之前, 需要采用 pack_padded_sequence 进行数据压缩(生成 PackedSequence 类), 压缩掉无效的填充值; (让padding值不起作用);
        *** 序列经过模型输出后仍然是压缩序列, 需要使用 pad_packed_sequence 进行解压缩, 就是把原序列填充回来;
        '''
        length = length.to('cpu').to(torch.int64)
        packed_sequence = pack_padded_sequence(input, length, batch_first=False, enforce_sorted=False)  # input:输入数据 lengths:每条数据本身的长度（padding前）
        packed_h, final_h_c_out = self.rnn_dict[modal](packed_sequence)
        padded_h, _ = pad_packed_sequence(packed_h, batch_first=False, total_length=max_t_seq_len - 1 if modal == 'l' else max_va_seq_len)     # (batch_size, seq_len, emb_size)

        h_sent_out = final_h_c_out[0]    # for lstm we don't need the cell state
        h_sent_seq = torch.cat((h_sent_out[0], h_sent_out[1]), dim=-1)     # (batch_size, 2*emb_size)
        h_sent_seq = self.proj_dict_h[modal](h_sent_seq)      # 直接linear降维，然后归一化    原Rnn：先droupout，再linear降维    合并试试，之后还可以加一个激活函数
        h_token_seq = self.proj_dict_seq[modal](padded_h)

        return h_token_seq, h_sent_seq

    def forward_enc(self, input_v, input_a, lengths=None):
        v_token_seq, v_sent_seq = self.forward_rnn_prj(input_v, lengths['v'], modal='v')
        a_token_seq, a_sent_seq = self.forward_rnn_prj(input_a, lengths['a'], modal='a')

        return {'visual': (v_token_seq, v_sent_seq), 'acoustic': (a_token_seq, a_sent_seq)}

    ###################################
    # TODO: Correct input shapes here #
    ###################################
    def forward(self, input_v, input_a, lengths):
        """Encode Sequential data from all modalities
        Params:
            @input_a, input_v (Tuple(Tensor, Tensor)):
            Tuple containing input and lengths of input. The vectors are in the size
            (seq_len, batch_size, embed_size)
        Returns:
            @hidden_dic (dict): A dictionary contains hidden representations of all
            modalities and for each modality the value includes the hidden vector of
            the whole sequence and the final hidden (a.k.a sequence hidden).
            All hidden representations are projected to the same size for transformer
            and its controller use.
        """
        return self.forward_enc(input_v, input_a, lengths)


class TransEncoder(nn.Module):
    def __init__(self, hp):
        super(TransEncoder, self).__init__()
        self.hp = hp

        self.origin_dim_a, self.origin_dim_v = hp.origin_dim_a, hp.origin_dim_v
        self.embed_dim = hp.dim_trans_atten
        self.num_heads_a, self.num_heads_v = hp.num_trans_heads_a, hp.num_trans_heads_v
        self.layers = hp.num_trans_layers
        self.drop_trans = hp.drop_trans

        self.linear_proj_a = nn.Linear(self.origin_dim_a, self.embed_dim)
        self.linear_proj_v = nn.Linear(self.origin_dim_v, self.embed_dim)

        self.activ = nn.ReLU()
        self.dropout = nn.Dropout(self.drop_trans)

        self.transformer_a = FeatureExtract(
            encoder_fea_dim=self.embed_dim,
            nhead=self.num_heads_a,
            dim_feedforward=self.embed_dim,
            num_layers=self.layers,
            drop_out=0.1
        )
        self.transformer_v = FeatureExtract(
            encoder_fea_dim=self.embed_dim,
            nhead=self.num_heads_v,
            dim_feedforward=self.embed_dim,
            num_layers=self.layers,
            drop_out=0.1
        )

        self.modal_dic = {
            'a': {
                'linear': self.linear_proj_a,
                'activ': self.activ,
                'dropout': self.dropout,
                'transformer': self.transformer_a
            },
            'v': {
                'linear': self.linear_proj_v,
                'activ': self.activ,
                'dropout': self.dropout,
                'transformer': self.transformer_v
            }
        }

    def _masked_avg_pool(self, lengths, mask, *inputs):
        """Perform a masked average pooling operation
        Args:
            lengths (Tensor): shape of (batch_size, max_seq_len)
            inputs (Tuple[Tensor]): shape of (batch_size, max_seq_len, embedding)

        """
        res = []
        for t in inputs:
            masked_mul = t * mask   # batch_size, seq_len, emb_size
            res.append(masked_mul.sum(1) / lengths.unsqueeze(-1))
        return res

    def extract_feature(self, inputs, modal_dic, mask):
        '''输入需要提取的模态特征
        Inputs:
            visual\\acoustic:   (seq_len, batch_size, dim_size)
            transformer inputs: (seq_len, batch_size, dim_size)
        Note:
            linear activ dropout layerNorm 搭配使用   \\  直接线性层
        '''
        hidd_data = modal_dic['linear'](inputs)
        token_sem_feature, sent_sem_feature = modal_dic['transformer'](hidd_data, mask)

        return token_sem_feature, sent_sem_feature

    def forward(self, visual, v_len, acoustic, a_len, padding_mask):
        acoustic, visual = acoustic.permute(1, 0, 2), visual.permute(1, 0, 2)   # (batch_size, seq_len, emb_size)

        token_sem_a, sent_sem_a = self.extract_feature(acoustic, self.modal_dic['a'], padding_mask['a'])
        token_sem_v, sent_sem_v = self.extract_feature(visual, self.modal_dic['v'], padding_mask['v'])

        return {'acoustic': (token_sem_a[:, 1:, :].permute(1, 0, 2), sent_sem_a), 'visual': (token_sem_v[:, 1:, :].permute(1, 0, 2), sent_sem_v)}


class PositionEncodingTraining(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, fea_size, tf_hidden_dim, drop_out):
        super().__init__()
        self.cls_token = nn.Parameter(torch.ones(1, 1, tf_hidden_dim))
        num_patches = max_va_seq_len
        self.proj = nn.Linear(fea_size, tf_hidden_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, tf_hidden_dim))
        self.dropout = nn.Dropout(drop_out)

    def forward(self, embeddings):
        batch_size = embeddings.shape[0]
        embeddings = self.proj(embeddings)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TfEncoder(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.2, activation='gelu'):
        super(TfEncoder, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionEncodingTraining(d_model, dim_feedforward, dropout)

        encoder_layers = TransformerEncoderLayer(dim_feedforward, nhead, dim_feedforward, dropout, activation=activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, has_mask=True, src_key_padding_mask=None):
        src = self.pos_encoder(src)

        src = src.transpose(0, 1)
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        output = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=src_key_padding_mask)

        return output.transpose(0, 1)


class FeatureExtract(nn.Module):
    def __init__(self, encoder_fea_dim, nhead, dim_feedforward, num_layers, drop_out=0.5):
        super(FeatureExtract, self).__init__()
        self.encoder = TfEncoder(
            d_model=encoder_fea_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=drop_out,
            activation='gelu'
        )

        self.cls_embedding = nn.Parameter()
        self.layernorm = nn.LayerNorm(dim_feedforward)
        self.dense = nn.Linear(encoder_fea_dim, encoder_fea_dim)
        self.activation = nn.Tanh()

    def forward(self, data, key_padding_mask):
        output = self.encoder(data, has_mask=False, src_key_padding_mask=key_padding_mask)
        token_sem_feature = self.layernorm(output)
        sent_sem_feature = torch.mean(token_sem_feature, dim=-2, keepdim=False)

        return token_sem_feature, sent_sem_feature
