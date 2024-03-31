import torch
from torch import nn
from modules.sub_modules import FeatureProjector, TransformerFusion, SubNet


class MultiClass(nn.Module):
    def __init__(self, hp):
        """Construct MultiMoldal model.
        Args:
            hp (dict): a dict stores training and model configurations
        """
        super(MultiClass, self).__init__()
        self.hp = hp
        dim_sum = hp.dim_seq_out_a + hp.dim_seq_out_v + hp.dim_trans_atten * 2 + hp.origin_dim_t
        dropout_type = {'attn': 0.0, 'relu': 0.0, 'res': 0.0, 'embed': 0.0}

        # align dimension of text to acoustic and visual
        self.embedding_align = FeatureProjector(
            input_dim=hp.origin_dim_t,
            output_dim=hp.dim_trans_atten,
            num_layers=3,
            drop_out=0.0
        )

        # Multimodal Fusion pipline
        self.fusion_net = TransformerFusion(hp=hp, dropout_type=dropout_type)

        # Predict layer Settings
        self.fusion_prj = SubNet(
            in_size=dim_sum + hp.dim_trans_atten * 4,
            hidden_size=hp.last_dim_proj,
            n_class=hp.n_class
        )

    def forward(self, batch_data):
        """
        feature_sem should have dimension [batch_size, seq_len, dim]
        feature_seq should have dimension [seq_len, batch_size, dim]
        """
        feature_text, feature_sem, feature_seq, lengths, mask, label = batch_data
        token_sem_a, sent_sem_a = feature_sem['acoustic']
        token_sem_v, sent_sem_v = feature_sem['visual']
        token_seq_a, sent_seq_a = feature_seq['acoustic']
        token_seq_v, sent_seq_v = feature_seq['visual']

        # == text_word_token
        token_feature_a = torch.cat([token_sem_a, token_seq_a], dim=-1)
        token_feature_v = torch.cat([token_sem_v, token_seq_v], dim=-1)

        # == text_cls_token
        sent_feature_a = torch.cat([sent_sem_a, sent_seq_a], dim=-1)
        sent_feature_v = torch.cat([sent_sem_v, sent_seq_v], dim=-1)

        align_feature_text = self.embedding_align(feature_text).permute(1, 0, 2)     # 将bert输出的维度适配到fusion模块

        # fusion
        fusion_info = self.fusion_net(
            seq_l=align_feature_text,
            seq_a=token_seq_a,
            seq_v=token_seq_v,
            lengths=lengths['t'],
            mask=mask['t']
        )

        # residral proj and pred
        unimodal_info = torch.cat([feature_text[:, 0, :], sent_feature_a, sent_feature_v], dim=-1)
        preds = self.fusion_prj(unimodal_info, fusion_info)

        return preds
