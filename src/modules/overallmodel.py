import torch
import torch.nn as nn
from dataset_argparse import max_t_seq_len, max_va_seq_len

from modules.encoders import LanguageEmbeddingLayer, SeqEncoder, TransEncoder
from modules.backbone import MultiClass


class _model(nn.Module):
    def __init__(self, hp):
        """Overall model.
        Include Modules:
            1. Modules T V A Encoder
            2. backbone module
        """
        super(_model, self).__init__()
        self.hp = hp

        self.text_encoder = LanguageEmbeddingLayer(self.hp)
        self.semantic_encoder = TransEncoder(self.hp)   # extract semantic feature
        self.sequence_encoder = SeqEncoder(self.hp)  # extract sequence feature

        self.backbone = MultiClass(self.hp)

    def forward(self, batch_data, lengths):
        """Input data dimension:
        text: [batch_size, seq_len, emb_size]
        visual: [seq_len, batch_size, emb_size]
        acoustic: [seq_len, batch_size, emb_size]

        sequence_encoder requires:
        text visual acoustic: [seq_len, batch_size, emb_size]

        semantic_encoder requires:
        text visual acoustic: [batch_size, seq_len, emb_size]
        """
        visual, _, acoustic, _, y, _, text_sent, text_sent_type, text_sent_mask, _ = batch_data
        mask = self._get_mask(text_sent_mask.clone(), lengths)

        text_embedding = self.text_encoder(text_sent, text_sent_type, text_sent_mask)  # [batch_size, seq_len, emb_size]

        # extract semantic feature
        feature_sem = self.semantic_encoder(visual, lengths['v'], acoustic, lengths['a'], mask)   # feature_sem: {'visual': x, 'acoustic': x}
        # extract sequence feature
        feature_seq = self.sequence_encoder(visual, acoustic, lengths)  # feature_seq: {'visual': x, 'acoustic': x, 'text': x}

        input_batch_data = [text_embedding, feature_sem, feature_seq, lengths, mask, y]
        preds = self.backbone(input_batch_data)

        return preds

    def _get_mask(self, text_sent_mask, lengths):
        mask = {}
        mask['t'] = text_sent_mask[:, 1:]
        mask['v'] = torch.zeros(len(lengths['v']) * max_va_seq_len).view(len(lengths['v']), max_va_seq_len).to(self.hp.device)
        mask['a'] = torch.zeros(len(lengths['a']) * max_va_seq_len).view(len(lengths['a']), max_va_seq_len).to(self.hp.device)

        # 按照 TransEncoder 对 Padding 的部分进行标记
        for module, length in lengths.items():
            if module == 't':
                continue
            for i, h in enumerate(length):
                mask[module][i][:h] = False
                mask[module][i][h:] = True
                mask[module][i][0] = False

            mask[module] = torch.cat((mask[module][:, 0:1], mask[module]), dim=-1)
            mask[module] = mask[module].bool()

        return mask
