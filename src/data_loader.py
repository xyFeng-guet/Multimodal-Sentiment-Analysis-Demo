import numpy as np
from dataset_argparse import max_t_seq_len, max_va_seq_len
from create_dataset import MOSI, MOSEI, SIMS, PAD, UNK

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel


bert_tokenizer = BertTokenizer.from_pretrained('./pretrained-language-models/bert-base-uncased', do_lower_case=True)     # do_lower_case 为True时，不区分大小写


class MSADataset(Dataset):
    def __init__(self, config):
        super(MSADataset, self).__init__()
        self.config = config

        # Fetch dataset
        if "mosi" in str(config.dataset).lower():
            dataset = MOSI(config)
        elif "mosei" in str(config.dataset).lower():
            dataset = MOSEI(config)
        elif "sims" in str(config.dataset).lower():
            dataset = SIMS(config)
        else:
            raise ValueError("Dataset not recognized.")

        # 使用bert，因此预训练embedding为None
        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode)
        self.len = len(self.data)

        self.config.visual_size = self.data[0][0][1].shape[1]
        self.config.acoustic_size = self.data[0][0][2].shape[1]
        self.config.word2id = self.word2id
        self.config.pretrained_emb = self.pretrained_emb

    @property
    def __getitem__(self, index):
        return self.data[index]
    @property
    def __len__(self):
        return self.len


def get_loader(params, config, shuffle=True):
    """Load DataLoader of given DialogDataset"""
    dataset = MSADataset(config)
    print('=' * 10, f"{config.mode:5} data size: {len(dataset):7} is loaded", '=' * 10)

    config.data_len = len(dataset)
    config.tva_dim = dataset.tva_dim
    if config.mode == 'train':
        params.n_train = len(dataset)
    elif config.mode == 'valid':
        params.n_valid = len(dataset)
    elif config.mode == 'test':
        params.n_test = len(dataset)

    '''
    先执行dataset中的__getitem__, 然后再将__getitem__的输出传给collate_fn函数,
    重新组合batch的形式, 然后再将重新组合的batch数据传给DataLoader, 这才是最终的输入数据, 最终传给模型
    '''
    # 可以不改变原数据集的样式，而是在获得原数据后，改变collate_fn函数来改变batch的样式
    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        按照vid进行分组 -> 按照数据种类进行分组
        一个vid有 8 个数据种类
        '''

        # Rewrite torch.nn.utils.rnn.pad_sequence to deal with 3D tensors of multimodal
        def pad_sequence(sequences, target_len=-1, batch_first=False, padding_value=0.0):
            if target_len < 0:
                max_size = sequences[0].size()
                trailing_dims = max_size[1:]
            else:
                max_size = target_len
                trailing_dims = sequences[0].size()[1:]     # 第一个维度为len，各不一样；[1:]只能取到第二个维度，即dim，所有样本都相同，所以直接[0]

            # max_len = max([s.size(0) for s in sequences])   # 取出所有样本中的len的最大值
            max_len = max_size if target_len >= 0 else max([s.size(0) for s in sequences])

            if batch_first:
                out_dims = (len(sequences), max_len) + trailing_dims
            else:
                out_dims = (max_len, len(sequences)) + trailing_dims

            # 原始思路：将原始数据逐个进行padding，变为max_len，循环很多次
            # 本文思路：直接生成一个全为padding的矩阵，然后再将原数据直接覆盖，没覆盖的都为padding好的
            out_tensor = sequences[0].new_full(out_dims, padding_value)     # 生成一个全为padding的矩阵
            # 把原数据覆盖到全为padding的矩阵中
            for i, tensor in enumerate(sequences):
                # 判断是否全部填充
                if tensor.size(0) > max_len:
                    length = max_len
                    tensor = tensor[:max_len, ...]
                else:
                    length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                # x:length 把数据覆盖到前部分，后部分为padding
                if batch_first:
                    out_tensor[i, :length, ...] = tensor
                else:
                    out_tensor[:length, i, ...] = tensor
            return out_tensor

        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: len(x[0][3]), reverse=True)     # 把一个batch的数据先按照text的长度排序
        v_lens, a_lens, labels, ids = [], [], [], []

        for sample in batch:
            # 非对齐数据中，sample[0]有6个元素，分别是 [], visual, acoustic, text, v_len, a_len （三个模态的len是不对应的）
            # 而在对齐数据中，sample[0]有4个元素，分别是 [], visual, acoustic, text （三个模态的len是对应的）
            # 所以以4为界限，判断是否为对齐数据
            # 若没有对齐，长度不一样，[4]和[5]分别是visual和acoustic的长度
            # 若对齐，t a v 长度一样，所以 [3]text 的长度就代表了 a v 的长度
            if len(sample[0]) > 4:  # unaligned case
                v_lens.append(torch.IntTensor([sample[0][4]]))
                a_lens.append(torch.IntTensor([sample[0][5]]))
            else:   # aligned cases
                v_lens.append(torch.IntTensor([len(sample[0][3])]))
                a_lens.append(torch.IntTensor([len(sample[0][3])]))
            labels.append(torch.from_numpy(sample[1]))
            ids.append(sample[2])

        # 将vlens中大于max_va_seq_len的长度的值都变为max_va_seq_len，其余的值不变，alens同理
        new_vlens = [lengths if lengths[0] <= max_va_seq_len else torch.IntTensor([max_va_seq_len]) for lengths in v_lens]
        new_alens = [lengths if lengths[0] <= max_va_seq_len else torch.IntTensor([max_va_seq_len]) for lengths in a_lens]
        vlens = torch.cat(new_vlens)
        alens = torch.cat(new_alens)
        labels = torch.cat(labels, dim=0)
        ids = np.array(ids)
        # 以上for循环操作，是将一个batch数据中语音长度、视觉长度、标签、id分别取出来；（4个）

        # MOSEI sentiment labels locate in the first column of sentiment matrix
        if labels.size(1) == 7:
            labels = labels[:, 0][:, None]

        # 获取每个vid数据中的句子、视觉、语音模态的信息（3个）
        # sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch], target_len=max_va_seq_len)
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch], target_len=max_va_seq_len)

        # BERT-based features input prep
        SENT_LEN = max_t_seq_len

        # Create text indices using tokenizer   获取最后一个文本的token信息
        details = []
        for sample in batch:
            text = " ".join(sample[0][3])
            encoded_sent = bert_tokenizer.encode_plus(text, max_length=SENT_LEN, add_special_tokens=True, truncation=True, padding='max_length')
            details.append(encoded_sent)

        # Bert things are batch_first
        encoder_sentences = torch.LongTensor([sample["input_ids"] for sample in details])
        encoder_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in details])
        encoder_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in details])

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([len(sample[0][0]) for sample in batch])
        if (vlens <= 0).sum() > 0:
            vlens[np.where(vlens == 0)] = 1     # 若进入，则说明该batch中有数据的视觉模态长度为0，即没有视觉模态

        return visual, vlens, acoustic, alens, labels, lengths, encoder_sentences, encoder_sentence_types, encoder_sentence_att_mask, ids

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=torch.Generator(device='cuda')
    )

    return data_loader
