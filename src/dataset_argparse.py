import os
import sys
import argparse


# 获取当前工作的绝对路径
abs_path = os.path.abspath(os.path.dirname(__file__))


# 给dataloader的collect_fn使用的全局变量，值同get_args
text_encoder = 'bert'   # bert, t5, bart, glove
max_t_seq_len = 51
max_va_seq_len = 50


def get_args(dataset):
    parser = argparse.ArgumentParser(description='Multimodal Sentiment Analysis Demo')

    parser.add_argument('--text_encoder', type=str, default='bert', choices=['bert', 't5', 'bart', 'glove'],
                        help='encoder to use for text (default: bert)')
    parser.add_argument('--bidirectional', action='store_false',
                        help='Whether to use bidirectional rnn')
    parser.add_argument('--dataset', type=str, default=f'{dataset}', choices=['mosi', 'mosei_senti', 'mosei_emo', 'sims'],
                        help='dataset to use (default: mosei)')
    parser.add_argument('--aligned', action='store_true',
                        help='whether to use aligned data')
    parser.add_argument('--save_path', type=str, default='./Plot_Test/picture',
                        help='path for checkpointing')
    parser.add_argument('--no_cuda', action='store_true',
                        help='do not use cuda')

    parser.add_argument('--see_fail', type=int, default=-1,
                        help='see fail sample for which epoch (default: 99)')
    parser.add_argument('--plot_conf_met', type=int, default=-1,
                        help='plot confusion matrix for which epoch (default: 99)')
    parser.add_argument('--threshold', type=float, default=2.5,
                        help='threshold for check faile sample (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')

    return parser


def get_args_mosi(args):
    # Dropouts
    args.add_argument('--drop_linear', type=float, default=0.0,
                      help='dropout of seq encoder linear layer')
    args.add_argument('--drop_seq', type=float, default=0.0,
                      help='dropout of l v a seq encoder')
    args.add_argument('--drop_trans', type=float, default=0.0,
                      help='dropout of transformer encoder')
    args.add_argument('--drop_last_proj', type=float, default=0.0,
                      help='dropout of last projection network')

    # Architecture
    args.add_argument('--num_seq_layers', type=int, default=1,
                      help='number of layers in rnn seq encoders (default: 1)')
    args.add_argument('--dim_seq_hidden_v', type=int, default=32,
                      help='hidden size in visual rnn')
    args.add_argument('--dim_seq_hidden_a', type=int, default=32,
                      help='hidden size in acoustic rnn')
    args.add_argument('--dim_seq_out_v', type=int, default=64,
                      help='output size in visual rnn')
    args.add_argument('--dim_seq_out_a', type=int, default=64,
                      help='output size in acoustic rnn')

    args.add_argument('--num_trans_layers', type=int, default=1,
                      help='number of layers in transformer encoders (default: 1)')
    args.add_argument('--num_trans_heads_a', type=int, default=2,
                      help='number of heads in acoustic multihead attention (default: 1)')
    args.add_argument('--num_trans_heads_v', type=int, default=2,
                      help='number of heads in visual multihead attention (default: 1)')
    args.add_argument('--dim_trans_atten', type=int, default=64,
                      help='the last output dimansion of sequence encoder')

    args.add_argument('--emb_align_layers', type=int, default=3,
                      help='number of layers in alignment model, use to align Bert output dimension of data (default: 2)')
    args.add_argument('--subnet_atten_nhead', type=int, default=2,
                      help='number of heads in subnet multihead attention (default: 1)')
    args.add_argument('--last_dim_proj', type=int, default=256,
                      help='hidden size in last projection network')

    args.add_argument('--fusion_layers', type=int, default=3,
                      help='number of layers in bi-bimodule fusion net')
    args.add_argument('--fusion_num_heads', type=int, default=2,
                      help='number of heads in multihead attention of bi-bimodule fusion net')
    args.add_argument('--fusion_attn_mask', action='store_false',
                      help='whether to use attention mask in bi-bimodule fusion net')

    # Training Setting
    args.add_argument('--batch_size', type=int, default=24, metavar='N',
                      help='batch size (default: 32)')
    args.add_argument('--clip', type=float, default=5.0,
                      help='gradient clip value (default: 0.8)')
    args.add_argument('--lr_bert', type=float, default=5e-5,
                      help='initial learning rate for bert parameters (default: 5e-5)')
    args.add_argument('--lr_pretrained', type=float, default=5e-3,
                      help='initial learning rate for va_encoder parameters (default: 1e-3)')
    args.add_argument('--lr_encoder', type=float, default=8e-5,
                      help='initial learning rate for va_encoder parameters (default: 1e-3)')
    args.add_argument('--lr_backbone', type=float, default=8e-5,
                      help='initial learning rate for main model parameters (default: 1e-3)')

    args.add_argument('--weight_decay_bert', type=float, default=1e-4,
                      help='L2 penalty factor of the main Adam optimizer')
    args.add_argument('--weight_decay_pretrained', type=float, default=1e-4,
                      help='L2 penalty factor of the main Adam optimizer')
    args.add_argument('--weight_decay_main', type=float, default=1e-4,
                      help='L2 penalty factor of the main Adam optimizer')

    args.add_argument('--optim', type=str, default='Adam',
                      help='optimizer to use (default: Adam)')
    args.add_argument('--num_epochs', type=int, default=40,
                      help='number of epochs (default: 40)')
    args.add_argument('--when', type=int, default=5,
                      help='when to decay learning rate (default: 20)')
    args.add_argument('--patience', type=int, default=20,
                      help='when to stop training if best never change')
    args.add_argument('--update_batch', type=int, default=1,
                      help='update batch interval')

    args = args.parse_args()

    return args


def get_args_mosei(args):
    # Dropouts
    args.add_argument('--drop_linear', type=float, default=0.0,
                      help='dropout of seq encoder linear layer')
    args.add_argument('--drop_seq', type=float, default=0.0,
                      help='dropout of l v a seq encoder')
    args.add_argument('--drop_trans', type=float, default=0.0,
                      help='dropout of transformer encoder')
    args.add_argument('--drop_last_proj', type=float, default=0.0,
                      help='dropout of last projection network')

    # Architecture
    args.add_argument('--num_seq_layers', type=int, default=1,
                      help='number of layers in rnn seq encoders (default: 1)')
    args.add_argument('--dim_seq_hidden_v', type=int, default=32,
                      help='hidden size in visual rnn')
    args.add_argument('--dim_seq_hidden_a', type=int, default=32,
                      help='hidden size in acoustic rnn')
    args.add_argument('--dim_seq_out_v', type=int, default=64,
                      help='output size in visual rnn')
    args.add_argument('--dim_seq_out_a', type=int, default=64,
                      help='output size in acoustic rnn')

    args.add_argument('--num_trans_layers', type=int, default=1,
                      help='number of layers in transformer encoders (default: 1)')
    args.add_argument('--num_trans_heads_a', type=int, default=2,
                      help='number of heads in acoustic multihead attention (default: 1)')
    args.add_argument('--num_trans_heads_v', type=int, default=2,
                      help='number of heads in visual multihead attention (default: 1)')
    args.add_argument('--dim_trans_atten', type=int, default=64,
                      help='the last output dimansion of sequence encoder')

    args.add_argument('--emb_align_layers', type=int, default=3,
                      help='number of layers in alignment model, use to align Bert output dimension of data (default: 2)')
    args.add_argument('--subnet_atten_nhead', type=int, default=2,
                      help='number of heads in subnet multihead attention (default: 1)')
    args.add_argument('--last_dim_proj', type=int, default=256,
                      help='hidden size in last projection network')

    args.add_argument('--fusion_layers', type=int, default=3,
                      help='number of layers in bi-bimodule fusion net')
    args.add_argument('--fusion_num_heads', type=int, default=2,
                      help='number of heads in multihead attention of bi-bimodule fusion net')
    args.add_argument('--fusion_attn_mask', action='store_false',
                      help='whether to use attention mask in bi-bimodule fusion net')

    # Training Setting
    args.add_argument('--batch_size', type=int, default=24, metavar='N',
                      help='batch size (default: 32)')
    args.add_argument('--clip', type=float, default=5.0,
                      help='gradient clip value (default: 0.8)')
    args.add_argument('--lr_bert', type=float, default=5e-5,
                      help='initial learning rate for bert parameters (default: 5e-5)')
    args.add_argument('--lr_pretrained', type=float, default=5e-3,
                      help='initial learning rate for va_encoder parameters (default: 1e-3)')
    args.add_argument('--lr_encoder', type=float, default=8e-5,
                      help='initial learning rate for va_encoder parameters (default: 1e-3)')
    args.add_argument('--lr_backbone', type=float, default=8e-5,
                      help='initial learning rate for main model parameters (default: 1e-3)')

    args.add_argument('--weight_decay_bert', type=float, default=1e-4,
                      help='L2 penalty factor of the main Adam optimizer')
    args.add_argument('--weight_decay_pretrained', type=float, default=1e-4,
                      help='L2 penalty factor of the main Adam optimizer')
    args.add_argument('--weight_decay_main', type=float, default=1e-4,
                      help='L2 penalty factor of the main Adam optimizer')

    args.add_argument('--optim', type=str, default='Adam',
                      help='optimizer to use (default: Adam)')
    args.add_argument('--num_epochs', type=int, default=40,
                      help='number of epochs (default: 40)')
    args.add_argument('--when', type=int, default=5,
                      help='when to decay learning rate (default: 20)')
    args.add_argument('--patience', type=int, default=20,
                      help='when to stop training if best never change')
    args.add_argument('--update_batch', type=int, default=1,
                      help='update batch interval')

    args = args.parse_args()

    return args


def get_args_sims(args):
    # Dropouts
    args.add_argument('--drop_linear', type=float, default=0.0,
                      help='dropout of seq encoder linear layer')
    args.add_argument('--drop_seq', type=float, default=0.0,
                      help='dropout of l v a seq encoder')
    args.add_argument('--drop_trans', type=float, default=0.0,
                      help='dropout of transformer encoder')
    args.add_argument('--drop_last_proj', type=float, default=0.0,
                      help='dropout of last projection network')

    # Architecture
    args.add_argument('--num_seq_layers', type=int, default=1,
                      help='number of layers in rnn seq encoders (default: 1)')
    args.add_argument('--dim_seq_hidden_v', type=int, default=32,
                      help='hidden size in visual rnn')
    args.add_argument('--dim_seq_hidden_a', type=int, default=32,
                      help='hidden size in acoustic rnn')
    args.add_argument('--dim_seq_out_v', type=int, default=64,
                      help='output size in visual rnn')
    args.add_argument('--dim_seq_out_a', type=int, default=64,
                      help='output size in acoustic rnn')

    args.add_argument('--num_trans_layers', type=int, default=1,
                      help='number of layers in transformer encoders (default: 1)')
    args.add_argument('--num_trans_heads_a', type=int, default=2,
                      help='number of heads in acoustic multihead attention (default: 1)')
    args.add_argument('--num_trans_heads_v', type=int, default=2,
                      help='number of heads in visual multihead attention (default: 1)')
    args.add_argument('--dim_trans_atten', type=int, default=64,
                      help='the last output dimansion of sequence encoder')

    args.add_argument('--emb_align_layers', type=int, default=3,
                      help='number of layers in alignment model, use to align Bert output dimension of data (default: 2)')
    args.add_argument('--subnet_atten_nhead', type=int, default=2,
                      help='number of heads in subnet multihead attention (default: 1)')
    args.add_argument('--last_dim_proj', type=int, default=256,
                      help='hidden size in last projection network')

    args.add_argument('--fusion_layers', type=int, default=3,
                      help='number of layers in bi-bimodule fusion net')
    args.add_argument('--fusion_num_heads', type=int, default=2,
                      help='number of heads in multihead attention of bi-bimodule fusion net')
    args.add_argument('--fusion_attn_mask', action='store_false',
                      help='whether to use attention mask in bi-bimodule fusion net')

    # Training Setting
    args.add_argument('--batch_size', type=int, default=24, metavar='N',
                      help='batch size (default: 32)')
    args.add_argument('--clip', type=float, default=5.0,
                      help='gradient clip value (default: 0.8)')
    args.add_argument('--lr_bert', type=float, default=5e-5,
                      help='initial learning rate for bert parameters (default: 5e-5)')
    args.add_argument('--lr_pretrained', type=float, default=5e-3,
                      help='initial learning rate for va_encoder parameters (default: 1e-3)')
    args.add_argument('--lr_encoder', type=float, default=8e-5,
                      help='initial learning rate for va_encoder parameters (default: 1e-3)')
    args.add_argument('--lr_backbone', type=float, default=8e-5,
                      help='initial learning rate for main model parameters (default: 1e-3)')

    args.add_argument('--weight_decay_bert', type=float, default=1e-4,
                      help='L2 penalty factor of the main Adam optimizer')
    args.add_argument('--weight_decay_pretrained', type=float, default=1e-4,
                      help='L2 penalty factor of the main Adam optimizer')
    args.add_argument('--weight_decay_main', type=float, default=1e-4,
                      help='L2 penalty factor of the main Adam optimizer')

    args.add_argument('--optim', type=str, default='Adam',
                      help='optimizer to use (default: Adam)')
    args.add_argument('--num_epochs', type=int, default=40,
                      help='number of epochs (default: 40)')
    args.add_argument('--when', type=int, default=5,
                      help='when to decay learning rate (default: 20)')
    args.add_argument('--patience', type=int, default=20,
                      help='when to stop training if best never change')
    args.add_argument('--update_batch', type=int, default=1,
                      help='update batch interval')

    args = args.parse_args()

    return args
