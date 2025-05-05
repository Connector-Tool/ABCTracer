import os
import argparse

import yaml

from utils import ROOT_DIR_PATH


def get_args(cfg_path: str = None):
    parser = argparse.ArgumentParser()
    # System
    parser.add_argument('--use_gpu', default=True, type=bool, help="Whether to train the model on GPU.")
    parser.add_argument('--multi_gpu', default=False, type=bool, help="Ensure multi-gpu training.")
    parser.add_argument('--device_ids', default=[0], type=list, help="GPU index.")
    parser.add_argument('--seed', type=int, default=21, help="Random seed for initialization.")

    # Dataset
    parser.add_argument('--in_dir', default=f"{ROOT_DIR_PATH}/dataset", type=str, help="Folder of input dataset.")
    parser.add_argument('--dataset', default='cct', choices=['cct'],
                        type=str, help="Dataset name.")
    parser.add_argument('--total_size', default=12345678, type=int, help="Size of Dataset.")
    parser.add_argument('--direction', default="bidirectional",
                        choices=['bidirectional', 'forward', 'backward'], type=str, help="Direction of Dataset.")
    parser.add_argument('--out_dir', default=f"{ROOT_DIR_PATH}/out", type=str, help="Folder of output data.")

    # Pre-trained Model
    parser.add_argument('--task_type', default='token_cls', choices=['token_cls'],
                        type=str, help="NLP task type.")
    parser.add_argument('--ptm_name', default='code5t-base',
                        choices=['code5t-base', 'codebert-base', 'graphcodebert-base', 'unixcoder-base'], type=str,
                        help="Pre-trained model name.")
    parser.add_argument('--ptm_type', default='RoBERTa', choices=['Auto', 'BERT', 'RoBERTa'], type=str,
                        help="Pre-trained model type.")
    parser.add_argument('--ptm_cache_dir', default=f"{ROOT_DIR_PATH}/pre/wgt", type=str,
                        help="File location where the pre-trained model is stored.")

    # Model
    parser.add_argument('--rnn', default='LSTM', choices=['RNN', 'GRU', 'LSTM'], type=str)
    parser.add_argument('--crf', default=True, type=bool, help="Whether to use CRF model.")

    parser.add_argument('--model_name', default='tir', choices=['tner', 'tir'],
                        type=str, help="Name of model.")
    parser.add_argument('--model_args', help="Args of Model.")
    parser.add_argument('--model_cache_dir', default=f"{ROOT_DIR_PATH}/out/wgt",
                        type=str, help="File location where the model weight is stored.")

    # Train
    parser.add_argument('--do_train', default=True, type=bool, help="Whether to train a model from scratch.")
    parser.add_argument('--epochs', default=200, type=int, help="Num of training epochs.")
    parser.add_argument('--train_batch_size', default=64, type=int, help="batch size of training.")
    parser.add_argument('--eval_batch_size', default=64, type=int, help="batch size of evaluating.")
    parser.add_argument('--eval_per_epoch', default=1, type=int,
                        help="how often evaluating the trained model on valid dataset during training.")
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'], type=str)
    parser.add_argument('--learning_rate', default=5e-05)
    parser.add_argument('--lr_factor', default=5e-05)
    parser.add_argument('--crf_learning_rate', default=5e-05)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--max_seq_len', default=512, type=int)
    args = parser.parse_args()

    if cfg_path is None:
        return args

    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args
