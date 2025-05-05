import sys

import torch
from torch.utils.data import DataLoader

from dataset.cct import NerData, NerDataset, ner_collate_fn
from model.ner import TNER
from pre.meta import PretrainedModel
from utils import ROOT_DIR_PATH
from utils.args import get_args
from utils.run import NERTrainer
from utils.sys import setup_seed, build_logger

# Step 0: Set Config
args = get_args(cfg_path=f"{ROOT_DIR_PATH}/model/conf/tner.yaml")
rq2_dir = f"{args.out_dir}/rq2"
args.out_dir = f"{args.out_dir}/rq2/{args.ptm_name}"
args.model_cache_dir = args.out_dir

cuda = torch.cuda.is_available() and args.use_gpu
setup_seed(seed=args.seed, cuda=cuda)
device = torch.device(f'cuda:{args.device_ids[0]}' if cuda else 'cpu')
logger = build_logger(
    name=args.model_name,
    log_dir=f"{args.out_dir}",
    log_type='log'
)
logger.info(sys.argv)
logger.info(args)

# Step 1: Prepare Dataset
data = NerData()
train_data = data.get_examples(mode='train')
valid_data = data.get_examples(mode='valid')
test_data = data.get_examples(mode='test')
entity2id = data.get_entity2id(fpath=f"{rq2_dir}/entity2id.json")
logger.info(f"Entity2Id: {entity2id}")

# Step 2: Design Model
pm = PretrainedModel(
    task_type=args.task_type,
    model_type=args.ptm_type,
    model_name=args.ptm_name,
    cache_dir=args.ptm_cache_dir
)
tokenizer, encoder = pm.tokenizer, pm.model

train_dataset = NerDataset(data=train_data, entity2id=entity2id, tokenizer=tokenizer)
valid_dataset = NerDataset(data=valid_data, entity2id=entity2id, tokenizer=tokenizer)
test_dataset = NerDataset(data=test_data, entity2id=entity2id, tokenizer=tokenizer)

train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=ner_collate_fn)
valid_data_loader = DataLoader(valid_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=ner_collate_fn)
test_data_loader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=ner_collate_fn)

model = TNER(
    enocder=encoder, rnn=args.rnn, crf=args.crf,
    hid_dim=args.model_args['hid_dim'], dropout=args.model_args['dropout'],
    entity2id=entity2id
)

# Step 3: Train and Test
trainer = NERTrainer(args=args, model=model, logger=logger, device=device, label2id=entity2id)
trainer.train(train_data_loader, valid_data_loader, test_data_loader)

logger.info('=' * 20 + 'Start prediction' + '=' * 20)
trainer.predict(test_data_loader)