import sys

import torch
from torch.utils.data import DataLoader

from dataset.cct import IrData, IrDataset, ir_collate_fn
from model.ir import TIR
from pre.meta import PretrainedModel
from utils import ROOT_DIR_PATH
from utils.args import get_args
from utils.run import IRTrainer
from utils.sys import setup_seed, build_logger

# Step 0: Set Config
args = get_args(cfg_path=f"{ROOT_DIR_PATH}/model/conf/tir.yaml")
rq1_dir = f"{args.out_dir}/rq1"
args.out_dir = f"{args.out_dir}/rq1/{args.direction}/{args.ptm_name}"
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
data = IrData(total_size=args.total_size, direction=args.direction)
address2id = data.get_address2id(fpath=f"{rq1_dir}/{args.direction}/address2id.json")
train_data = data.get_examples(mode='train')
valid_data = data.get_examples(mode='valid')
test_data = data.get_examples(mode='test')
logger.info(f"Length of Address2id: {len(address2id)}")

# Step 2: Design Model
pm = PretrainedModel(
    task_type=args.task_type,
    model_type=args.ptm_type,
    model_name=args.ptm_name,
    cache_dir=args.ptm_cache_dir,
    device=device,
)
tokenizer, encoder = pm.tokenizer, pm.model
encoder.to(device)
train_dataset = IrDataset(train_data, address2id, tokenizer)
valid_dataset = IrDataset(valid_data, address2id, tokenizer)
test_dataset = IrDataset(test_data, address2id, tokenizer)


train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=ir_collate_fn)
valid_data_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=True, collate_fn=ir_collate_fn)
test_data_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=True, collate_fn=ir_collate_fn)

# Step 2: Design Model
model_args = args.model_args
kv_args, sn_args = model_args.get('kv_encoder'), model_args.get('siamese_net')

model = TIR(
    pre_encoder=encoder, address_num=len(address2id), hid_dim=kv_args.get('hid_dim'),
    embed_dim=kv_args.get('embed_dim'), seq_num=kv_args.get('seq_num'),
    heads_num=kv_args.get('heads_num'), layers_num=kv_args.get('layers_num'),
    dense_hid_dim=sn_args.get('dense_hid_dim'), dense_dropout=sn_args.get('dense_dropout'),
    rff_hid_dim=sn_args.get('rff_hid_dim'), act=sn_args.get('act'), device=device
)
model.to(device)

# Step 3: Train and Test
trainer = IRTrainer(model=model, args=args, logger=logger, device=device)
trainer.train(train_data_loader, valid_data_loader, test_data_loader)

logger.info('=' * 20 + 'Start prediction' + '=' * 20)
trainer.predict(test_data_loader)
