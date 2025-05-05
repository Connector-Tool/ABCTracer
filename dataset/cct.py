import csv
import json
import os
import re
from abc import abstractmethod
from collections import defaultdict
from enum import Enum
from typing import List, Literal, Dict, Union, Any

import torch
from pydantic import BaseModel, Field, field_validator

from dataset import RAW_DIR_PATH
from dataset._meta import MetaExample, MetaData, MetaDataset, MetaSample
from utils.data import CLS_TOKEN, SEP_TOKEN, bin_to_int, num_to_fixed_len_bin, pad_with_zeros, find_sublist_indices


class Kv(BaseModel):
    key: str = Field(default='')
    val: Union[str, int, float, Any] = Field(default='')

    class Type(Enum):
        NUMBER = 1
        STRING = 2
        ADDRESS = 3
        OTHER = 4

    @staticmethod
    def is_address(address: str) -> bool:
        if not isinstance(address, str):
            return False
        pattern = r'^0x[a-fA-F0-9]{40}$'
        return bool(re.match(pattern, address))

    @staticmethod
    def is_numeric(number: Union[int, float, str]) -> bool:
        if isinstance(number, (int, float)):
            return True
        if not number.strip():
            return False
        # Check if the string is a valid hexadecimal number
        if re.fullmatch(r'0[xX][0-9a-fA-F]+', number):
            return True
        try:
            float(number)
            return True
        except ValueError:
            return False

    def val_type(self):
        if not isinstance(self.val, (int, float, str)):
            return Kv.Type.OTHER

        if self.is_address(self.val):
            return Kv.Type.ADDRESS
        if self.is_numeric(self.val):
            return Kv.Type.NUMBER
        return Kv.Type.STRING


class Input(BaseModel):
    func: str = Field(default='')
    param: List[Kv] = Field(default=[])

    @field_validator('param', mode='before')
    def validate_param(cls, value):
        if isinstance(value, dict):
            return [Kv(key=k, val=v) for k, v in value.items()]
        return value


class Event(BaseModel):
    event: str = Field(default='')
    param: List[Kv] = Field(default=[])

    @field_validator('param', mode='before')
    def validate_param(cls, value):
        if isinstance(value, dict):
            return [Kv(key=k, val=v) for k, v in value.items()]
        return value


class Tx(BaseModel):
    hash: str = Field(default='')
    input: Input = Field(default=None)
    event_logs: List[Event] = Field(default=[])

    def get_kvs(self):
        return [
            kv
            for kv in self.input.param
            if kv.val_type() != Kv.Type.OTHER

        ] + [
            kv
            for e in self.event_logs
            for kv in e.param
            if kv.val_type() != Kv.Type.OTHER
        ]


class Pair(BaseModel):
    src_net: str = Field(default='')
    dst_net: str = Field(default='')
    bridge: str = Field(default='')
    src_tx: Tx = Field(default=None)
    dst_tx: Tx = Field(default=None)


class IrExample(MetaExample):
    q_net: str = Field(default='')
    t_net: str = Field(default='')
    query: Tx = Field(default=None)
    targets: List[Tx] = Field(default=[])
    interval: int = Field(default=30)
    label: int = Field(default=-1)


class NerExample(MetaExample):
    tokens: List[str] = Field(default=[])
    labels: List[str] = Field(default=[])


class CctData(MetaData):
    def __init__(self, source_dir: str = f"{RAW_DIR_PATH}/cct", total_size: int = MetaData.MAX_SIZE):
        super().__init__(f"{source_dir}", total_size)
        os.makedirs(self.source_dir, exist_ok=True)

    def get_tx(self, net: Literal['eth', 'bsc', 'pol'], hash: str) -> Tx:
        net, hash = net.lower(), hash.lower()
        with open(f"{self.source_dir}/tx/{net}/{hash}.json", 'r', encoding='utf-8') as file:
            content = file.read().strip()
            data = json.loads(content)
            return Tx(**data)

    def get_pairs(self) -> List[Pair]:
        with open(f"{self.source_dir}/pairs.csv", mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            raw_pairs = [row for row in reader]

        pairs = []
        for p in raw_pairs:
            src_net, dst_net = p.get('Src').lower(), p.get('Dst').lower()
            src_tx_hash = p.get('SrcTxHash').lower()
            dst_tx_hash = p.get('DstTxHash').lower()

            txs = [
                self.get_tx(net, hash)
                for net, hash in zip([src_net, dst_net], [src_tx_hash, dst_tx_hash])
            ]
            pairs.append(Pair(
                src_net=src_net,
                dst_net=dst_net,
                bridge=p.get('Bridge'),
                src_tx=txs[0],
                dst_tx=txs[1]
            ))
        return pairs

    def _get_address2id(self, txs):
        kvs = [
            kv
            for tx in txs
            for kv in tx.get_kvs()
        ]
        return {
            addr: i + 1
            for i, addr in enumerate(set([
                kv.val.lower()
                for kv in kvs
                if kv.val_type() == Kv.Type.ADDRESS
            ]))
        }

    @abstractmethod
    def get_examples(self, mode: Literal['train', 'valid', 'test']) -> List[MetaExample]:
        raise NotImplementedError()


class IrData(CctData):
    def __init__(
            self,
            source_dir: str = f"{RAW_DIR_PATH}/cct",
            total_size: int = MetaData.MAX_SIZE,
            direction: Literal['bidirectional', 'forward', 'backward', 'per_25'] = 'bidirectional'
    ):
        super().__init__(source_dir, total_size)
        self.ir_dir = f"{self.source_dir}/ir/{direction}"
        os.makedirs(self.ir_dir, exist_ok=True)

        self.create_train_valid_test(
            fpath=f"{self.ir_dir}/all.json",
            uniform_group=self.uniform_group
        )

    @staticmethod
    def uniform_group(data: List[Dict]):
        grouped_data = defaultdict(list)
        for item in data:
            grouped_data[len(item['targets'])].append(item)
        yield from grouped_data.values()

    def get_address2id(self, fpath: str = None):
        if fpath is not None and os.path.exists(fpath):
            with open(fpath, 'r') as json_file:
                address2id = json.load(json_file)
            if len(address2id) != 0:
                return address2id

        hash_map = {'eth': set(), 'bsc': set(), 'pol': set()}
        for m in ['train', 'valid', 'test']:
            with open(f"{self.ir_dir}/{m}-{self.total_size}.json", 'r', encoding='utf-8') as file:
                data = json.load(file)

            for n in ['eth', 'bsc', 'pol']:
                hash_map[n].update([e['query'].lower() for e in data if e['q_net'].lower() == n])
                hash_map[n].update([h.lower() for e in data if e['t_net'].lower() == n for h in e['targets']])

        txs = [
            self.get_tx(n, h)
            for n in ['eth', 'bsc', 'pol']
            for h in hash_map[n]
        ]
        address2id = self._get_address2id(txs)
        _dir = os.path.dirname(fpath)
        os.makedirs(_dir, exist_ok=True)
        with open(fpath, 'w') as json_file:
            json.dump(address2id, json_file, indent=4)

        return address2id

    def get_examples(self, mode: Literal['train', 'valid', 'test']) -> List[MetaExample]:
        with open(f"{self.ir_dir}/{mode}-{self.total_size}.json", 'r', encoding='utf-8') as file:
            data = json.load(file)

        query = [self.get_tx(e.get('q_net'), e.get('query')) for e in data]
        targets = [[self.get_tx(e.get('t_net'), t) for t in e.get('targets')] for e in data]

        return [
            IrExample(
                uid='%s-%s' % (mode, i),
                q_net=d.get('q_net'),
                t_net=d.get('t_net'),
                query=q,
                targets=t,
                interval=d.get('interval'),
                label=d.get('label')
            )
            for i, (d, q, t) in enumerate(zip(data, query, targets))
        ]


class NerData(CctData):
    def __init__(self, source_dir: str = f"{RAW_DIR_PATH}/cct", total_size: int = MetaData.MAX_SIZE):
        super().__init__(source_dir, total_size)
        self.ner_dir = f"{self.source_dir}/ner"
        os.makedirs(self.ner_dir, exist_ok=True)
        self.create_train_valid_test(fpath=f"{self.ner_dir}/all.json")

    @staticmethod
    def tokenize(text: str):
        return re.findall(r'\w+|[(),]', text)

    def to_bio_format(self, text: str, entities: List[Dict]):
        tokens = self.tokenize(text)
        labels = ['O'] * len(tokens)

        for entity in entities:
            entity_name = entity['name']
            entity_type = entity['type']

            entity_tokens = self.tokenize(entity_name)
            pos = find_sublist_indices(tokens, entity_tokens)

            if pos is None:
                continue

            start, end = pos
            labels[start] = f'B-{entity_type}'
            labels[start + 1:end + 1] = [f'I-{entity_type}'] * (end - start)
        return tokens, labels

    def get_entity2id(self, fpath: str = None):
        if fpath is not None and os.path.exists(fpath):
            with open(fpath, 'r') as json_file:
                entity2id = json.load(json_file)
            if len(entity2id) != 0:
                return entity2id

        entities = set()

        with open(f"{self.ner_dir}/all.json", 'r', encoding='utf-8') as file:
            data = json.load(file)

        entities.update(
            {f'B-{e["type"]}' for d in data for e in d.get('entities', [])}
        )
        entities.update(
            {f'I-{e["type"]}' for d in data for e in d.get('entities', [])}
        )

        entity2id = {'O': 0}
        entity2id.update({e: i + 1 for i, e in enumerate(entities)})

        _dir = os.path.dirname(fpath)
        os.makedirs(_dir, exist_ok=True)
        with open(fpath, 'w') as json_file:
            json.dump(entity2id, json_file, indent=4)
        return entity2id

    def get_examples(self, mode: Literal['train', 'valid', 'test']) -> List[MetaExample]:
        with open(f"{self.ner_dir}/{mode}-{self.total_size}.json", 'r', encoding='utf-8') as file:
            data = json.load(file)
        return [
            NerExample(
                uid=f'{mode}-{i}',
                tokens=bio_format[0],
                labels=bio_format[1],
            )
            for i, e in enumerate(data)
            if (bio_format := self.to_bio_format(e['text'], e['entities']))
        ]


class IrSample(MetaSample):
    query: List[List[int]]
    query_type: List[int]
    targets: List[List[List[int]]]
    targets_type: List[List[int]]
    label: int


class IrDataset(MetaDataset):
    def __init__(
            self, data: List[MetaExample], address2id: Dict, tokenizer=None,
            token_size: int = 8, kv_size: int = 12 * 8, target_num: int = 64
    ):
        super().__init__()
        self.token_size = token_size
        self.kv_size = kv_size
        self.target_num = target_num
        self.address2id = address2id
        self.tokenizer = tokenizer
        self.samples = self.get_samples(data)

    def get_str_id(self, word: str) -> List[int]:
        tokens = [CLS_TOKEN] + self.tokenizer.tokenize(word)[:self.token_size-2] + [SEP_TOKEN]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        padding_needed = self.token_size - len(token_ids)
        if padding_needed > 0:
            sep_id = self.tokenizer.convert_tokens_to_ids([SEP_TOKEN])
            token_ids.extend(sep_id * padding_needed)

        return token_ids

    def get_addr_id(self, address: str) -> List[int]:
        return [self.address2id.get(address.lower(), 0)] + [0] * (self.token_size - 1)

    def get_num_id(self, number: Union[int, float, str]):
        # Check if the input is a hexadecimal string
        if isinstance(number, str) and re.fullmatch(r'0[xX][0-9a-fA-F]+', number):
            number = int(number, 16)
        else:
            number = float(number)
        return [bin_to_int(num_to_fixed_len_bin(number))] + [0] * (self.token_size - 1)

    def get_tx_id(self, tx: Tx):
        kvs = tx.get_kvs()[:self.kv_size]

        val_type_to_id_func = {
            Kv.Type.ADDRESS: self.get_addr_id,
            Kv.Type.STRING: self.get_str_id,
            Kv.Type.NUMBER: self.get_num_id
        }

        tx_ids = [
            self.get_str_id(kv.key) +
            val_type_to_id_func[kv.val_type()](kv.val)
            for kv in kvs
        ]

        padding_needed = self.kv_size - len(tx_ids)
        if padding_needed > 0:
            padding = [[0] * (self.token_size * 2)] * padding_needed
            tx_ids.extend(padding)
        return tx_ids

    def get_tx_type_id(self, tx: Tx):
        kvs = tx.get_kvs()[:self.kv_size]
        type_ids = [kv.val_type().value for kv in kvs]

        padding_needed = self.kv_size - len(type_ids)
        if padding_needed > 0:
            padding = [0] * padding_needed
            type_ids.extend(padding)
        return type_ids

    def get_samples(self, data: List[MetaExample]) -> List[MetaSample]:
        samples = []
        for e in data:
            targets_items = e.targets[:self.target_num]
            targets = [self.get_tx_id(t) for t in targets_items][:self.target_num]
            targets = pad_with_zeros(targets, [self.target_num, self.kv_size, self.token_size*2])

            targets_type = [self.get_tx_type_id(t) for t in targets_items]
            targets_type = pad_with_zeros(targets_type, [self.target_num, self.kv_size])

            samples.append(
                IrSample(
                    uid=e.uid,
                    query=self.get_tx_id(e.query),
                    query_type=self.get_tx_type_id(e.query),
                    targets=targets,
                    targets_type=targets_type,
                    label=e.label,
                )
            )
        return samples


def ir_collate_fn(batch: List[IrSample]):
    queries = torch.tensor([e.query for e in batch], dtype=torch.long)
    query_types = torch.tensor([e.query_type for e in batch], dtype=torch.long)
    targets = torch.tensor([e.targets for e in batch], dtype=torch.long)
    targets_types = torch.tensor([e.targets_type for e in batch], dtype=torch.long)
    labels = torch.tensor([e.label for e in batch], dtype=torch.long)

    return queries, query_types, targets, targets_types, labels


class NerSample(MetaSample):
    tokens: List[int]
    labels: List[int]
    token_masks: List[int]
    label_masks: List[int]


class NerDataset(MetaDataset):
    def __init__(
            self, data: List[MetaExample], entity2id: Dict,
            tokenizer=None, token_size: int = 64
    ):
        super().__init__()
        self.token_size = token_size
        self.entity2id = entity2id
        self.tokenizer = tokenizer
        self.samples = self.get_samples(data)

    def get_samples(self, data: List[MetaExample]) -> List[MetaSample]:
        samples = []
        for e in data:
            new_tokens = [CLS_TOKEN]
            new_labels = [self.entity2id['O']]
            for token, label in zip(e.tokens, e.labels):
                sub_tokens = self.tokenizer.tokenize(token)
                new_tokens.extend(sub_tokens)
                if label.startswith('B-'):
                    new_labels.append(self.entity2id[label])
                    new_labels.extend([self.entity2id[label.replace("B-", "I-")]] * (len(sub_tokens) - 1))
                else:
                    new_labels.extend([self.entity2id[label]] * len(sub_tokens))
            new_tokens = new_tokens[:self.token_size-1]
            new_labels = new_labels[:self.token_size-1]

            padding_needed = self.token_size - len(new_tokens)
            if padding_needed > 0:
                new_tokens.extend([SEP_TOKEN] * padding_needed)
                new_labels.extend([self.entity2id['O']] * padding_needed)

            new_tokens = self.tokenizer.convert_tokens_to_ids(new_tokens)
            samples.append(
                NerSample(
                    uid=e.uid,
                    tokens=new_tokens,
                    labels=new_labels,
                    token_masks=[1] * len(new_tokens),
                    label_masks=[1] * len(new_labels)
                )
            )
        return samples


def ner_collate_fn(batch: List[NerSample]):
    tokens = torch.tensor([e.tokens for e in batch], dtype=torch.long)
    labels = torch.tensor([e.labels for e in batch], dtype=torch.long)
    token_masks = torch.tensor([e.token_masks for e in batch], dtype=torch.long)
    label_masks = torch.tensor([e.label_masks for e in batch], dtype=torch.long)

    return tokens, labels, token_masks, label_masks
