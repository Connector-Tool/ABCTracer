import json
import os
import random
from abc import ABC, abstractmethod
from itertools import chain
from typing import List, Literal, Tuple, Callable, Generator, Any

from pydantic import BaseModel, Field
from torch.utils.data import Dataset


class MetaExample(BaseModel):
    uid: str = Field(default='')


class MetaData(ABC):
    MAX_SIZE = 12345678

    def __init__(self, source_dir: str, total_size: int = MAX_SIZE):
        if not os.path.exists(source_dir):
            raise FileNotFoundError()
        if not isinstance(total_size, int) or total_size <= 0:
            raise ValueError()

        self.source_dir = source_dir
        self.total_size = total_size

    @staticmethod
    def split_data(
            data: List,
            train_ratio: float = 0.7,
            valid_ratio: float = 0.15,
            test_ratio: float = 0.15
    ) -> Tuple[List, List, List]:
        assert train_ratio + valid_ratio + test_ratio == 1, \
            "Train, validation, and test ratios must sum to 1."

        random.shuffle(data)

        total_size = len(data)
        train_size = int(total_size * train_ratio)
        valid_size = int(total_size * valid_ratio)

        return (
            data[:train_size],
            data[train_size:train_size + valid_size],
            data[train_size + valid_size:]
        )

    def create_train_valid_test(
            self, fpath: str,
            uniform_group: Callable[..., Generator[Any, None, None]] = None
    ):
        assert os.path.isfile(fpath), f"{fpath} does not exist."
        _dir = os.path.dirname(fpath)

        required_files = [f"{_dir}/train-{self.total_size}.json",
                          f"{_dir}/valid-{self.total_size}.json",
                          f"{_dir}/test-{self.total_size}.json"]

        if all(os.path.isfile(file) for file in required_files):
            return

        with open(fpath, 'r', encoding='utf-8') as file:
            data = json.load(file)[:self.total_size]

        if uniform_group:
            split_data = (
                list(chain.from_iterable(self.split_data(sub_data)[0] for sub_data in uniform_group(data))),
                list(chain.from_iterable(self.split_data(sub_data)[1] for sub_data in uniform_group(data))),
                list(chain.from_iterable(self.split_data(sub_data)[2] for sub_data in uniform_group(data))),
            )
        else:
            split_data = self.split_data(data)
        file_names = [f'train-{self.total_size}.json', f'valid-{self.total_size}.json', f'test-{self.total_size}.json']
        for fn, sd in zip(file_names, split_data):
            with open(f"{_dir}/{fn}", 'w', encoding='utf-8') as file:
                json.dump(sd, file, ensure_ascii=False, indent=4)

    @abstractmethod
    def get_examples(self, mode: Literal['train', 'valid', 'test']) -> List[MetaExample]:
        raise NotImplementedError()


class MetaSample(BaseModel):
    uid: str = Field(default='')


class MetaDataset(Dataset):
    def __init__(self):
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample

    @abstractmethod
    def get_samples(self, data: List[MetaExample]) -> List[MetaSample]:
        raise NotImplementedError()
