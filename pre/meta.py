import logging
import os

import torch.nn as nn
from transformers import \
    AutoTokenizer, BertTokenizerFast, RobertaTokenizerFast, AutoModel, RobertaModel, BertModel

from utils import ROOT_DIR_PATH

TOKEN_MODEL_CLASS = {
    'bert': (BertTokenizerFast, BertModel),
    'roberta': (RobertaTokenizerFast, RobertaModel),
    'auto': (AutoTokenizer, AutoModel)
}


TASK_CLASS = {
    'token_cls': TOKEN_MODEL_CLASS
}


class PretrainedModel(nn.Module):
    def __init__(
            self, task_type: str = 'token_cls',
            model_type: str = 'auto', model_name: str = 'codebert-base',
            cache_dir: str = f"{ROOT_DIR_PATH}/pre/wgt/",
            device=None
    ):
        super().__init__()
        self.task_type = task_type.lower()
        self.model_type = model_type.lower()
        self.model_name = model_name.lower()
        self.cache_dir = cache_dir
        self.device = device

        self.tokenizer, self.model = None, None
        self.load_model()

    def load_model(self):
        task_class = TASK_CLASS[self.task_type]
        if self.model_type not in task_class.keys():
            self.model_type = 'auto'
        tokenizer_class, model_class = task_class[self.model_type]
        model_save_path = f"{self.cache_dir}/{self.model_name}"
        if os.path.exists(model_save_path) and len(os.listdir(model_save_path)) > 0:
            self.tokenizer = tokenizer_class.from_pretrained(model_save_path)
            self.model = model_class.from_pretrained(model_save_path)
            logging.info(f'Load the pretrained model {self.model_name}')
        else:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.tokenizer = tokenizer_class.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = model_class.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            logging.info(f'Download and load the pretrained model {self.model_name}')
