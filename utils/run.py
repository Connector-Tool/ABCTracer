import sys
import time
from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model._meta import MetaModel
from utils.metrics import ner_score_report, cal_sk_metrics


class Trainer(ABC):
    def __init__(self, model: MetaModel, args, logger, device=torch.device('cpu')):
        self.model = model
        self.args = args
        self.logger = logger
        self.device = device

        self.out_dir = args.model_cache_dir
        self.model.to(self.device)
        self.optimizer, self.scheduler = None, None

        # Early stopping
        self.patience = 50
        self.counter = 0
        self.best_loss = None

        # Regular maintenance of checkpoints
        self.save_interval = 5

    @abstractmethod
    def init_optimizer_scheduler(self, model):
        raise NotImplementedError()

    @abstractmethod
    def step(self, batch, no_grad: bool = False):
        raise NotImplementedError()

    @abstractmethod
    def report(self, y_true, y_pred):
        raise NotImplementedError()

    def train(self, train_loader, valid_loader, test_loader=None):
        best_epoch = 0
        best_precision, best_recall, best_f1 = 0.0, 0.0, 0.0
        self.optimizer, self.scheduler = self.init_optimizer_scheduler(self.model)

        self.logger.info('=' * 20 + 'Start training' + '=' * 20)
        try:
            start_epoch, cur_loss = self.model.load_checkpoint(
                self.optimizer, self.scheduler,
                self.out_dir, "checkpoint.pth"
            )
            start_epoch += 1
        except FileNotFoundError:
            start_epoch, cur_loss = 0, 0.
            self.logger.info("Checkpoint not found, starting training from scratch.")

        try:
            for epoch in range(start_epoch, self.args.epochs):
                self.model.train()
                total_loss, total_steps = 0.0, 0
                for step, batch in tqdm(enumerate(train_loader), desc=f'Training epoch [{epoch}/{self.args.epochs}]',
                                        mininterval=0.5, colour='red', leave=False, file=sys.stdout):
                    self.optimizer.zero_grad()
                    loss, _ = self.step(batch)
                    # if self.args.gradient_accumulation_steps > 1:
                    #     loss = loss / self.args.gradient_accumulation_steps

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    total_loss += loss.item()
                    total_steps += 1

                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.scheduler.step()
                        # self.model.zero_grad()
                cur_loss = total_loss / total_steps
                if (epoch + 1) % self.save_interval == 0:
                    self.model.save_checkpoint(
                        self.optimizer, self.scheduler, epoch, cur_loss,
                        self.out_dir, "checkpoint.pth"
                    )

                self.logger.info(
                    '=' * 10 + 'Training epoch %d：step = %d, loss=%.5f' % (
                        epoch, total_steps, cur_loss) + '=' * 10
                )

                if (epoch + 1) % self.args.eval_per_epoch == 0:
                    report = self.evaluate(valid_loader)
                    if test_loader:
                        self.evaluate(test_loader)
                    if report['f1'] > best_f1:
                        best_epoch = epoch
                        best_precision = report['precision']
                        best_recall = report['recall']
                        best_f1 = report['f1']
                        self.logger.info('* Finding new best valid results, wgt model...')
                        self.model.save(self.out_dir, "wgt.pth")

                if self.best_loss is None:
                    self.best_loss = cur_loss
                elif cur_loss > self.best_loss:
                    self.counter += 1
                    self.logger.info(f"Early stopping counter increased: {self.counter} / {self.patience}")
                    if self.counter >= self.patience:
                        break
                else:
                    self.best_loss = cur_loss
                    self.counter = 0

        except Exception as e:
            self.logger.info(f"An exception occurred during training: {e}")

        self.logger.info('=' * 20 + ' End training ' + '=' * 20)
        self.logger.info(
            'Best valid epoch: %d, best valid results: [ Precision: %.5f, Recall: %.5f, F1: %.5f ]' %
            (best_epoch, best_precision, best_recall, best_f1)
        )

    def evaluate(self, loader: DataLoader = None, epoch: int = 0) -> Dict[str, float]:
        self.model.eval()
        Y_true, Y_pred = [], []
        total_loss, total_steps = 0.0, 0

        start_time = time.time()
        for batch_id, batch in tqdm(enumerate(loader), desc=f'Evaluating model',
                                    mininterval=0.5, colour='red', leave=False, file=sys.stdout):
            loss, (y_true, y_pred) = self.step(batch, True)

            total_loss += loss
            total_steps += 1

            Y_true.extend(y_true)
            Y_pred.extend(y_pred)

        report = self.report(Y_true, Y_pred)
        report['loss'] = total_loss / total_steps

        self.logger.info(
            'Evaluating epoch %d：used time = %f, loss = %.5f' %
            (epoch, time.time() - start_time, report['loss'])
        )
        self.logger.info(
            'Precision: %.5f, Recall: %.5f, Accuracy: %.5f, F1: %.5f' %
            (report['precision'], report['recall'], report['accuracy'], report['f1'])
        )
        return report

    def predict(self, loader: DataLoader = None) -> Dict[str, float]:
        self.model.eval()
        Y_true, Y_pred = [], []

        for batch_id, batch in tqdm(enumerate(loader), desc=f'Predicting',
                                    mininterval=0.5, colour='red', leave=False, file=sys.stdout):
            _, (y_true, y_pred) = self.step(batch, True)
            Y_true.extend(y_true)
            Y_pred.extend(y_pred)

        report = self.report(Y_true, Y_pred)

        self.logger.info(
            'Precision: %.5f, Recall: %.5f, Accuracy: %.5f, F1: %.5f' %
            (report['precision'], report['recall'], report['accuracy'], report['f1'])
        )
        return report


class NERTrainer(Trainer):
    def __init__(self, model, args, logger, device=torch.device('cpu'), label2id=None):
        super().__init__(model, args, logger, device)

        # Early stopping
        self.patience = 20
        self.label2id = label2id
        self.id2label = {id: label for label, id in label2id.items()}

    def init_optimizer_scheduler(self, model):
        optimizer = optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return optimizer, scheduler

    def step(self, batch, no_grad: bool = False):
        tokens, labels, token_masks, label_masks = [ts.to(self.device) for ts in batch]

        if no_grad:
            with torch.no_grad():
                loss, logits, labels = self.model(tokens, labels, token_masks, label_masks)
        else:
            loss, logits, labels = self.model(tokens, labels, token_masks, label_masks)

        if not self.args.crf:
            logits = logits.detach().cpu().numpy()

        label_ids = labels.to('cpu').numpy()

        y_true, y_pred = [], []
        for batch_index, seq_label in enumerate(logits):
            temp_true, temp_pred = [], []
            for seq_index, label_id in enumerate(seq_label):
                temp_true.append(self.id2label[label_ids[batch_index][seq_index]])
                temp_pred.append(self.id2label[logits[batch_index][seq_index]])

            y_true.append(temp_true)
            y_pred.append(temp_pred)

        return loss, (y_true, y_pred)

    def report(self, y_true, y_pred):
        return ner_score_report(y_true, y_pred)


class IRTrainer(Trainer):
    def __init__(self, model, args, logger, device=torch.device('cpu')):
        super().__init__(model, args, logger, device)
        self.losser = nn.CrossEntropyLoss()

    def init_optimizer_scheduler(self, model):
        optimizer = optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return optimizer, scheduler

    def step(self, batch, no_grad: bool = False):
        queries, query_types, targets, targets_types, labels = batch
        queries, targets, labels = queries.to(self.device), targets.to(self.device), labels.to(self.device)
        query_types, targets_types = query_types.to(self.device), targets_types.to(self.device)

        n_classes = targets.shape[1]
        indices = (labels >= 0) & (labels < n_classes)
        queries, query_types = queries[indices], query_types[indices]
        targets, targets_types = targets[indices], targets_types[indices]
        labels = labels[indices]

        if no_grad:
            with torch.no_grad():
                y_pred = self.model(queries, query_types, targets, targets_types)
        else:
            y_pred = self.model(queries, query_types, targets, targets_types)
        y_true = labels

        loss = self.losser(y_pred, y_true).mean()
        # loss = list_wise_cross_entropy_loss(y_pred, y_true)
        # return loss, (y_pred, y_true)
        return loss, (torch.argmax(y_pred, dim=1), y_true)

    def report(self, y_true, y_pred):
        # Y_pred = torch.cat(y_pred, dim=0)
        # Y_true = torch.cat(y_true, dim=0)
        # return calculate_metrics(Y_true, Y_pred)
        return cal_sk_metrics(torch.stack(y_true).cpu(), torch.stack(y_pred).cpu())

