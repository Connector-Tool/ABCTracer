import warnings
from typing import List, Dict
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def scores_convert_to_binary_label(scores):
    max_indices = torch.argmax(scores, dim=1)
    binary_labels = torch.zeros_like(scores)
    binary_labels[torch.arange(scores.size(0)), max_indices] = 1.0

    return binary_labels


def label_convert_to_binary_label(indices, target_size: int):
    binary_labels = torch.zeros((indices.size(0), target_size), dtype=torch.float32)
    binary_labels[torch.arange(indices.size(0)), indices] = 1.0
    return binary_labels


def calculate_metrics(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred > threshold).float()  # (batch_size, target_num)

    # Calculate TP, FP, TN, FN
    TP = (y_true * y_pred_binary).sum(dim=0)  # True Positives
    FP = ((1 - y_true) * y_pred_binary).sum(dim=0)  # False Positives
    TN = ((1 - y_true) * (1 - y_pred_binary)).sum(dim=0)  # True Negatives
    FN = (y_true * (1 - y_pred_binary)).sum(dim=0)  # False Negatives

    # Calculate Precision, Recall, Accuracy
    precision = TP / (TP + FP + 1e-10)  # Add small constants to prevent division by zero
    recall = TP / (TP + FN + 1e-10)
    accuracy = (TP + TN) / (TP + FP + TN + FN + 1e-10)

    # Calculate F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Calculate Accuracy
    # correct_predictions = (y_true * y_pred_binary).sum()
    # total_predictions = y_true.numel()
    # accuracy = correct_predictions / total_predictions

    return {
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'accuracy': accuracy.mean().item(),
        'f1': f1_score.mean().item()
    }


def cal_sk_metrics(y_true, y_pred):
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()

    precision = precision_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np, average='weighted')

    return {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1
    }


def ner_score_report(label_true: List[List[str]], label_pred: List[List[str]]) -> Dict:
    nb_correct, nb_pred, nb_true = extract_tp_actual_correct(label_true, label_pred)

    precision = nb_correct / nb_pred if nb_correct > 0 else 0.0
    recall = nb_correct / nb_true if nb_correct > 0 else 0.0
    total_samples = sum(1 for labels in label_pred for label in labels if label != 'O')
    accuracy = nb_correct / total_samples if total_samples > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if nb_correct > 0 else 0.0

    cls_score = {'correct_num': nb_correct, 'gold_num': nb_true, 'pred_num': nb_pred,
                 'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1': f1}

    return cls_score


def extract_tp_actual_correct(label_true: List[List[str]], label_pred:List[List[str]], suffix: bool=False, *args):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)

    for type_name, start, end in get_entities(label_true, suffix):
        entities_true[type_name].add((start, end))
    for type_name, start, end in get_entities(label_pred, suffix):
        entities_pred[type_name].add((start, end))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
        pred_sum = np.append(pred_sum, len(entities_pred_type))
        true_sum = np.append(true_sum, len(entities_true_type))

    tp_sum = np.sum(tp_sum)
    pred_sum = np.sum(pred_sum)
    true_sum = np.sum(true_sum)

    return tp_sum, pred_sum, true_sum


# copy from 'seqeval'
def get_entities(seq, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """

    def _validate_chunk(chunk, suffix):
        if chunk in ['O', 'B', 'I', 'E', 'S', 'M']:
            return

        if suffix:
            if not chunk.endswith(('-B', '-I', '-E', '-S', '-M')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

        else:
            if not chunk.startswith(('B-', 'I-', 'E-', 'S-', 'M-')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        _validate_chunk(chunk, suffix)

        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit('-', maxsplit=1)[0] or '_'
        else:
            tag = chunk[0]  # 'B'
            type_ = chunk[1:].split('-', maxsplit=1)[-1] or '_'  # 'LOC'

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_
    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start