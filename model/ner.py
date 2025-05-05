import torch
from torch import nn
from torch.nn import CrossEntropyLoss, TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

from layers.crf import CRF
from layers.mlp import MLP
from layers.rnn import RNNModel
from model._meta import MetaModel


class TNER(MetaModel):
    def __init__(self, enocder=None, rnn=None, crf=False, hid_dim=768, dropout=0.5, entity2id=None) -> None:
        super().__init__()

        self.tag_num = len(entity2id)
        self.entity2id = entity2id
        self.id2entity = {id: entity for entity, id in entity2id.items()}
        self.encoder = enocder
        self.mlp = MLP(in_dim=hid_dim, hid_dim=256, out_dim=128)

        encoder_layers = TransformerEncoderLayer(128, 8)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 6)
        self.rnn = RNNModel(input_size=128, hidden_size=128, rnn_type=rnn) if rnn else None
        self.crf = CRF(num_tags=self.tag_num, batch_first=True) if crf else None
        self.classifier = MLP(in_dim=128, hid_dim=64, out_dim=self.tag_num)

        # nn.Linear(hid_dim, self.tag_num)
        self.dropout = nn.Dropout(p=dropout)
        self.loss_fct = CrossEntropyLoss()

    def concat_tokens(self, tokens, labels, token_masks, label_masks):
        device = tokens.device
        batch_size, seq_len, embed_size = tokens.size()

        new_tokens, new_labels = [], []
        new_token_masks, new_label_masks = [], []

        for b in range(batch_size):
            output, label = tokens[b], labels[b]
            token_mask, label_mask = token_masks[b], label_masks[b]

            current_output, current_labels = [], []
            current_token_masks, current_label_masks = [], []

            prev_label, segment_tokens = None, []
            for i in range(seq_len):
                current_label = self.id2entity[label[i].item()]

                if current_label == 'O':
                    if prev_label:
                        token_sum = sum(segment_tokens)
                        current_output.append(token_sum)
                        current_labels.append(self.entity2id[prev_label])
                        current_token_masks.append(token_mask[i - 1])
                        current_label_masks.append(label_mask[i - 1])
                        prev_label, segment_tokens = None, []

                    current_output.append(output[i])
                    current_labels.append(label[i])
                    current_token_masks.append(token_mask[i])
                    current_label_masks.append(label_mask[i])
                    continue

                if current_label.startswith('B-'):
                    prev_label = current_label
                segment_tokens.append(output[i])

            if segment_tokens:
                token_sum = sum(segment_tokens)
                current_output.append(token_sum)
                current_labels.append(self.entity2id[prev_label])
                current_token_masks.append(token_mask[-1])
                current_label_masks.append(label_mask[-1])

            new_seq_output_b = torch.stack(current_output) if current_output else output.new_zeros((1, embed_size))
            new_labels_b = torch.tensor(current_labels, device=device) if current_labels else label.new_zeros(1)
            new_token_masks_b = torch.tensor(current_token_masks,
                                             device=device) if current_token_masks else token_mask.new_zeros(1)
            new_label_masks_b = torch.tensor(current_label_masks,
                                             device=device) if current_label_masks else label_mask.new_zeros(1)

            padding_length = seq_len - new_seq_output_b.size(0)
            if padding_length > 0:
                last_token = output[-1, :]
                new_seq_output_b = torch.cat([new_seq_output_b, last_token.unsqueeze(0).expand(padding_length, -1)])
                new_labels_b = torch.cat(
                    [new_labels_b, torch.full((padding_length,), new_labels_b[-1].item(), device=device)])
                new_token_masks_b = torch.cat(
                    [new_token_masks_b, torch.full((padding_length,), 0, device=device)])
                new_label_masks_b = torch.cat(
                    [new_label_masks_b, torch.full((padding_length,), 0, device=device)])

            new_tokens.append(new_seq_output_b)
            new_labels.append(new_labels_b)
            new_token_masks.append(new_token_masks_b)
            new_label_masks.append(new_label_masks_b)

        new_tokens = torch.stack(new_tokens)  # (batch_size, seq_len, embed_size)
        new_labels = torch.stack(new_labels)  # (batch_size, seq_len)
        new_token_masks = torch.stack(new_token_masks)  # (batch_size, seq_len)
        new_label_masks = torch.stack(new_label_masks)  # (batch_size, seq_len)

        return new_tokens.to(device), new_labels.to(device), new_token_masks.to(device), new_label_masks.to(device)

    def forward(self, tokens, labels, token_masks, label_masks):
        # (batch_size, seq_len, hid_dim)
        tokens = self.encoder(tokens, attention_mask=token_masks)[0]
        tokens, labels, token_masks, label_masks = self.concat_tokens(tokens, labels, token_masks, label_masks)

        tokens = self.mlp(tokens)
        tokens = tokens.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        tokens = self.transformer_encoder(tokens)  # (seq_len, batch_size, embed_dim)
        tokens = tokens.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)

        if self.rnn:
            tokens, sequence_hn = self.rnn(tokens, token_masks)
        tokens = self.dropout(tokens)
        # (batch_size, seq_len, tag_num)
        logits = self.classifier(tokens)

        outputs = ()
        if labels is not None:
            if self.crf is not None:
                loss = self.crf(emissions=logits, tags=labels, mask=label_masks.eq(1)) * (-1)
            else:
                if label_masks is not None:
                    active_loss = label_masks.view(-1) == 1
                    active_logits = logits.view(-1, self.tag_num)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = self.loss_fct(active_logits, active_labels)
                else:
                    loss = self.loss_fct(logits.view(-1, self.tag_num), labels.view(-1))
            outputs = (loss, )

        if self.crf is not None:
            # (batch_size, seq_len)
            logits = self.crf.decode(emissions=logits, mask=label_masks.eq(1))
        else:
            # (batch_size, seq_len)
            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        outputs = outputs + (logits, labels)
        return outputs
