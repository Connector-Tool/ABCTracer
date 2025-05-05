import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from dataset.cct import Kv
from layers.mlp import MLP
from model._meta import MetaModel


class AddressEncoder(nn.Module):
    def __init__(self, address_num: int, embed_dim: int):
        super().__init__()
        self.address_num = address_num + 1
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embeddings=self.address_num, embedding_dim=embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids -> (batch_size,)
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("Expected addresses to be a torch.Tensor")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.address_num):
            raise ValueError(f"Invalid addresses found.")
        # out -> (batch_size, embed_dim)
        return self.embedding(input_ids)


class NumberEncoder(nn.Module):
    def __init__(self, hid_dim: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = MLP(
            in_dim=10,
            hid_dim=hid_dim,
            out_dim=embed_dim
        )

    def forward(self, bins: torch.Tensor) -> torch.Tensor:
        # bins -> (batch_size,)
        bins = bins.long()
        str_bins = [str(num).zfill(10) for num in bins.tolist()]
        char_tensor = torch.tensor(
            [[int(char) for char in num_str] for num_str in str_bins],
            dtype=torch.float, device=bins.device
        )
        # out -> (batch_size, embed_dim)
        return self.mlp(char_tensor)


class StringEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, hid_dim: int, embed_dim: int):
        super().__init__()
        self.encoder = encoder
        self.hid_dim = hid_dim
        self.embed_dim = embed_dim
        self.mlp = MLP(
            in_dim=encoder.config.hidden_size,
            hid_dim=hid_dim,
            out_dim=embed_dim
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids -> (batch_size, seq_len)
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("Expected input_ids to be a torch.Tensor")

        with torch.no_grad():
            outputs = self.encoder(input_ids)
            seq_output = outputs.last_hidden_state
            embedding = torch.mean(seq_output, dim=1)
        # out -> (batch_size, embed_dim)
        return self.mlp(embedding)


class KvEncoder(nn.Module):
    def __init__(
            self, pre_encoder: nn.Module, address_num: int, hid_dim: int, embed_dim: int,
            seq_num: int, heads_num: int, layers_num: int, out_dim: int
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.str_encoder = StringEncoder(pre_encoder, hid_dim, embed_dim)
        self.num_encoder = NumberEncoder(hid_dim, embed_dim)
        self.addr_encoder = AddressEncoder(address_num, embed_dim)

        self.seq_num = seq_num
        encoder_layers = TransformerEncoderLayer(embed_dim*2, heads_num)
        self.transformer_encoder = TransformerEncoder(encoder_layers, layers_num)
        # self.fc = nn.Linear(embed_dim*2, out_dim)
        self.fc = MLP(embed_dim*2, embed_dim, out_dim)

    def forward(self, kv: torch.Tensor, kv_type: torch.Tensor):
        # kv -> (batch_size, kv_size, seq_size)
        # kv_type -> (batch_size, kv_size)
        unique_types = torch.unique(kv_type)
        unique_types = unique_types[unique_types != 0]

        processed_kvs = []
        for t in unique_types:
            mask = kv_type == t
            selected_kvs = kv[mask]  # (selected_size, seq_size)
            ks, vs = torch.split(selected_kvs, int(selected_kvs.shape[-1]/2), dim=-1)
            processed_ks = self.str_encoder(ks)
            if t.item() != Kv.Type.STRING.value:
                vs = vs[..., 0]

            processed_vs = {
                Kv.Type.STRING.value: self.str_encoder,
                Kv.Type.NUMBER.value: self.num_encoder,
                Kv.Type.ADDRESS.value: self.addr_encoder
            }[t.item()](vs)
            processed_kvs.append(torch.cat((processed_ks, processed_vs), dim=-1))

        kv_embeddings = torch.zeros(*kv.shape[:-1], self.embed_dim*2, device=kv.device)
        for kv_t in unique_types:
            mask = kv_type == kv_t
            kv_embeddings[mask] = processed_kvs.pop(0)

        embeddings = kv_embeddings.permute(1, 0, 2)    # (kv_size, batch_size, embed_dim)
        embeddings = self.transformer_encoder(embeddings)   # (kv_size, batch_size, embed_dim)
        embeddings = embeddings.mean(dim=0)  # (batch_size, embed_dim)
        return self.fc(embeddings)  # (batch_size, out_dim)


class SiameseNet(nn.Module):
    def __init__(
            self, dense_hid_dim: int, dense_dropout: float, act
    ) -> None:
        super().__init__()
        self.act = act

        self.dense = nn.Sequential(
            nn.Dropout(dense_dropout),
            nn.Linear(dense_hid_dim, dense_hid_dim),
            self.act,
            nn.Dropout(dense_dropout),
        )

    def forward(self, query: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # query -> (batch_size, hid_dim)
        # targets -> (batch_size, target_num, hid_dim)
        query_vector = self.dense(query)
        target_vectors = self.dense(targets)

        # Compute cosine similarity
        # (batch_size, target_num)
        similarities = torch.cosine_similarity(query_vector.unsqueeze(1), target_vectors, dim=2)

        return similarities

        # return sorted_indices  # Shape (batch_size, target_num)


class TIR(MetaModel):
    def __init__(
            self, pre_encoder, address_num: int, hid_dim: int, embed_dim: int,
            seq_num: int, heads_num: int, layers_num: int, dense_hid_dim: int,
            dense_dropout: float, rff_hid_dim: int, act: str = 'relu', device=None
    ):
        super().__init__()
        self.kv = KvEncoder(
            pre_encoder, address_num, hid_dim, embed_dim,
            seq_num, heads_num, layers_num, dense_hid_dim
        )
        if act.lower() not in ['relu', 'tanh', 'sigmoid', 'leakyrelu']:
            raise ValueError()

        self.act = {
            'leakyrelu': nn.LeakyReLU(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }[act]

        self.sn = SiameseNet(dense_hid_dim, dense_dropout, self.act)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(dense_hid_dim)

        # # Row-wise feed-forward network
        self.rff = nn.Sequential(
            nn.Linear(dense_hid_dim, dense_hid_dim),
            nn.Sigmoid()
        )
        self.device = device

    def forward(
            self, query: torch.Tensor, query_type: torch.Tensor,
            targets: torch.Tensor, targets_type: torch.Tensor
    ):
        # query -> (batch_size, kv_size, seq_size)
        # query_type -> (batch_size, kv_size)
        # targets -> (batch_size, target_num, kv_size, seq_size)
        # targets_type -> (batch_size, target_num, kv_size)

        batch_size, target_num, kv_size = targets.shape[0], targets.shape[1], targets.shape[2]
        query_kv = self.kv(query, query_type)   # (batch_size, embed_dim)
        targets_kv = self.kv(
            targets.view(batch_size * target_num, kv_size, -1),
            targets_type.view(batch_size * target_num, kv_size)
        ).view(batch_size, target_num, -1)    # (batch_size, target_num, embed_dim)

        similarities = self.sn(query_kv, targets_kv)    # (batch_size, target_num)

        # Apply Layer Normalization to the similarities
        normed_similarities = self.layer_norm(similarities)     # (batch_size, target_num)
        # normed_similarities = self.rff(normed_similarities)

        # # Generate relevance scores using the rFF network
        # relevance_scores = self.rff(normed_similarities)  # (batch_size, target_num, 1)
        # print()
        # print("relevance_scores:", relevance_scores.shape)
        # relevance_scores = relevance_scores.squeeze(-1)  # Shape (batch_size, target_num)

        # # Get indices of sorted relevance scores (descending)
        # sorted_indices = torch.argsort(relevance_scores, dim=1, descending=True)

        batch_size, target_num = targets.shape[0], targets.shape[1]
        valid_flag = torch.full((batch_size,), -1, dtype=torch.int64)  # Default
        for b in range(batch_size):
            for i in range(target_num - 1, -1, -1):
                if not torch.all(targets[b, i, :, :] == 0):
                    valid_flag[b] = i
                    break
        for b in range(batch_size):
            if valid_flag[b] != -1:
                normed_similarities[b, valid_flag[b] + 1:] = 0
        return normed_similarities  # Shape (batch_size, target_num)
        # return torch.argmax(normed_similarities, dim=1)  # Shape (batch_size,)
