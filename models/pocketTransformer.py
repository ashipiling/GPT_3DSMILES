import torch.nn as nn

from .common import (
    NonLinearHead,
    GaussianLayer,
    TransformerEncoderWithPair
)


mapping = {'PAD': 0, 'A': 1, 'R': 2, 'N': 3, 'D': 4,
           'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11,
           'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18,
           'Y': 19, 'V': 20}


def init_bert_params(module):
    if not getattr(module, 'can_global_init', True):
        return
    def normal_(data):
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )
    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

class PocketTransformer3D(nn.Module):
    def __init__(
        self,
        encoder_layers: int,
        encoder_embed_dim: int,
        encoder_ffn_embed_dim: int ,
        encoder_attention_heads: int,
        dropout: float,
        emb_dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        activation_fn: str,
        post_ln: bool,
    ):
        super().__init__()
        self.padding_idx = mapping['PAD']
        self.embed_tokens = nn.Embedding(len(mapping)+1, encoder_embed_dim)
        self.coords_linear = nn.Linear(3, encoder_embed_dim)
        # self.distance_embeding = nn.Linear(1, encoder_embed_dim, bias=False)
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=encoder_layers,
            embed_dim=encoder_embed_dim,
            ffn_embed_dim=encoder_ffn_embed_dim,
            attention_heads=encoder_attention_heads,
            emb_dropout=emb_dropout,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            post_ln=post_ln,
        )

        K = 128
        n_edge_type = len(mapping) * len(mapping)
        self.gbf_proj = NonLinearHead(K, encoder_attention_heads, activation_fn)
        self.gbf = GaussianLayer(K, n_edge_type)
        self.apply(init_bert_params)
        # self.gbf = GaussianLayer(K, n_edge_type)

    def forward(
            self,
            src_tokens,
            src_distance,
            src_edge_type,
            src_esm=None,
            src_coords=None,
            encoder_masked_tokens=None,
            features_only=False,
            **kwargs
    ):

        padding_mask = src_tokens.eq(self.padding_idx)
        # if not padding_mask.any():
        #     padding_mask = None
        # print('src', src_tokens.shape, padding_mask.shape)

        # x = self.embed_tokens(src_tokens)
        x = self.coords_linear(src_coords)
        if src_esm is not None:
            x = x + src_esm

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)

        encoder_rep, encoder_pair_rep, delta_encoder_pair_rep = self.encoder(x,padding_mask=padding_mask,attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float('-inf')] = 0
        if src_esm is not None:
            encoder_rep = encoder_rep + src_esm
        return encoder_rep, padding_mask


