"""Manuscript-aligned TCA architecture with explicit missing-frame handling."""
import math
import torch
from torch import nn


def masked_pool(x, valid):
    mask = valid.unsqueeze(-1)
    avg = x.masked_fill(~mask, 0).sum(1) / mask.sum(1).clamp_min(1)
    maximum = x.masked_fill(~mask, -torch.inf).amax(1)
    maximum = torch.where(valid.any(1, keepdim=True), maximum, 0)
    return torch.cat((avg, maximum), dim=-1)


class TemporalBranch(nn.Module):
    def __init__(self, channels, frames, depth, config):
        super().__init__()
        layers = []
        width = 64
        for _ in range(depth):
            layers += [nn.Conv2d(channels, width, 3, stride=2, padding=1),
                       nn.BatchNorm2d(width), nn.GELU(), nn.Dropout2d(config.branch_dropout)]
            channels = width
            width = min(width * 2, 512)
        self.spatial = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                     nn.Linear(channels, config.embed_dim), nn.LayerNorm(config.embed_dim))
        self.time_embedding = nn.Embedding(frames, config.embed_dim)
        self.type_embedding = nn.Parameter(torch.zeros(config.embed_dim))
        nn.init.normal_(self.type_embedding, std=0.02)
        self.availability_embedding = nn.Embedding(2, config.embed_dim)
        layer = nn.TransformerEncoderLayer(config.embed_dim, config.heads,
            dim_feedforward=config.embed_dim * 4, dropout=config.branch_dropout,
            batch_first=True, activation='gelu', norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, config.temporal_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x, valid):
        b, c, t, h, w = x.shape
        # Exclude missing frames BEFORE BatchNorm so their values cannot contaminate valid tokens.
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        present = valid.reshape(-1)
        tokens = x.new_zeros(b * t, self.time_embedding.embedding_dim)
        if present.any():
            tokens[present] = self.spatial(frames[present])
        tokens = tokens.reshape(b, t, -1)
        tokens = tokens + self.time_embedding(torch.arange(t, device=x.device))[None]
        tokens = tokens + self.type_embedding + self.availability_embedding(valid.long())
        # A zero sentinel avoids all-masked Transformer softmax; output is subsequently zeroed.
        padding = ~valid.clone()
        empty = ~valid.any(1)
        padding[empty, 0] = False
        tokens = self.norm(self.temporal(tokens, src_key_padding_mask=padding))
        return tokens.masked_fill(~valid.unsqueeze(-1), 0)


class TCAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.embed_dim
        self.cause_encoder = TemporalBranch(config.cause_channels, config.cause_frames,
                                           config.cause_cnn_depth, config)
        self.trigger_encoder = TemporalBranch(config.trigger_channels, config.trigger_frames,
                                             config.trigger_cnn_depth, config)
        self.environment_encoder = nn.Sequential(
            nn.Conv2d(config.environment_channels, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.GELU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        # Persistent environment + per-warning AEP are integrated BEFORE attention.
        self.cause_fusion = nn.Sequential(nn.Linear(d + 64, d), nn.GELU(),
                                          nn.Dropout(config.cross_dropout), nn.LayerNorm(d))
        self.cross_attention = nn.MultiheadAttention(d, config.heads,
                                                     dropout=config.cross_dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(config.cross_dropout),
                                 nn.Linear(4*d, d), nn.Dropout(config.cross_dropout))
        self.output_norm = nn.LayerNorm(d)
        def head():
            return nn.Sequential(nn.Linear(2*d, config.head_hidden), nn.GELU(),
                                 nn.Dropout(config.head_dropout), nn.Linear(config.head_hidden, 1))
        self.occurrence_head = head()
        self.magnitude_head = head()
        nn.init.constant_(self.occurrence_head[-1].bias,
                          math.log(config.positive_prior / (1-config.positive_prior)))

    def forward(self, cause, cause_valid, trigger, trigger_valid, environment, explain=False):
        dynamic = self.cause_encoder(cause, cause_valid)
        env = self.environment_encoder(environment)
        query = self.cause_fusion(torch.cat((dynamic, env[:, None].expand(-1, dynamic.shape[1], -1)), -1))
        query = query.masked_fill(~cause_valid.unsqueeze(-1), 0)
        trigger_tokens = self.trigger_encoder(trigger, trigger_valid)
        padding = ~trigger_valid.clone()
        empty = ~trigger_valid.any(1)
        padding[empty, 0] = False
        attended, weights = self.cross_attention(query, trigger_tokens, trigger_tokens,
            key_padding_mask=padding, need_weights=explain, average_attn_weights=False)
        # All-unavailable Trigger has zero contribution, not a fabricated observed frame.
        attended = attended.masked_fill(empty[:, None, None], 0)
        fused = self.cross_norm(query + attended)
        fused = self.output_norm(fused + self.ffn(fused))
        pooled = masked_pool(fused, cause_valid)
        result = {'logits': self.occurrence_head(pooled).squeeze(-1),
                  'log_magnitude': self.magnitude_head(pooled).squeeze(-1)}
        if explain:
            weights = weights.masked_fill(~trigger_valid[:, None, None, :], 0)
            weights = weights.masked_fill(~cause_valid[:, None, :, None], 0)
            # Export actual projected Q/K/V, not merely pre-projection branch embeddings.
            d = self.config.embed_dim
            wq, wk, wv = self.cross_attention.in_proj_weight.split(d)
            bq, bk, bv = self.cross_attention.in_proj_bias.split(d)
            result.update(cause_tokens=query, trigger_tokens=trigger_tokens,
                query=torch.nn.functional.linear(query, wq, bq).masked_fill(~cause_valid[..., None], 0),
                key=torch.nn.functional.linear(trigger_tokens, wk, bk).masked_fill(~trigger_valid[..., None], 0),
                value=torch.nn.functional.linear(trigger_tokens, wv, bv).masked_fill(~trigger_valid[..., None], 0),
                attention=weights)
        return result
