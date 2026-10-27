"""Separate masked reconstruction of the two meteorological encoders."""
import argparse
import sys
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.tca_config import DATA_DIR, SCHEMA, TCAConfig, TrainConfig
from neural_networks.tca import TCAModel
from train.data import load_data, fit_normalization, tensors, fingerprint
from train.runtime import initialize


class ReconstructionHead(nn.Module):
    def __init__(self, channels, size, depth, dim):
        super().__init__()
        self.channels, self.size = channels, size
        self.initial = size // 2**depth
        self.projection = nn.Linear(dim, dim * self.initial**2)
        layers = []
        width = dim
        for _ in range(depth-1):
            next_width = max(width//2, 32)
            layers += [nn.ConvTranspose2d(width, next_width, 4, 2, 1),
                       nn.BatchNorm2d(next_width), nn.GELU()]
            width = next_width
        layers += [nn.ConvTranspose2d(width, channels, 4, 2, 1)]
        self.decoder = nn.Sequential(*layers)

    def forward(self, tokens):
        b, t, d = tokens.shape
        image = self.projection(tokens).reshape(b*t, d, self.initial, self.initial)
        return self.decoder(image).reshape(b, t, self.channels, self.size, self.size).permute(0, 2, 1, 3, 4)


def mask_patches(x, valid, patch, ratio):
    b, _, t, h, w = x.shape
    # Sample an exact rounded 30% of available spatio-temporal patches per sample.
    mask = torch.zeros((b, 1, t, h//patch, w//patch), dtype=torch.bool, device=x.device)
    for i in range(b):
        eligible = valid[i, :, None, None].expand(t, h//patch, w//patch).reshape(-1).nonzero().flatten()
        count = min(len(eligible), max(1, round(len(eligible)*ratio)))
        chosen = eligible[torch.randperm(len(eligible), device=x.device)[:count]]
        mask[i, 0].view(-1)[chosen] = True
    mask = mask.repeat_interleave(patch, -2).repeat_interleave(patch, -1).expand_as(x)
    return x.masked_fill(mask, 0), mask


def pretrain(model, batch, config, steps):
    history = []
    model.train()
    for name, size, depth, patch in [
        ('cause', model.config.cause_size, model.config.cause_cnn_depth, config.cause_patch),
        ('trigger', model.config.trigger_size, model.config.trigger_cnn_depth, config.trigger_patch)]:
        encoder = getattr(model, name + '_encoder')
        decoder = ReconstructionHead(batch[name].shape[1], size, depth, model.config.embed_dim).to(batch[name].device)
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()),
            lr=config.pretrain_lr, weight_decay=config.pretrain_weight_decay)
        for step in range(steps):
            corrupted, mask = mask_patches(batch[name], batch[name+'_valid'], patch, config.mask_ratio)
            if not mask.any():
                history.append({'branch': name, 'step': step+1, 'status': 'skipped: no observed frames'})
                continue
            reconstruction = decoder(encoder(corrupted, batch[name+'_valid']))
            loss = (reconstruction[mask]-batch[name][mask]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite pretraining loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.gradient_clip)
            optimizer.step()
            history.append({'branch': name, 'step': step+1, 'loss': float(loss.detach())})
            print(f'MAE {name} step={step+1} loss={float(loss.detach()):.6f}', flush=True)
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1: parser.error('--steps must be positive')
    cfg, training = TCAConfig(), TrainConfig()
    device = initialize(args.device, training.seed, args.threads)
    raw = load_data(args.data_dir, cfg, require_targets=False)
    norm = fit_normalization(raw)
    batch = tensors(raw, norm, device)
    model = TCAModel(cfg).to(device)
    history = pretrain(model, batch, training, args.steps)
    path = args.data_dir/'checkpoints'/'TCA_encoders.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'schema': SCHEMA, 'config': cfg.to_dict(), 'normalization': norm,
                'input_sha256': fingerprint(args.data_dir), 'pretraining_history': history,
                'cause_encoder': model.cause_encoder.state_dict(),
                'trigger_encoder': model.trigger_encoder.state_dict()}, path)
    print(path)
