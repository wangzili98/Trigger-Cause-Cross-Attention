"""One-epoch, one-window TCA training demonstration, with MAE initialization."""
import argparse
import csv
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.tca_config import DATA_DIR, VERSION, SCHEMA, TCAConfig, TrainConfig
from neural_networks.tca import TCAModel
from train.data import load_data, fit_normalization, tensors, model_inputs, fingerprint
from train.losses import multitask_loss
from train.pretrain_tca import pretrain
from train.runtime import initialize


def train(data_dir, device_name='auto', epochs=1, pretrain_steps=1, threads=4, encoder_checkpoint=None):
    if epochs < 1 or pretrain_steps < 1:
        raise ValueError('epochs and pretrain_steps must be positive')
    cfg, training = TCAConfig(), TrainConfig()
    device = initialize(device_name, training.seed, threads)
    raw = load_data(data_dir, cfg)
    norm = fit_normalization(raw)
    model = TCAModel(cfg).to(device)
    if encoder_checkpoint:
        state = torch.load(encoder_checkpoint, map_location=device, weights_only=True)
        if state['schema'] != SCHEMA or state['config'] != cfg.to_dict():
            raise ValueError('Encoder checkpoint architecture/schema mismatch')
        # Normalization belongs to the pretrained encoders; never silently refit it.
        norm = state['normalization']
        model.cause_encoder.load_state_dict(state['cause_encoder'], strict=True)
        model.trigger_encoder.load_state_dict(state['trigger_encoder'], strict=True)
        pretraining = state['pretraining_history']
    batch = tensors(raw, norm, device)
    if not encoder_checkpoint:
        pretraining = pretrain(model, batch, training, pretrain_steps)
    branch_params = list(model.cause_encoder.parameters()) + list(model.trigger_encoder.parameters())
    head_params = list(model.occurrence_head.parameters()) + list(model.magnitude_head.parameters())
    excluded = {id(p) for p in branch_params+head_params}
    groups = [
        {'params': model.cause_encoder.parameters(), 'lr': training.cause_lr, 'weight_decay': training.branch_weight_decay},
        {'params': model.trigger_encoder.parameters(), 'lr': training.trigger_lr, 'weight_decay': training.branch_weight_decay},
        {'params': [p for p in model.parameters() if id(p) not in excluded], 'lr': training.fusion_lr, 'weight_decay': training.other_weight_decay},
        {'params': head_params, 'lr': training.head_lr, 'weight_decay': training.other_weight_decay}]
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=training.minimum_lr)
    history = []
    print(f'TCA device={device}, samples={len(raw["sample_ids"])}, parameters={sum(p.numel() for p in model.parameters()):,}', flush=True)
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(**model_inputs(batch))
        total, cls, reg = multitask_loss(output['logits'], output['log_magnitude'],
                                       batch['occurrence'], batch['magnitude'], training)
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite supervised loss')
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        record = {'epoch': epoch, 'loss': float(total.detach()),
                  'focal_loss': float(cls.detach()), 'positive_log_magnitude_mse': float(reg.detach())}
        history.append(record)
        print(f'Epoch={epoch} loss={record["loss"]:.6f}', flush=True)
    data_dir = Path(data_dir)
    checkpoint = data_dir/'checkpoints'/'TCA.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'schema': SCHEMA, 'version': VERSION, 'model': model.state_dict(),
        'config': cfg.to_dict(), 'training_config': training.to_dict(), 'normalization': norm,
        'input_sha256': fingerprint(data_dir, include_targets=True), 'training_sample_ids': raw['sample_ids'].tolist(),
        'pretraining_history': pretraining, 'training_history': history,
        'purpose': 'minimum real-data training demonstration; no independent evaluation'}, checkpoint)
    with (data_dir/'training_history.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    print(f'Saved {checkpoint}', flush=True)
    return checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--pretrain-steps', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--encoder-checkpoint', type=Path)
    args = parser.parse_args()
    train(args.data_dir, args.device, args.epochs, args.pretrain_steps, args.threads, args.encoder_checkpoint)
