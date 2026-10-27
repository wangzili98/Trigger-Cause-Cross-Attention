"""Read a TCA checkpoint and write sample-level probabilities and attention."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.tca_config import DATA_DIR, SCHEMA, TCAConfig
from neural_networks.tca import TCAModel
from train.data import load_data, tensors, model_inputs, fingerprint
from train.runtime import initialize


def run(data_dir, checkpoint=None, device_name='auto', threads=4):
    data_dir = Path(data_dir)
    checkpoint = Path(checkpoint) if checkpoint else data_dir/'checkpoints'/'TCA.pt'
    device = initialize(device_name, 42, threads)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state['schema'] != SCHEMA:
        raise ValueError('Unsupported checkpoint schema')
    cfg = TCAConfig(**state['config'])
    # Labels are optional at inference. They never enter the model or normalization.
    raw = load_data(data_dir, cfg, require_targets=False)
    batch = tensors(raw, state['normalization'], device)
    model = TCAModel(cfg).to(device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    with torch.inference_mode():
        result = model(**model_inputs(batch), explain=True)
    values = {k: v.cpu().numpy() for k,v in result.items()}
    values['probability'] = torch.sigmoid(result['logits']).cpu().numpy()
    # Auxiliary head is unconstrained in log space; clip only for reported nonnegative scale.
    values['magnitude'] = np.expm1(np.clip(values['log_magnitude'], 0, 20))
    values['sample_ids'] = raw['sample_ids']
    values['cause_valid'] = raw['cause_valid']
    values['trigger_valid'] = raw['trigger_valid']
    for key in ('logits', 'log_magnitude', 'probability', 'magnitude', 'attention', 'query', 'key', 'value'):
        if not np.isfinite(values[key]).all():
            raise FloatingPointError(f'Nonfinite output {key}')
    out = data_dir/'outputs'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'TCA_outputs.npz', **values)
    with (out/'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'probability', 'log_magnitude', 'magnitude'])
        writer.writerows(zip(values['sample_ids'], values['probability'],
                              values['log_magnitude'], values['magnitude']))
    overlap = sorted(set(raw['sample_ids'].tolist()) & set(state['training_sample_ids']))
    metadata = {'schema': SCHEMA, 'version': state['version'], 'config': state['config'],
        'training_config': state['training_config'], 'normalization': state['normalization'],
        'input_sha256': fingerprint(data_dir), 'checkpoint': checkpoint.name,
        'training_sample_overlap': overlap,
        'interpretation': 'training-example outputs, not independent validation or calibrated operational warnings',
        'shapes': {k: list(v.shape) for k,v in values.items()},
        'axes': {'attention': ['sample', 'head', 'cause_time', 'trigger_time'],
                 'query': ['sample', 'cause_time', 'embedding'],
                 'key': ['sample', 'trigger_time', 'embedding'],
                 'value': ['sample', 'trigger_time', 'embedding']},
        'representations': 'query/key/value are actual projected multi-head tensors before head reshaping; '
                           'cause_tokens and trigger_tokens are pre-projection embeddings',
        'magnitude_units': 'same provider scale as target_data.npz; physical unit undocumented',
        'magnitude_transform': 'log1p target; reported magnitude=expm1(clip(predicted_log,0,20))',
        'availability': 'masked keys and queries have zero exported attention; all-missing trigger contributes zero'}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(f'Saved {out/"TCA_outputs.npz"}; probability={values["probability"].tolist()}', flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    run(args.data_dir, args.checkpoint, args.device, args.threads)
