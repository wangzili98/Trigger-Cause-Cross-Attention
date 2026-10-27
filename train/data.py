"""One explicit, pickle-free NPZ contract; no implicit channel or unit conversion."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from config.tca_config import SCHEMA


INPUT_FILES = ('dynamic_data.npz', 'static_data.npz')


def fingerprint(folder, include_targets=False):
    names = INPUT_FILES + (('target_data.npz',) if include_targets else ())
    return {name: hashlib.sha256((Path(folder)/name).read_bytes()).hexdigest() for name in names}


def load_data(folder, config, require_targets=True):
    folder = Path(folder)
    result = {}
    for filename in INPUT_FILES + (('target_data.npz',) if require_targets else ()):
        with np.load(folder / filename, allow_pickle=False) as z:
            for key in z.files:
                if key in result and not np.array_equal(result[key], z[key]):
                    raise ValueError(f'Inconsistent {key} across input files')
                result[key] = z[key].copy()
    n = len(result['sample_ids'])
    if n == 0 or len(set(result['sample_ids'].tolist())) != n:
        raise ValueError('Sample IDs must be unique and nonempty')
    shapes = {'cause': (n, config.cause_channels, config.cause_frames, config.cause_size, config.cause_size),
              'trigger': (n, config.trigger_channels, config.trigger_frames, config.trigger_size, config.trigger_size),
              'cause_valid': (n, config.cause_frames), 'trigger_valid': (n, config.trigger_frames),
              'environment': (n, config.environment_channels, config.cause_size, config.cause_size)}
    if require_targets:
        shapes.update(occurrence=(n,), magnitude=(n,))
    for key, expected in shapes.items():
        if key not in result or result[key].shape != expected:
            raise ValueError(f'{key} must have shape {expected}')
    for key in ('cause_valid', 'trigger_valid'):
        if result[key].dtype != np.bool_:
            raise ValueError(f'{key} must be boolean')
    for key in ('cause', 'trigger'):
        mask = result[key + '_valid'][:, None, :, None, None]
        if not np.isfinite(result[key][np.broadcast_to(mask, result[key].shape)]).all():
            raise ValueError(f'{key} contains nonfinite observed values')
        result[key] = np.where(mask, result[key], 0).astype(np.float32)
    if not np.isfinite(result['environment']).all():
        raise ValueError('environment contains nonfinite values')
    if require_targets:
        if not np.isin(result['occurrence'], [0, 1]).all():
            raise ValueError('occurrence must be binary')
        if not np.isfinite(result['magnitude']).all() or (result['magnitude'] < 0).any():
            raise ValueError('magnitude must be finite and nonnegative')
    for key in ('dynamic_metadata', 'static_metadata'):
        meta = json.loads(str(result[key]))
        if meta['schema'] != SCHEMA:
            raise ValueError(f'Unsupported schema in {key}')
    return result


def fit_normalization(data):
    stats = {}
    for key in ('cause', 'trigger', 'environment'):
        x = data[key]
        means, scales = [], []
        for channel in range(x.shape[1]):
            values = x[:, channel]
            if key != 'environment':
                values = values[data[key + '_valid']]
            if values.size == 0:
                means.append(0.0); scales.append(1.0)
            else:
                means.append(float(values.mean(dtype=np.float64)))
                scales.append(max(float(values.std(dtype=np.float64)), 1e-6))
        stats[key] = {'mean': means, 'std': scales}
    return stats


def tensors(data, normalization, device):
    result = {}
    for key in ('cause', 'trigger', 'environment'):
        x = data[key].astype(np.float32)
        shape = (1, x.shape[1]) + (1,) * (x.ndim-2)
        mean = np.asarray(normalization[key]['mean'], dtype=np.float32).reshape(shape)
        std = np.asarray(normalization[key]['std'], dtype=np.float32).reshape(shape)
        x = (x-mean) / std
        if key != 'environment':
            x = np.where(data[key + '_valid'][:, None, :, None, None], x, 0)
        result[key] = torch.tensor(x, device=device)
    for key in ('cause_valid', 'trigger_valid'):
        result[key] = torch.tensor(data[key], dtype=torch.bool, device=device)
    for key in ('occurrence', 'magnitude'):
        if key in data:
            result[key] = torch.tensor(data[key], dtype=torch.float32, device=device)
    return result


def model_inputs(batch):
    return {k: batch[k] for k in ('cause', 'cause_valid', 'trigger', 'trigger_valid', 'environment')}
