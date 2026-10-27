"""Extract two real positives and two real negatives from 2025-05, without replacement.

This is a real-data I/O demonstration, NOT the paper's 2020-2024 training split.
The supplied 2022 files have no available radar frames. Missing frames stay unavailable.
Source paths are supplied by the caller and are not hardcoded in the release.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.tca_config import DATA_DIR, SCHEMA


def prepare(source, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    chunks, provenance = [], []
    for label, kind in [(1, 'pos'), (0, 'neg')]:
        path = Path(source) / f'fusion_{kind}_202505.npz'
        with np.load(path, allow_pickle=False) as z:
            # Prefer frames with Trigger coverage so the cross-attention is exercised.
            eligible = np.flatnonzero((z['precip_exist'].sum(1) > 0) & (z['radar_exist'].sum(1) >= 9)
                                      & (z['y'] == label))
            if len(eligible) < 2:
                raise ValueError(f'{path.name}: need two samples with >=9 valid Trigger frames')
            indices = eligible[:2]
            chunks.append({k: z[k][indices].copy() for k in
                           ('precip_x', 'precip_exist', 'radar_x', 'radar_exist', 'static', 'y', 'y_scale')})
            provenance.append({'filename': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               'zero_based_rows': indices.tolist(), 'label': label})
    data = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    ids = np.array([f'{p["filename"]}:row={i}' for p in provenance for i in p['zero_based_rows']])
    cleanup = {}
    for key, mask_key in [('precip_x', 'precip_exist'), ('radar_x', 'radar_exist')]:
        x = data[key]
        observed = data[mask_key][:, None, :, None, None]
        # Follow the supplied loader's explicit missing-pixel/negative-rain sanitation.
        cleanup[key] = {'nonfinite_pixels': int((~np.isfinite(x)).sum()),
                        'negative_pixels': int((x < 0).sum())}
        data[key] = np.where(observed, np.maximum(np.nan_to_num(x, nan=0, posinf=0, neginf=0), 0), 0)
    cleanup['static'] = {'nonfinite_pixels': int((~np.isfinite(data['static'])).sum())}
    data['static'] = np.nan_to_num(data['static'], nan=0, posinf=0, neginf=0)
    dynamic_meta = {'schema': SCHEMA, 'origin': 'real supplied training NPZ subset',
        'cause_channels': ['log1p_gridded_precipitation', 'log1p_gauge_precipitation', 'gauge_distance_field'],
        'trigger_channels': ['log1p_radar_rainfall', 'log1p_gauge_rainfall', 'gauge_distance_field'],
        'cause_window': '48 hourly input frames',
        'trigger_window': '18 ten-minute frames',
        'distance_units': 'inherited provider field; exact scaling not documented',
        'timestamps_locations': 'not present in source NPZ; no inferred alignment',
        'provenance': provenance, 'cleanup': cleanup}
    static_meta = {'schema': SCHEMA, 'channels': 16,
        'encoding': 'supplied preprocessed channels 0:15 plus per-warning log1p AEP at channel 15',
        'channel_order': 'retained exactly; complete variable/scale dictionary absent in source NPZ',
        'aep': 'time-varying per-warning conditioning snapshot, not a persistent physical attribute'}
    np.savez_compressed(destination / 'dynamic_data.npz', sample_ids=ids,
        cause=data['precip_x'].astype('float32'), cause_valid=data['precip_exist'].astype(bool),
        trigger=data['radar_x'].astype('float32'), trigger_valid=data['radar_exist'].astype(bool),
        dynamic_metadata=json.dumps(dynamic_meta))
    np.savez_compressed(destination / 'static_data.npz', sample_ids=ids,
        environment=data['static'].astype('float32'), static_metadata=json.dumps(static_meta))
    np.savez_compressed(destination / 'target_data.npz', sample_ids=ids,
        occurrence=data['y'].astype('float32'), magnitude=data['y_scale'].astype('float32'))
    (destination/'provenance.json').write_text(json.dumps(dynamic_meta, indent=2)+'\n', encoding='utf-8')
    print(f'Prepared {len(ids)} real samples in {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    args = parser.parse_args()
    prepare(args.source_dir, args.data_dir)
