"""Meaningful model-contract checks; tensors made here are test fixtures only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.tca_config import DATA_DIR, TCAConfig, TrainConfig
from neural_networks.tca import TCAModel
from train.losses import multitask_loss
from train.data import load_data, fit_normalization, tensors, model_inputs
from train.pretrain_tca import mask_patches


class TCATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(42)
        cls.config = TCAConfig()
        cls.raw = load_data(DATA_DIR, cls.config)
        cls.batch = tensors(cls.raw, fit_normalization(cls.raw), 'cpu')
        cls.model = TCAModel(cls.config).eval()

    def test_real_data_contract(self):
        self.assertEqual(len(self.raw['sample_ids']), 4)
        self.assertEqual(int(self.raw['occurrence'].sum()), 2)
        self.assertTrue(self.raw['trigger_valid'].any(1).all())
        self.assertIn('real', json.loads(str(self.raw['dynamic_metadata']))['origin'])
        self.assertEqual(self.config.embed_dim, 256)
        self.assertEqual(self.config.temporal_layers, 6)

    def test_missing_frame_invariance_and_attention(self):
        batch = {k: v[:1].clone() for k,v in model_inputs(self.batch).items()}
        batch['cause_valid'][:, 0] = False
        batch['trigger_valid'][:, 0] = False
        with torch.inference_mode():
            a = self.model(**batch, explain=True)
            batch['trigger'][:, :, 0] = 99999
            batch['cause'][:, :, 0] = -99999
            b = self.model(**batch, explain=True)
        torch.testing.assert_close(a['logits'], b['logits'], rtol=0, atol=0)
        self.assertEqual(tuple(a['attention'].shape), (1,8,48,18))
        self.assertEqual(float(a['attention'][:, :, :, 0].abs().sum()), 0)
        self.assertEqual(float(a['attention'][:, :, 0].abs().sum()), 0)
        active = batch['cause_valid'][0]
        torch.testing.assert_close(a['attention'][0, :, active].sum(-1), torch.ones(8,int(active.sum())))

    def test_all_missing_is_finite_and_not_fabricated(self):
        batch = {k: v[:1].clone() for k,v in model_inputs(self.batch).items()}
        batch['trigger_valid'].zero_()
        with torch.inference_mode():
            output = self.model(**batch, explain=True)
        self.assertTrue(all(torch.isfinite(v).all() for v in output.values()))
        self.assertEqual(float(output['attention'].abs().sum()), 0)
        self.assertEqual(float(output['key'].abs().sum()), 0)
        batch['cause_valid'].zero_()
        with torch.inference_mode():
            output = self.model(**batch, explain=True)
        self.assertTrue(all(torch.isfinite(v).all() for v in output.values()))
        self.assertEqual(float(output['query'].abs().sum()), 0)

    def test_environment_changes_query_before_attention(self):
        batch = {k: v[:1].clone() for k,v in model_inputs(self.batch).items()}
        with torch.inference_mode():
            a = self.model(**batch, explain=True)
            batch['environment'] += 1
            b = self.model(**batch, explain=True)
        self.assertGreater(float((a['query']-b['query']).abs().max()), 1e-5)
        torch.testing.assert_close(a['key'], b['key'])

    def test_regression_is_positive_only_mse(self):
        prediction = torch.tensor([1., 100.], requires_grad=True)
        labels, targets = torch.tensor([1.,0.]), torch.tensor([3., 999.])
        total, _, reg = multitask_loss(torch.zeros(2), prediction, labels, targets, TrainConfig())
        torch.testing.assert_close(reg, (torch.tensor(1.)-torch.log1p(torch.tensor(3.))).square())
        total.backward()
        self.assertEqual(float(prediction.grad[1]), 0.)
        empty, _, reg_empty = multitask_loss(torch.zeros(2), prediction, torch.zeros(2), targets, TrainConfig())
        self.assertEqual(float(reg_empty), 0.)
        self.assertTrue(torch.isfinite(empty))

    def test_mae_mask_only_observed_patches(self):
        x = torch.ones(1, 3, 18, 16, 16)
        valid = torch.ones(1,18,dtype=torch.bool); valid[:,0] = False
        corrupted, mask = mask_patches(x,valid,4,0.3)
        self.assertFalse(mask[:, :, 0].any())
        self.assertEqual(int(mask[0,0].sum()), round(17*16*0.3)*16)
        self.assertEqual(float(corrupted[mask].sum()), 0)

    def test_checkpoint_round_trip(self):
        batch = {k: v[:1] for k,v in model_inputs(self.batch).items()}
        with tempfile.TemporaryDirectory() as folder, torch.inference_mode():
            path = Path(folder)/'test.pt'
            torch.save(self.model.state_dict(),path)
            model = TCAModel(self.config).eval()
            model.load_state_dict(torch.load(path,weights_only=True),strict=True)
            torch.testing.assert_close(self.model(**batch)['logits'], model(**batch)['logits'],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
