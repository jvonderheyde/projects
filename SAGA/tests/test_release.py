import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from saga.campaign import launch, parse_args
from saga.config import Config


class ReleaseTests(unittest.TestCase):
    def test_all_public_presets_validate(self):
        for path in sorted(Path('examples').glob('*.json')):
            recipe = json.loads(path.read_text())
            reaction = recipe.pop('reaction')
            cfg = Config.for_reaction(reaction, **recipe)
            self.assertEqual(cfg.relaxation_steps('ads'), 500)
            self.assertEqual(cfg.ads_relax_fmax, .05)
            self.assertTrue(cfg.require_adsorbate_contact)
        toy = json.loads(Path('examples/toy_cpu.json').read_text())
        self.assertEqual(toy['device'], 'cpu')
        self.assertFalse(toy.get('demo', False))

    def test_four_adsorbate_launch_resume_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(['--seed', '8', '--recipe', 'examples/toy_cpu.json',
                               '--smoke-test', '--generations', '1', '--root', tmp])
            result = launch(args)
            self.assertEqual(result['config']['adsorbates'], ('H', 'N', 'N2', 'NH3'))
            with patch('saga.optimizer.ActiveLoop', side_effect=AssertionError('should skip')):
                launch(args)
            args.generations = 2
            launch(args)
            state = json.loads((Path(tmp)/'seed_8/checkpoint.json').read_text())
            self.assertEqual(state['gen'], 2)
            self.assertTrue(state['validation_complete'])

    def test_missing_model_cannot_become_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(['--seed', '8', '--recipe', 'examples/toy_cpu.json', '--root', tmp])
            with self.assertRaisesRegex(ValueError, 'MACE model'):
                launch(args)
