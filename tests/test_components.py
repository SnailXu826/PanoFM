"""Regression tests for the paper objectives and context isolation."""

import unittest
from pathlib import Path

import torch

from panofm.config import Config, PrototypeConfig, load_config
from panofm.context import context_prompts
from panofm.data import Tooth
from panofm.models.anatomy import sinkhorn
from panofm.models.prototype import PrototypeBank
from panofm.models.vision import contrastive_loss
from panofm.train import focal_loss


class ComponentsTest(unittest.TestCase):
    def test_paper_configuration(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "paper.toml"
        paper = load_config(path)
        self.assertEqual(paper.ot, Config().ot)
        self.assertEqual(paper.vision, Config().vision)
        self.assertEqual(tuple(paper.training.betas), Config().training.betas)

    def test_sinkhorn_marginals_and_gradients(self):
        costs = torch.tensor([[0.1, 0.3, 0.2], [0.5, 0.2, 0.4]], requires_grad=True)
        plan = sinkhorn(costs, 0.1, 50)
        torch.testing.assert_close(plan.sum(dim=0), torch.full((3,), 1 / 3))
        torch.testing.assert_close(plan.sum(dim=1), torch.full((2,), 1 / 2))
        (plan * costs).sum().backward()
        self.assertTrue(torch.isfinite(costs.grad).all())

    def test_prototypes_update_and_rejection(self):
        bank = PrototypeBank(2, 1, PrototypeConfig())
        self.assertTrue(bank.update(torch.tensor([[1.0], [2.0]]), torch.tensor([0, 1])))
        torch.testing.assert_close(bank.centroids[:, 0], torch.tensor([0.01, 0.02]))
        self.assertAlmostEqual(float(bank.rate), 0.0101)
        previous = bank.centroids.clone()
        self.assertFalse(bank.update(torch.tensor([[float("nan")]]), torch.tensor([0])))
        torch.testing.assert_close(bank.centroids, previous)
        self.assertAlmostEqual(float(bank.rate), 0.00505)

    def test_contralateral_positives_stay_within_a_radiograph(self):
        features = torch.eye(3).repeat(2, 1)
        same = torch.tensor([[0, 11, 0], [0, 21, 0], [1, 11, 0],
                             [0, 11, 1], [0, 21, 1], [1, 11, 1]])
        separate = same.clone()
        separate[[1, 4], 0] = 2
        self.assertLess(contrastive_loss(features, same, 1).item(), contrastive_loss(features, separate, 1).item())

    def test_phase_one_excludes_other_diseased_negatives(self):
        targets = torch.tensor([[1.0, 0], [0, 1], [0, 0]])
        logits = torch.zeros_like(targets, requires_grad=True)
        focal_loss(logits, targets, 1, 0.25, 2).backward()
        self.assertEqual(float(logits.grad[0, 1]), 0)
        self.assertNotEqual(float(logits.grad[2, 1]), 0)
        logits.grad.zero_()
        focal_loss(logits, targets, 2, 0.25, 2).backward()
        self.assertGreater(float(logits.grad[0, 1]), 0)

    def test_prompt_masks_target_and_drops_other_images(self):
        records = [Tooth(image, tooth, Path(), None, "site-a") for image, tooth in
                   (("private-a", 11), ("private-a", 12), ("private-b", 21))]
        scores = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
        thresholds = dict.fromkeys(("PP", "DC", "DF", "IT", "TAC", "TAR"), 0.5)
        prompt = context_prompts(records, scores, thresholds)[0]
        self.assertIn("condition is unknown", prompt)
        self.assertIn("Tooth 12: dental caries", prompt)
        for excluded in ("periapical", "Tooth 21", "private-a", "site-a"):
            self.assertNotIn(excluded, prompt)
