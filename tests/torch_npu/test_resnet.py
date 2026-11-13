# Copyright (c) 2020-2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from Ascend/pytorch at c798c6532a0d05c1863489e83cb7e3985be4e830.
# Source: test/npu/test_resnet.py (network and training settings preserved).
"""ResNet18 training and evaluation from the upstream torch_npu smoke test.

Keep the original synthetic inputs, eleven training batches, one validation
batch, CrossEntropyLoss and SGD settings. Standalone training CLI, checkpoint
helpers and progress-meter formatting are not needed by this pytest case.
"""
import torch
from torch import nn
from torchvision import models
from torch_npu.testing.testcase import TestCase


BATCH_SIZE = 128
TRAIN_BATCHES = 11
DEVICE = "npu:0"


def make_batch():
    """Generate the same NPU-generated inputs and CPU labels as upstream."""
    shape = (BATCH_SIZE, 3, 224, 224)
    images = (torch.randn(shape).npu() + torch.randint(-2, 2, shape).npu()).cpu()
    target = torch.randint(1, 1000, (BATCH_SIZE,)).to(torch.int32)
    return images.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)


def accuracy(output, target):
    """Compute the upstream top-1 and top-5 classification accuracy."""
    with torch.no_grad():
        prediction = output.topk(5, 1, True, True)[1].t()
        correct = prediction.eq(target.view(1, -1).expand_as(prediction))
        return tuple(correct[:k].reshape(-1).float().sum().mul_(100.0 / target.size(0)).item()
                     for k in (1, 5))


class TestResnet(TestCase):
    """Exercise ResNet18 forward, backward, SGD updates and evaluation."""

    def test_resnet(self):
        """Run the upstream default single-device training workload."""
        torch.npu.set_device(DEVICE)
        model = models.resnet18().to(DEVICE)
        criterion = nn.CrossEntropyLoss().to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

        model.train()
        for step in range(TRAIN_BATCHES):
            images, target = make_batch()
            output = model(images)
            loss = criterion(output, target)
            top1, top5 = accuracy(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.assertTrue(torch.isfinite(loss).item(), f"Non-finite loss at training step {step}")
            print(f"Train [{step}/{TRAIN_BATCHES}] loss={loss.item():.4e} top1={top1:.3f} top5={top5:.3f}")

        model.eval()
        with torch.no_grad():
            images, target = make_batch()
            output = model(images)
            loss = criterion(output, target)
            top1, top5 = accuracy(output, target)
            self.assertTrue(torch.isfinite(loss).item(), "Non-finite validation loss")
            print(f"Validation loss={loss.item():.4e} top1={top1:.3f} top5={top5:.3f}")
