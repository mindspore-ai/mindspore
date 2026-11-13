# Copyright (c) 2020-2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from Ascend/pytorch at c798c6532a0d05c1863489e83cb7e3985be4e830.
# Source: test/test_dvm.py::TestDvmSmallNetworks.test_training_loop_3steps.
"""Compare the upstream three-step training network with DVM enabled and disabled."""
import os
import re
import subprocess
import sys
import tempfile
import unittest

import torch
import torch_npu


def _run_in_subprocess_raw(code, dvm_enabled, save_path=None):
    """Run code in a subprocess, optionally with DVM enabled."""
    env = os.environ.copy()
    # Remove any existing DVM flags to ensure clean state
    env.pop("TORCH_NPU_LAZY_FUSION", None)
    if dvm_enabled:
        env["TORCH_NPU_LAZY_FUSION"] = "True"
    full_code = f"""
import numpy as np
import torch, torch_npu
torch.manual_seed(42)
np.random.seed(1)
{code}
{"torch.save(result, " + repr(save_path) + ")" if save_path is not None else ""}
"""
    return subprocess.run(
        [sys.executable, "-c", full_code],
        env=env, capture_output=True, text=True, timeout=120, check=False
    )


def _run_in_subprocess(code, dvm_enabled, save_path):
    """Run code in a subprocess, optionally with DVM enabled, and persist result."""
    r = _run_in_subprocess_raw(code, dvm_enabled, save_path)
    if r.returncode != 0:
        raise RuntimeError(
            f"Subprocess (dvm={'on' if dvm_enabled else 'off'}) failed:\n{r.stderr[-500:]}"
        )


def compare_dvm_on_off(code):
    """Run code with DVM on and off, return (result_off, result_on)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path_off = os.path.join(tmpdir, "off.pt")
        path_on = os.path.join(tmpdir, "on.pt")
        _run_in_subprocess(code, dvm_enabled=False, save_path=path_off)
        _run_in_subprocess(code, dvm_enabled=True, save_path=path_on)
        r_off = torch.load(path_off, weights_only=False)
        r_on = torch.load(path_on, weights_only=False)
        return r_off, r_on


class TestDvmSmallNetworks(unittest.TestCase):
    """Keep the upstream DVM network and numerical tolerances."""

    def setUp(self):
        """Reject devices without DVM support through the runner's skip handling."""
        device_name = torch_npu.npu.get_device_name(0)
        if not re.match(r"^(Ascend910B[0-9]|Ascend910_9[0-9]{3}|Ascend950)", device_name):
            self.skipTest(f"DVM not supported on {device_name}")

    def test_training_loop_3steps(self):
        """3-step training loop with SGD-like update."""
        r_off, r_on = compare_dvm_on_off("""
H = 32
w = torch.randn(H, H)
x = torch.randn(8, H)
target = torch.randn(8, H)
lr = 0.01

wn = w.clone().npu().requires_grad_(True)
xn = x.npu()
tn = target.npu()

losses = []
for step in range(3):
    out = torch.relu(xn @ wn)
    loss = ((out - tn) ** 2).mean()
    loss.backward()
    losses.append(loss.detach().cpu().item())
    with torch.no_grad():
        wn -= lr * wn.grad
        wn.grad.zero_()

result = {
    "losses": losses,
    "final_w": wn.detach().cpu(),
}
""")
        for i, (a, b) in enumerate(zip(r_off["losses"], r_on["losses"])):
            self.assertAlmostEqual(a, b, places=4, msg=f"loss step {i}")
        diff = (r_off["final_w"].float() - r_on["final_w"].float()).abs().max().item()
        self.assertLessEqual(diff, 1e-5, f"final_w max_diff={diff:.2e}")
