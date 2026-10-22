# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tensor tuple input regression for mixed plugin and native CPU execution."""

import ctypes
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tests.mark_utils import arg_mark
from tests.st.ops.op_plugin.mock_build import configure_and_build_mock_plugin


def run_generator_fallback(step, offset):
    """Check native Generator with a strided state after proving plugin dispatch is active."""
    import mindspore as ms  # pylint: disable=import-outside-toplevel
    from mindspore import mint  # pylint: disable=import-outside-toplevel
    from mindspore.common.generator import Generator  # pylint: disable=import-outside-toplevel
    from mindspore.ops.auto_generate.gen_ops_prim import generator_op  # pylint: disable=import-outside-toplevel

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    library = ctypes.CDLL(os.environ["MS_OP_PLUGIN_PATH"])
    library.GetRegisteredOpCount.restype = ctypes.c_int
    library.GetAllRegisteredOps.restype = ctypes.POINTER(ctypes.c_char_p)
    names = library.GetAllRegisteredOps()
    registered = {names[index].decode() for index in range(library.GetRegisteredOpCount())}
    assert "Generator" not in registered
    # This repository's mock LogicalAnd computes OR, unlike the native implementation.
    probe = mint.logical_and(ms.Tensor([True]), ms.Tensor([False]))
    np.testing.assert_array_equal(probe.asnumpy(), [True])

    source = Generator().manual_seed(1234567)
    expected = source.get_state().asnumpy()
    backing = np.full(expected.size * step + offset, 0x55, dtype=expected.dtype)
    backing[offset:offset + expected.size * step:step] = expected
    state = ms.Tensor(backing)[offset:offset + expected.size * step:step]
    assert not state.is_contiguous(), "The regression requires a non-contiguous Tensor tuple element"

    seed = ms.Tensor([9], ms.int64)
    generator_offset = ms.Tensor([0], ms.int64)
    # Use the generated primitive: common.Generator uses a legacy wrapper on some branches.
    output_seed, output_offset, output_state = generator_op(3, (seed, generator_offset, state))
    np.testing.assert_array_equal(output_seed.asnumpy(), [1234567])
    np.testing.assert_array_equal(output_offset.asnumpy(), [0])
    np.testing.assert_array_equal(output_state.asnumpy(), expected)
    np.testing.assert_array_equal(seed.asnumpy(), [1234567])
    np.testing.assert_array_equal(state.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize("step, offset", [(2, 0), (3, 1)])
def test_native_cpu_fallback_noncontiguous_tensor_tuple(step, offset):
    """
    Feature: Native CPU fallback preparation of Tensor tuple inputs with a plugin loaded.
    Description: Restore Generator state from a strided tuple element using the repository mock plugin.
    Expectation: Native Generator restores exact state bytes and seed without altering the source View.
    """
    library = configure_and_build_mock_plugin()
    assert library, "Failed to build the repository mock plugin"
    environment = os.environ.copy()
    environment["MS_OP_PLUGIN_PATH"] = library
    root = str(Path(__file__).resolve().parents[4])
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [root, environment.get("PYTHONPATH")]))
    # A fresh process prevents static plugin-registry caches from another ST masking the fallback.
    command = (f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r})"
               f"['run_generator_fallback']({step}, {offset})")
    subprocess.run([sys.executable, "-c", command], env=environment, check=True, timeout=180)
