# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Test signal handling."""

import ctypes
import re
import traceback

import numpy as np

from mindspore.dataset.dataloader import Dataset, DataLoader
from tests.mark_utils import arg_mark


class SignalHandlingDataset(Dataset):
    """
    Dataset for testing signal handling.
    """

    def __init__(self, process_fn):
        self.process_fn = process_fn
        self.num_samples = 10
        self.data = list(range(self.num_samples))

    def __getitem__(self, index):
        data = np.array(self.data[index], dtype=np.uint8)
        return self.process_fn(data)

    def __len__(self):
        return self.num_samples


class TestSignalHandler:
    """
    Test signal handling.
    """

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_segmentation_fault(self):
        """
        Test worker raises segmentation fault.
        """

        def segment_fault_fn(data):
            ctypes.memmove(0, b"crash", 5)
            return data

        dataset = SignalHandlingDataset(segment_fault_fn)
        data_loader = DataLoader(dataset, num_workers=4)
        try:
            for _ in data_loader:
                pass
        except RuntimeError:
            tb_info = traceback.format_exc()
            assert re.search(r"DataLoader worker \(pid\(s\): .*\) exited unexpectedly", tb_info)
            assert re.search(r"DataLoader worker \(pid: .*\) core dumped: Segmentation fault", tb_info)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_signal_abort(self):
        """
        Test worker raises abort.
        """

        def abort_fn(data):
            ctypes.CDLL(None).abort()
            return data

        dataset = SignalHandlingDataset(abort_fn)
        data_loader = DataLoader(dataset, num_workers=4)
        try:
            for _ in data_loader:
                pass
        except RuntimeError:
            tb_info = traceback.format_exc()
            assert re.search(r"DataLoader worker \(pid\(s\): .*\) exited unexpectedly", tb_info)
            assert re.search(r"DataLoader worker \(pid: .*\) core dumped: Aborted", tb_info)
