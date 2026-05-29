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
"""Sanity tests for MinimalTraceableTensor before using it in Cell tests."""
# pylint: disable=protected-access
import numpy as np

from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark
from tests.st.frontend.interface.cell.conftest import MinimalTraceableTensor


MINIMAL_TRACEABLE_TENSOR_MARK = arg_mark(plat_marks=['cpu_linux', 'platform_ascend'],
                                         level_mark='level0',
                                         card_mark='onecard',
                                         mem_peak=0.01)


class TestMinimalTraceableTensorSanity:
    """MinimalTraceableTensor behavior required by later Cell.to / to_empty tests."""

    @MINIMAL_TRACEABLE_TENSOR_MARK
    def test_basic_op_preserves_subclass(self):
        tensor = MinimalTraceableTensor(Tensor(np.ones((2, 2)), mstype.float32), payload="p")

        result = tensor + 1

        assert isinstance(result, MinimalTraceableTensor), f"result type: {type(result)}"
        assert result._payload_slot == "p", f"payload lost: {result._payload_slot}"
        np.testing.assert_allclose(result.asnumpy(), np.full((2, 2), 2.0))

    @MINIMAL_TRACEABLE_TENSOR_MARK
    def test_traceable_protocol_present(self):
        tensor = MinimalTraceableTensor(Tensor(np.ones((2, 2)), mstype.float32), payload="p")

        names, metadata = tensor.__tensor_flatten__()

        assert names == []
        assert metadata == {"payload": "p"}
        assert hasattr(tensor, "__tensor_unflatten__")

    @MINIMAL_TRACEABLE_TENSOR_MARK
    def test_new_from_existing_preserves_payload(self):
        tensor = MinimalTraceableTensor(Tensor(np.ones((2, 2)), mstype.float32), payload="original")

        tensor2 = MinimalTraceableTensor(tensor)

        assert tensor2._payload_slot == "original"

    @MINIMAL_TRACEABLE_TENSOR_MARK
    def test_parameter_wrap_preserves_subclass_and_slot(self):
        tensor = MinimalTraceableTensor(Tensor(np.ones((2, 2)), mstype.float32), payload="p")

        param = Parameter(tensor)

        assert isinstance(param, Parameter), f"not Parameter: {type(param)}"
        assert isinstance(param, MinimalTraceableTensor), f"lost MinimalTraceableTensor type: {type(param)}"
        assert param._payload_slot == "p"
